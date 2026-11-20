#include "engine/ops/qsa.h"


#include "engine/ops/ops.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/sequence.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/graph/window.h"
#include "engine/backends/cuda/executor/qsa/tq_cache.h"
#include "engine/state/kv/qsa/session.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/ops/qwen4_exp.h"
#include "engine/backends/cuda/ops/norm/launch.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/attention/kv/quantization.h"
#include "engine/backends/cuda/executor/kv/memory.h"
#include "engine/backends/cuda/ops/attention/kv/hot_cache.h"
#include "engine/generation/prefill/budget.h"
#include "engine/state/kv/offload.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <iostream>
#include <optional>

namespace tyloi::ops::detail {
using namespace ::tyloi::cuda;

class QsaCore {
    struct Projection;
  public:
    struct State {
        explicit State(const QsaCore& model)
            : keys(model.c_.maximum, model.c_.kv_heads * model.key_width(), model.cache_dtype()),
              values(model.c_.maximum, model.c_.kv_heads * model.value_width(), model.cache_dtype()),
              index(model.c_.maximum, model.c_.index_width)
              {
            const auto& c = model.c_;
            if (qsa_tq_cache_enabled()) {
                if (!(c.pool == 4 && !c.quantization.enabled()))
                    throw std::runtime_error("adaptive QSA requires four-token blocks and its own KV codec");
                ResidentAllocationScope resident;
                adaptive = std::make_unique<QsaTqCache>(c.maximum,c.heads,c.kv_heads,c.width,
                    c.index_heads,c.index_width,c.budget,model.w_->query_norm.options().device(c.device));
                return;
            }
            auto& executor = tyloi::cuda::Executor::current();
            auto manager = executor.kv_memory();
            if (manager)
                memory=manager->add(model.c_.maximum,model.c_.kv_heads,model.key_width(),model.value_width(),
                    model.c_.index_width,model.c_.pool,model.w_->query_norm.options().dtype(model.cache_dtype()),
                    model.c_.maximum<=model.c_.budget?model.c_.maximum:model.c_.budget+model.c_.pool-1);
            else offload = executor.kv_offload();
            if(offload) {
                offloaded_kv=std::make_unique<state::KvOffloadSequence>(offload,model.c_.kv_heads*(model.key_width()+model.value_width()),model.c_.pool,model.cache_dtype());
                offloaded_index=std::make_unique<state::KvOffloadSequence>(offload,model.c_.index_width,model.c_.pool);
                offloaded_pooled=std::make_unique<state::KvOffloadSequence>(offload,model.c_.index_width,16);
            }
        }
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = default;
        State& operator=(State&&) = default;
        SequenceCache keys,values,index;
        std::unique_ptr<QsaTqCache> adaptive;
        std::shared_ptr<state::KvOffloadStore> offload;
        std::unique_ptr<state::KvOffloadSequence> offloaded_kv,offloaded_index,offloaded_pooled;
        ::tyloi::Tensor graph_keys,graph_values,graph_pooled,graph_index;
        std::shared_ptr<KvMemoryAllocation> memory;
        bool graph_prepared=false;
    };
    QsaCore(std::shared_ptr<const QsaWeights> weights, QsaConfig config, std::shared_ptr<RotaryEmbedding> rotary,
        bool fused_projection = true, bool grouped_projection = true)
        : w_(std::move(weights)), c_(config), rotary_(std::move(rotary)), fused_projection_(fused_projection), grouped_projection_(grouped_projection)
          {
        if (!(c_.heads > 0 && c_.kv_heads > 0 && c_.heads % c_.kv_heads == 0 &&
                              c_.width > 0 && c_.index_heads > 0 && c_.pool > 0 && c_.budget > 0 &&
                              c_.budget % c_.pool == 0 && rotary_))
            throw std::runtime_error("invalid Qwen4 QSA configuration");
    }
    int64_t position(const State& request) const {
        if (request.adaptive) return request.adaptive->position();
        return request.offload && request.offloaded_kv ? request.offloaded_kv->position() : request.keys.position();
    }
    bool managed_memory(const State& request) const {return request.memory && request.memory->ready;}
    int64_t graph_capacity(const State& request) const {return managed_memory(request)?c_.maximum:0;}
    std::vector<::tyloi::Tensor*> graph_state(State& request) const {
        return {&request.graph_keys,&request.graph_values,&request.graph_pooled,&request.graph_index};
    }
    void activate_memory(State& request) {
        if(!managed_memory(request))return;
        if(!request.index.storage().defined())request.index.replace_storage(request.memory->index);
        request.graph_index=request.index.storage();
        request.graph_pooled=request.memory->pooled;
        if(request.memory->offload) {
            request.offload=request.memory->offload;
            if(!request.offloaded_kv)request.offloaded_kv=std::make_unique<state::KvOffloadSequence>(request.offload,
                c_.kv_heads*(key_width()+value_width()),c_.pool,cache_dtype());
        } else {request.graph_keys=request.memory->keys;request.graph_values=request.memory->values;}
    }
    std::vector<DenseTensor> project_inputs(const DenseTensor& x) const {
        std::vector<DenseTensor> projected;
        const bool grouped=tyloi::cuda::Executor::current().compute().attention_grouped.value_or(grouped_projection_) && ::tyloi::ops::grouped(w_->input_projection);
        if(grouped)
            { auto values=::tyloi::ops::project_group(x,w_->input_projection,1); for(const auto& value:values)projected.push_back(value.dense()); }
        else
            projected={project_query(x),
                ::tyloi::ops::project(x,w_->input_projection[1]).dense(),
                ::tyloi::ops::project(x,w_->input_projection[2]).dense(),
                ::tyloi::ops::project(x,w_->input_projection[3]).dense()};
        if(grouped && w_->has_gate) {
            if (!(projected.size()==5))
                throw std::runtime_error("QSA input projection count mismatch");
            const auto b=x.size(0),t=x.size(1);
            projected[0]=::tyloi::cuda::cat({projected[0].reshape({b,t,c_.heads,c_.width}),
                projected[4].reshape({b,t,c_.heads,c_.width})},-1).reshape({b,t,2*c_.heads*c_.width});
            projected.erase(projected.begin()+4);
        }
        if (!(projected.size()==4))
            throw std::runtime_error("QSA input projection count mismatch");
        return projected;
    }
    DenseTensor project_output(const DenseTensor& x) const {
        return ::tyloi::ops::project(x,w_->output).dense();
    }
    DenseTensor project_query(const DenseTensor& x) const {
        auto query=::tyloi::ops::project(x,w_->input_projection[0]).dense();
        if(!w_->has_gate)return query;
        const auto b=x.size(0),t=x.size(1);
        auto gate=::tyloi::ops::project(x,w_->input_projection[4]).dense();
        return ::tyloi::cuda::cat({query.reshape({b,t,c_.heads,c_.width}),
            gate.reshape({b,t,c_.heads,c_.width})},-1).reshape({b,t,2*c_.heads*c_.width});
    }
    std::size_t growth_bytes(const State& request,int64_t tokens) const {
        if(managed_memory(request))return request.memory->growth_bytes(position(request)+tokens);
        if (request.offload) return 0;
        const auto required = position(request) + tokens;
        if (!(tokens > 0 && required <= c_.maximum))
            throw std::runtime_error("QSA growth exceeds max_context");
        if (request.adaptive) return 0;
        std::size_t bytes = 0;
        for (const auto* cache : {&request.keys, &request.values, &request.index}) {
            const auto& storage = cache->storage();
            const int64_t current = request.graph_keys.defined() && cache != &request.index
                ? request.graph_keys.size(2) : (storage.defined() ? storage.size(1) : 0);
            if (required <= current) continue;
            int64_t capacity = current ? current : std::min<int64_t>(16, c_.maximum);
            while (capacity < required)
                capacity = std::min(c_.maximum, capacity > c_.maximum / 2 ? c_.maximum : capacity * 2);
            const auto width = cache == &request.index ? c_.index_width : c_.kv_heads * (cache == &request.keys ? key_width() : value_width());
            bytes += std::size_t(capacity) * width * (cache != &request.index && quantization_.enabled() ? 4 : 2);
        }
        return bytes;
    }
    /** Estimate the backing required for an append or a fresh reset. */
    std::size_t allocation_bytes(const State& request, int64_t batch, int64_t tokens, bool reset) const {
        const auto start = reset ? int64_t(0) : position(request);
        if (!(batch > 0 && tokens > 0 && start >= 0 && start <= c_.maximum &&
            tokens <= c_.maximum - start))
            throw std::runtime_error("invalid QSA allocation window");
        if (request.adaptive) {
            if (!(batch == 1))
                throw std::runtime_error("adaptive QSA needs one cache per sequence");
            return 0;
        }
        const auto add = [](std::size_t a, std::size_t b) {
            if (a > std::numeric_limits<std::size_t>::max() - b)
                throw std::overflow_error("QSA allocation size overflow");
            return a + b;
        };
        const auto multiply = [](std::size_t a, std::size_t b) {
            if (b && a > std::numeric_limits<std::size_t>::max() / b)
                throw std::overflow_error("QSA allocation size overflow");
            return a * b;
        };
        if (managed_memory(request)) return request.memory->growth_bytes(start + tokens);
        if (request.offload) return 0;
        if (!reset && !request.graph_keys.defined()) {
            return add(add(request.keys.growth(batch, tokens), request.values.growth(batch, tokens)),
                request.index.growth(batch, tokens));
        }
        if (!reset) return growth_bytes(request, tokens);
        auto capacity = std::min<int64_t>(16, c_.maximum);
        while (capacity < tokens)
            capacity = std::min(c_.maximum, capacity > c_.maximum / 2 ? c_.maximum : capacity * 2);
        const auto rows = multiply(std::size_t(batch), std::size_t(capacity));
        const auto key = multiply(std::size_t(c_.kv_heads), std::size_t(key_width()));
        const auto value = multiply(std::size_t(c_.kv_heads), std::size_t(value_width()));
        const auto scalar = quantization_.enabled() ? std::size_t(4) : std::size_t(2);
        const auto key_bytes = multiply(multiply(rows, key), scalar);
        const auto value_bytes = multiply(multiply(rows, value), scalar);
        const auto index_bytes = multiply(multiply(rows, std::size_t(c_.index_width)), 2);
        return add(add(key_bytes, value_bytes), index_bytes);
    }
    /** Estimate temporary backing needed when leaving an unmanaged graph. */
    std::size_t exit_growth(const State& request, int64_t tokens) const {
        const auto start = position(request);
        if (!(tokens > 0 && start >= 0 && start <= c_.maximum &&
            tokens <= c_.maximum - start))
            throw std::runtime_error("invalid QSA allocation window");
        if (!request.graph_keys.defined() || managed_memory(request)) return 0;
        const auto add = [](std::size_t a, std::size_t b) {
            if (a > std::numeric_limits<std::size_t>::max() - b)
                throw std::overflow_error("QSA allocation size overflow");
            return a + b;
        };
        const auto multiply = [](std::size_t a, std::size_t b) {
            if (b && a > std::numeric_limits<std::size_t>::max() / b)
                throw std::overflow_error("QSA allocation size overflow");
            return a * b;
        };
        const auto widths = add(std::size_t(key_width()), std::size_t(value_width()));
        const auto kv = multiply(multiply(std::size_t(c_.kv_heads), widths),
            quantization_.enabled() ? std::size_t(4) : std::size_t(2));
        const auto index = multiply(std::size_t(c_.index_width), request.index.storage().element_size());
        return multiply(multiply(std::size_t(start + tokens), add(kv, index)), 2);
    }
    std::size_t graph_exit_growth_bytes(const State& request) const {
        if (!request.graph_keys.defined() || managed_memory(request)) return 0;
        const auto multiply = [](std::size_t a, std::size_t b) {
            if (b && a > std::numeric_limits<std::size_t>::max() / b)
                throw std::overflow_error("QSA graph exit size overflow");
            return a * b;
        };
        const auto key = multiply(std::size_t(key_width()), request.graph_keys.element_size());
        const auto value = multiply(std::size_t(value_width()), request.graph_values.element_size());
        if (key > std::numeric_limits<std::size_t>::max() - value)
            throw std::overflow_error("QSA graph exit size overflow");
        return multiply(multiply(multiply(std::size_t(request.graph_keys.size(0)),
            std::size_t(position(request))), std::size_t(c_.kv_heads)), key + value);
    }
    std::size_t graph_growth_bytes(const State& request,int64_t capacity) const {
        if(request.adaptive)return 0;
        if(managed_memory(request))return request.memory->growth_bytes(capacity);
        if (request.graph_keys.defined()) return 0;
        auto bytes = std::size_t(capacity) * c_.kv_heads * (key_width()+value_width()) * (quantization_.enabled()?4:2);
        if (!request.index.storage().defined() || request.index.storage().size(1) != capacity)
            bytes += std::size_t(capacity) * c_.index_width * 2;
        bytes += std::size_t((capacity+c_.pool-1)/c_.pool) * c_.index_width * 2;
        return bytes;
    }
    bool supports_session(const State& request) const {
        return request.adaptive ? request.adaptive->supports_session() :
            !request.offload && !(request.memory && request.memory->offload);
    }
    /** Validate a QSA snapshot against this layer's cache geometry and device. */
    void validate_session(const State& request, const ::tyloi::ops::Qsa::Snapshot& state,
                          int64_t count, tyloi::Device device) const {
        if(request.adaptive) {
            if (!(state.ram && !state.keys.defined() && !state.values.defined() && !state.index.defined() &&
                device==c_.device))
                throw std::runtime_error("incompatible Qwen4 QSA RAM session state");
            request.adaptive->validate_session(*state.ram,count);
            return;
        }
        if (!(!state.ram))
            throw std::runtime_error("incompatible Qwen4 QSA session state");
        const auto check = [&](const tyloi::Tensor& tensor, int64_t width, tyloi::ScalarType type) {
            return tensor.defined() && tensor.representation<DenseTensor>() && tensor.device()==device &&
                tensor.scalar_type()==type && tensor.dim()==3 && tensor.size(0)==1 &&
                tensor.size(1)==count && tensor.size(2)==width;
        };
        if (!(supports_session(request) && count>0 && count<=c_.maximum &&
            check(state.keys,c_.kv_heads*key_width(),cache_dtype()) &&
            check(state.values,c_.kv_heads*value_width(),cache_dtype()) &&
            check(state.index,c_.index_width,::tyloi::kFloat16)))
            throw std::runtime_error("incompatible Qwen4 QSA session state");
    }
    /** Estimate fresh unmanaged sequence backing; managed allocation has its own admission. */
    std::size_t fresh_session_growth(const State& current, int64_t count) const {
        if (current.adaptive) return current.adaptive->fresh_session_growth();
        if (current.memory && !current.offload) {
            if (!managed_memory(current))
                throw std::runtime_error("Qwen4 managed session storage is not ready");
            return 0;
        }
        SequenceCache keys(c_.maximum,c_.kv_heads*key_width(),cache_dtype());
        SequenceCache values(c_.maximum,c_.kv_heads*value_width(),cache_dtype());
        SequenceCache index(c_.maximum,c_.index_width);
        const auto a=keys.growth(1,count), b=values.growth(1,count), c=index.growth(1,count);
        if (a>std::numeric_limits<std::size_t>::max()-b || a+b>std::numeric_limits<std::size_t>::max()-c)
            throw std::overflow_error("Qwen4 session allocation planning overflow");
        return a+b+c;
    }
    /** Stage a snapshot into independently owned sequence or managed KV storage. */
    void stage_session(State& request, const ::tyloi::ops::Qsa::Snapshot& saved,
                       int64_t count, tyloi::Device device) {
        if (request.adaptive) {
            if (!(device==c_.device))
                throw std::runtime_error("QSA restore target differs from layer device");
            restore_session(request,saved,count);
            return;
        }
        if (managed_memory(request)) { restore_session(request,saved,count); return; }
        const auto stage = [&](SequenceCache& cache, const ::tyloi::Tensor& source) {
            int64_t capacity=std::min<int64_t>(16,c_.maximum);
            while(capacity<count)
                capacity=std::min(c_.maximum,capacity>c_.maximum/2?c_.maximum:capacity*2);
            auto shape=source.dense().sizes().vec(); shape[1]=capacity;
            cache.replace_storage(tyloi::ops::empty(shape,source.options().device(device)));
            auto view=tyloi::ops::slice(cache.storage(),1,0,count);
            tyloi::ops::copy_into(view,source);
            cache.advance_fixed(count);
        };
        stage(request.keys,saved.keys);
        stage(request.values,saved.values);
        stage(request.index,saved.index);
    }
    std::size_t session_bytes(const State& request,int64_t count) const {
        if(count<=0)return 0;
        if(request.adaptive)return request.adaptive->session_bytes(count);
        return std::size_t(count)*(c_.kv_heads*(key_width()+value_width())*(quantization_.enabled()?4:2)+
            c_.index_width*request.index.storage().element_size());
    }
    void capture_session(const State& request,::tyloi::ops::Qsa::Snapshot& state,int64_t count,std::size_t& bytes) const {
        if (!(supports_session(request) && count>0 && count<=position(request)))
            throw std::runtime_error("invalid Qwen4 QSA session boundary");
        if(request.adaptive) {
            state.keys={};state.values={};state.index={};
            state.ram=request.adaptive->capture_session(count);
            const auto captured=state.ram->resident_bytes();
            if(bytes>std::numeric_limits<std::size_t>::max()-captured)throw std::overflow_error("QSA session size overflow");
            bytes+=captured;
            return;
        }
        state.ram.reset();
        const auto save=[&](const DenseTensor& tensor) {
            auto saved=std::make_shared<DenseTensor>(tensor.contiguous().clone());bytes+=std::size_t(saved->numel())*saved->element_size();return saved;
        };
        if(managed_memory(request) && request.memory->hot.exclusive()) {
            auto pair=request.memory->hot.gather(count);
            state.keys=save(pair.first.permute({0,2,1,3}).reshape({1,count,c_.kv_heads*key_width()}));
            state.values=save(pair.second.permute({0,2,1,3}).reshape({1,count,c_.kv_heads*value_width()}));
        } else if(request.graph_keys.defined()) {
            state.keys=save((request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{}).narrow(2,0,count).permute({0,2,1,3}).reshape({1,count,c_.kv_heads*key_width()}));
            state.values=save((request.graph_values.defined()?request.graph_values.dense():DenseTensor{}).narrow(2,0,count).permute({0,2,1,3}).reshape({1,count,c_.kv_heads*value_width()}));
        } else {
            state.keys=save(request.keys.storage().dense().narrow(1,0,count));
            state.values=save(request.values.storage().dense().narrow(1,0,count));
        }
        state.index=save(request.index.storage().dense().narrow(1,0,count));
    }
    void restore_session(State& request,const ::tyloi::ops::Qsa::Snapshot& state,int64_t count) {
        if(request.adaptive) {
            validate_session(request,state,count,c_.device);
            ResidentAllocationScope resident;
            request.adaptive->restore_session(*state.ram,false);
            return;
        }
        const auto* keys=state.keys.representation<DenseTensor>();
        const auto* values=state.values.representation<DenseTensor>();
        const auto* index=state.index.representation<DenseTensor>();
        if (!(!request.adaptive && !request.offload && keys && values && index && keys->defined() && values->defined() && index->defined() &&
            keys->dim()==3 && keys->size(0)==1 && keys->size(1)==count &&
            keys->size(2)==c_.kv_heads*key_width() && keys->scalar_type()==cache_dtype() &&
            values->dim()==3 && values->size(0)==1 && values->size(1)==count &&
            values->size(2)==c_.kv_heads*value_width() && values->scalar_type()==cache_dtype() &&
            index->dim()==3 && index->size(0)==1 && index->size(1)==count && index->size(2)==c_.index_width))
            throw std::runtime_error("incompatible Qwen4 QSA session state");
        reset(request);activate_memory(request);
        if(managed_memory(request)) {
            request.memory->ensure(count);
            auto k=keys->to(request.graph_keys.device()).reshape({1,count,c_.kv_heads,key_width()}).permute({0,2,1,3});
            auto v=values->to(request.graph_values.device()).reshape({1,count,c_.kv_heads,value_width()}).permute({0,2,1,3});
            auto positions=::tyloi::cuda::arange(count,request.graph_keys.dense().options().dtype(::tyloi::kInt64));
            if(request.memory->hot.exclusive())request.memory->hot.append(k,v,positions);
            else {
                (request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{}).narrow(2,0,count).copy_(k);(request.graph_values.defined()?request.graph_values.dense():DenseTensor{}).narrow(2,0,count).copy_(v);
                request.memory->hot.publish((request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{}),(request.graph_values.defined()?request.graph_values.dense():DenseTensor{}),positions);
            }
            request.keys.advance_fixed(count);request.values.advance_fixed(count);request.index.append(index->to(request.graph_index.device()));request.graph_prepared=false;
        } else {request.keys.append(*keys);request.values.append(*values);request.index.append(*index);}
    }
    /** Return owning handles for the request's unmanaged resident cache. */
    std::vector<::tyloi::Tensor> resident_cache_tensors(const State& request) const {
        if (request.adaptive) {
            std::vector<::tyloi::Tensor> result;
            for (auto& value : request.adaptive->resident_tensors())
                if (value.defined()) result.emplace_back(std::move(value));
            return result;
        }
        if(managed_memory(request))return {}; // Count physical GPU/RAM tiers through KvMemoryManager.
        return {request.keys.storage(), request.values.storage(), request.index.storage(),
            request.graph_keys, request.graph_values, request.graph_pooled};
    }
    void leave_graph(State& request) {
        if(managed_memory(request)){request.graph_prepared=false;return;}
        if(!request.graph_keys.defined())return;
        const auto n=position(request);
        request.keys.replace_storage((request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{}).narrow(2,0,n).permute({0,2,1,3}).contiguous().reshape({1,n,c_.kv_heads*key_width()}));
        request.values.replace_storage((request.graph_values.defined()?request.graph_values.dense():DenseTensor{}).narrow(2,0,n).permute({0,2,1,3}).contiguous().reshape({1,n,c_.kv_heads*value_width()}));
        request.graph_keys={};request.graph_values={};request.graph_pooled={};request.graph_index={};
    }
    void reset(State& request) {
        if (request.adaptive) { request.adaptive->reset(); return; }
        if (request.offload) {
            if(request.offloaded_kv)request.offloaded_kv->reset();
            if(request.offloaded_index)request.offloaded_index->reset();
            if(request.offloaded_pooled)request.offloaded_pooled->reset();
        }
        request.keys.reset();
        request.values.reset();
        request.index.reset();
        request.graph_prepared=false;
        if(managed_memory(request)){request.memory->hot.reset();activate_memory(request);return;}
        request.graph_keys={};request.graph_values={};request.graph_pooled={};request.graph_index={};
    }
    void prepare_graph(State& request,const DenseTensor& full_positions, int64_t capacity,int64_t pending_tokens) {
        if(request.adaptive) {
            if (!(pending_tokens>0 && request.adaptive->pending_supported()))
                throw std::runtime_error("adaptive QSA requires an admitted pending graph");
            request.adaptive->prepare_decode_tokens(pending_tokens,true,0,full_positions);
            return;
        }
        activate_memory(request);
        if (!(!request.offload || managed_memory(request)))
            throw std::runtime_error("streamed QSA graphs require a resident indexer");
        if(managed_memory(request)) {
            if(request.graph_prepared)return;
            const auto n=position(request),pools=n/c_.pool;
            request.memory->ensure(n);
            if(pools) {
                auto pooled=request.index.storage().dense().narrow(1,0,pools*c_.pool).reshape({1,pools,c_.pool,c_.index_width}).to(::tyloi::kFloat32).mean(2).to(request.index.storage().scalar_type());
                pooled=::tyloi::cuda::attention_ops::rms_norm(pooled,w_->index_key_norm.dense().to(::tyloi::kFloat32)+1,c_.eps);
                auto starts=::tyloi::cuda::arange(pools,full_positions.options().dtype(::tyloi::kInt64))*c_.pool;
                (request.graph_pooled.defined()?request.graph_pooled.dense():DenseTensor{}).narrow(1,0,pools).copy_(rotary_->forward(pooled.unsqueeze(1),full_positions.index_select(-1,starts)).squeeze(1));
            }
            request.graph_prepared=true;return;
        }
        if(request.graph_keys.defined())return;
        if (!(position(request)>0 && request.keys.storage().size(0)==1))
            throw std::runtime_error("QSA graph requires a prefilled single sequence");
        const auto n=position(request);
        if (!(capacity>=n && capacity<=c_.maximum))
            throw std::runtime_error("invalid QSA graph capacity");
        auto graph_keys=::tyloi::cuda::zeros({1,c_.kv_heads,capacity,key_width()},request.keys.storage().options());
        auto graph_values=::tyloi::cuda::zeros({1,c_.kv_heads,capacity,value_width()},request.values.storage().options());
        graph_keys.narrow(2,0,n).copy_(request.keys.storage().dense().narrow(1,0,n).reshape({1,n,c_.kv_heads,key_width()}).permute({0,2,1,3}));
        graph_values.narrow(2,0,n).copy_(request.values.storage().dense().narrow(1,0,n).reshape({1,n,c_.kv_heads,value_width()}).permute({0,2,1,3}));
        request.index.prepare_fixed(capacity);
        request.graph_index=request.index.storage();
        auto graph_pooled=::tyloi::cuda::zeros({1,(capacity+c_.pool-1)/c_.pool,c_.index_width},request.index.storage().options());
        const auto pools=n/c_.pool;
        if(pools) {
            auto pooled=request.index.storage().dense().narrow(1,0,pools*c_.pool).reshape({1,pools,c_.pool,c_.index_width}).to(::tyloi::kFloat32).mean(2).to(request.index.storage().scalar_type());
            pooled=::tyloi::cuda::attention_ops::rms_norm(pooled,w_->index_key_norm.dense().to(::tyloi::kFloat32)+1,c_.eps);
            auto starts=::tyloi::cuda::arange(pools,full_positions.options().dtype(::tyloi::kInt64))*c_.pool;
            graph_pooled.narrow(1,0,pools).copy_(rotary_->forward(pooled.unsqueeze(1),full_positions.index_select(-1,starts)).squeeze(1));
        }
        // Commit only after all allocation/copy submissions succeed. Replay
        // reads graph KV; retaining the old layout duplicates the full prefix
        // at every growth boundary. leave_graph recreates it when needed.
        request.graph_keys=std::move(graph_keys);request.graph_values=std::move(graph_values);request.graph_pooled=std::move(graph_pooled);
        request.keys.release_storage();request.values.release_storage();
    }
    DenseTensor forward_graph(State& request,const DenseTensor& hidden,
                         const DenseTensor& positions,const DenseTensor& cache_positions,
                         std::vector<DenseTensor> projections,bool project_output,bool history_only,bool query_tail,
                         const Projection* prepared=nullptr,const DenseTensor& pool_positions={}) {
        if (!(!prepared || (!history_only && !query_tail && hidden.size(0)==1)))
            throw std::runtime_error("QSA prepared graph projection requires a complete independent row");
        if(request.adaptive) {
            if (!(hidden.size(0)==1 && request.adaptive->pending_graph_enabled()))
                throw std::runtime_error("adaptive QSA requires a complete pending graph window");
            const auto tokens=hidden.size(1);
            Projection p;
            if(prepared) p=*prepared;
            else if(history_only) p=project_history(hidden,positions);
            else if(query_tail && tokens>1) {
                const auto prefix=tokens-1;
                auto history=project_history(hidden.narrow(1,0,prefix),positions.narrow(-1,0,prefix).contiguous());
                p=project(hidden.narrow(1,prefix,1),positions.narrow(-1,prefix,1).contiguous());
                p.key=::tyloi::cuda::cat({history.key,p.key},1);
                p.value=::tyloi::cuda::cat({history.value,p.value},1);
                p.raw=::tyloi::cuda::cat({history.raw,p.raw},1);
            }else p=project(hidden,positions,true,{},{},{},std::move(projections));
            const auto rows=query_tail?1:tokens;
            auto output=history_only ? DenseTensor{} : ::tyloi::cuda::empty({1,rows,c_.heads*c_.width},hidden.options().dtype(
                hidden.scalar_type()==::tyloi::kFloat16 ? ::tyloi::kFloat16 : ::tyloi::kFloat32));
            output=request.adaptive->forward_pending_graph(p.query,p.key,p.value,p.iq,p.raw,cache_positions,p.gate,
                std::move(output),[this](const DenseTensor& raw,const DenseTensor& starts) {
                    return pool_index(raw,starts);
                },positions);
            if(history_only)return {};
            output=output.to(hidden.scalar_type());
            return project_output?this->project_output(output):output;
        }
        if (!(!query_tail))
            throw std::runtime_error("ordinary QSA requires separate graph history and tail writes");
        const auto mark=[](const char* phase) {
            if(auto* window=DecodeWindow::recording())window->mark_current(phase);
        };
        const bool streamed=bool(request.offload);
        if (!((request.graph_keys.defined() || (streamed && managed_memory(request))) && hidden.size(0)==1 && cache_positions.dim()==1))
            throw std::runtime_error("QSA graph cache is not prepared");
        auto graph_keys = (request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{});
        auto graph_values = (request.graph_values.defined()?request.graph_values.dense():DenseTensor{});
        auto graph_pooled = (request.graph_pooled.defined()?request.graph_pooled.dense():DenseTensor{});
        const auto t=hidden.size(1),pools=request.graph_pooled.size(1);
        const bool dense_capacity=(streamed?c_.maximum:request.graph_keys.size(2))<=c_.budget;
        const bool exclusive=managed_memory(request) && request.memory->hot.exclusive();
        auto p=prepared?*prepared:history_only?project_history(hidden,positions):
            project(hidden,positions,!dense_capacity,cache_positions,
                       quantization_.enabled() || exclusive || streamed?DenseTensor{}:graph_keys,quantization_.enabled() || exclusive || streamed?DenseTensor{}:graph_values,
                       std::move(projections));
        mark("qsa_input_end");
        const auto finish=[&](DenseTensor output) {
            mark("qsa_attention_end");
            output=output.reshape({1,t,c_.heads*c_.width});
            return project_output?this->project_output(output):output;
        };
        bool index_written=false;
        const auto* prepare_option=tyloi::cuda::env_lookup("TYLOINFER_KV_PREPARE_FUSED");
        const bool begin_prepare=!prepare_option || prepare_option[0]!='0';
        if(exclusive) {
            const auto keys=(quantization_.enabled()?encode_key(p.key):p.key).permute({0,2,1,3});
            const auto values=(quantization_.enabled()?encode_value(p.value):p.value).permute({0,2,1,3});
            index_written=request.memory->hot.append_decode(keys,values,cache_positions,p.raw,request.index.storage().dense(),begin_prepare);
            if(!index_written)request.memory->hot.append(keys.to(cache_dtype()),values.to(cache_dtype()),cache_positions);
        }
        else if (!p.cache_written && !streamed) {
            graph_keys.index_copy_(2,cache_positions,encode_key(p.key).permute({0,2,1,3}));
            graph_values.index_copy_(2,cache_positions,encode_value(p.value).permute({0,2,1,3}));
        }
        if(managed_memory(request) && !exclusive && !streamed)request.memory->hot.publish(graph_keys,graph_values,cache_positions);
        if(!index_written)request.index.append_fixed(p.raw,cache_positions);
        mark("qsa_cache_end");
        if(history_only && streamed) {
            auto* window=DecodeWindow::recording();
            if (!window)
                throw std::runtime_error("streamed QSA history requires a decode window");
            struct HistoryStage {DenseTensor raw;};
            const auto stage=window->workspace<HistoryStage>(&request,[&] {
                ResidentAllocationScope resident;
                auto value=std::make_shared<HistoryStage>();
                value->raw=::tyloi::cuda::empty({t,c_.kv_heads*(key_width()+value_width())},hidden.options().dtype(cache_dtype()));
                return value;
            });
            stage->raw.copy_(::tyloi::cuda::cat({encode_key(p.key).reshape({t,c_.kv_heads*key_width()}),
                encode_value(p.value).reshape({t,c_.kv_heads*value_width()})},1));
            window->host_barrier([this,&request,stage] {request.offloaded_kv->append(stage->raw);});
        }
        if(dense_capacity) {
            if(history_only)return {};
            if(streamed) {
                auto ids=::tyloi::cuda::arange(c_.maximum,cache_positions.options()).reshape({1,1,c_.maximum}).expand({1,t,c_.maximum});
                ids=::tyloi::cuda::where(ids<=cache_positions.reshape({1,t,1}),ids,::tyloi::cuda::full_like(ids,-1)).to(::tyloi::kInt32).contiguous();
                return finish(streamed_graph_attention(request,p,ids,hidden.scalar_type()));
            }
            const auto columns=std::min(request.graph_keys.size(2),c_.budget+c_.pool-1);
            const auto hot=prepare_hot(request,{},cache_positions,columns,index_written && begin_prepare);
            if(quantization_.enabled()) {
                auto attended=cuda_kv_attention(p.query,graph_keys,graph_values,quantization_,{},cache_positions,0,{},0,{},{},0,hot);
                return finish((attended*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32))).to(hidden.scalar_type()));
            }
            if(fused_projection_ && rotary_->fused() &&
                (hidden.scalar_type()==::tyloi::kFloat16 || hidden.scalar_type()==::tyloi::kFloat32)) {
                auto gated=mfq_qwen4_exp::causal_gqa_attention_gate(p.query,graph_keys,graph_values,
                    cache_positions,p.gate,columns,hidden.scalar_type()==::tyloi::kFloat16,hot);
                return finish(std::move(gated));
            }
            auto absolute=cache_positions.reshape({1,t,1});
            auto indices=::tyloi::cuda::arange(columns,cache_positions.options()).reshape({1,1,columns}).expand({1,t,columns});
            indices=::tyloi::cuda::where(indices<=absolute,indices,::tyloi::cuda::full_like(indices,-1)).to(::tyloi::kInt32).contiguous();
            auto attended=mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys,graph_values,indices,hot);
            auto gated=attended.to(::tyloi::kFloat32)*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32));
            return finish(gated.to(hidden.scalar_type()));
        }
        DenseTensor block,pooled;
        // The existing pooling kernels derive the block start from a token's offset.
        const auto index_positions=pool_positions.defined() ? pool_positions+cache_positions.remainder(c_.pool) : positions;
        const auto* pool_fusion=tyloi::cuda::env_lookup("TYLOINFER_QSA_POOL_FUSED");
        const auto* pool_write=tyloi::cuda::env_lookup("TYLOINFER_QSA_POOL_WRITE_FUSED");
        const bool pool_written=(!pool_fusion || pool_fusion[0]!='0') && (!pool_write || pool_write[0]!='0') &&
            request.index.storage().size(1)>=c_.maximum && rotary_->pool_normalized_into(request.index.storage().dense(),w_->index_key_norm.dense(),
                index_positions,cache_positions,graph_pooled,c_.pool,c_.maximum,c_.eps);
        if(!pool_written && rotary_->fused() && (!pool_fusion || pool_fusion[0]!='0') && request.index.storage().size(1)>=c_.maximum) {
            auto prepared=mfq_qwen4_exp::live_index_pool(request.index.storage().dense(),index_positions,cache_positions,c_.pool,c_.maximum);
            block=prepared[1];
            pooled=rotary_->forward_normalized(prepared[0].unsqueeze(2),w_->index_key_norm.dense(),prepared[2],c_.eps).squeeze(1);
        }else if(!pool_written) {
            block=(cache_positions/c_.pool).to(::tyloi::kInt64);
            auto row=(block.unsqueeze(-1)*c_.pool+::tyloi::cuda::arange(c_.pool,block.options())).clamp(0,c_.maximum-1);
            pooled=request.index.storage().dense().index_select(1,row.reshape({-1})).reshape({1,t,c_.pool,c_.index_width}).to(::tyloi::kFloat32).mean(2).to(request.index.storage().scalar_type());
            pooled=::tyloi::cuda::attention_ops::rms_norm(pooled,w_->index_key_norm.dense().to(::tyloi::kFloat32)+1,c_.eps);
            auto delta=index_positions-cache_positions;
            auto block_positions=block*c_.pool+delta;
            pooled=rotary_->forward(pooled.unsqueeze(1),block_positions).squeeze(1);
        }
        if(!pool_written)graph_pooled.index_copy_(1,block,pooled);
        mark("qsa_pool_end");
        if(history_only)return {};
        auto ranked=mfq_qwen4_exp::live_block_scores(p.iq,graph_pooled,cache_positions,c_.pool);
        mark("qsa_scores_end");
        auto absolute=cache_positions.reshape({1,t,1});
        const auto count=std::min(c_.budget/c_.pool,pools),columns=c_.budget+c_.pool-1;
        auto indices=mfq_qwen4_exp::live_topk_blocks(ranked,cache_positions,c_.pool,count);
        mark("qsa_topk_end");
        DenseTensor selected;
        const auto fused_selection=::tyloi::cuda::options::qsa_select_fused();
        if(!fused_selection || *fused_selection) {
            selected=mfq_qwen4_exp::qsa_selected_tokens(indices,cache_positions,c_.pool,c_.budget);
        } else {
        // Live top-k pads inactive slots with an out-of-range block ID.
        // Check positions directly rather than gathering a visibility table
        // with those sentinels when the unfused selector is requested.
        auto valid=((indices>=0)&(indices*c_.pool+c_.pool-1<=absolute)).unsqueeze(-1).expand({1,t,count,c_.pool});
        auto expanded=indices.unsqueeze(-1)*c_.pool+::tyloi::cuda::arange(c_.pool,indices.options());
        expanded=::tyloi::cuda::where(valid,expanded,::tyloi::cuda::full_like(expanded,-1));
        selected=::tyloi::cuda::full({1,t,columns},-1,cache_positions.options());
        selected.narrow(-1,0,count*c_.pool).copy_(expanded.reshape({1,t,count*c_.pool}));
        if(c_.pool>1) {
            auto tail_count=(absolute+1).remainder(c_.pool);
            auto offsets=::tyloi::cuda::arange(c_.pool-1,cache_positions.options()).reshape({1,1,c_.pool-1});
            auto tail=absolute+1-tail_count+offsets;
            selected.narrow(-1,c_.budget,c_.pool-1).copy_(::tyloi::cuda::where(offsets<tail_count,tail,::tyloi::cuda::full_like(tail,-1)));
        }
        // Dense-budget queries select their complete causal prefix. The same
        // native selected-attention kernel serves both paths inside one graph.
        auto dense=::tyloi::cuda::arange(columns,cache_positions.options()).reshape({1,1,columns}).expand({1,t,columns});
        dense=::tyloi::cuda::where(dense<=absolute,dense,::tyloi::cuda::full_like(dense,-1));
        selected=::tyloi::cuda::where(absolute+1<=c_.budget,dense,selected).to(::tyloi::kInt32).contiguous();
        }
        mark("qsa_select_end");
        if(streamed)return finish(streamed_graph_attention(request,p,selected,hidden.scalar_type()));
        DenseTensor gated;
        const auto hot=prepare_hot(request,selected,cache_positions,0,index_written && begin_prepare);
        mark("qsa_hot_end");
        if(quantization_.enabled()) {
            auto attended=cuda_kv_attention(p.query,graph_keys,graph_values,quantization_,selected,cache_positions,0,{},0,{},{},0,hot);
            gated=(attended*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32))).to(hidden.scalar_type());
        } else if(fused_projection_ && rotary_->fused() &&
            (hidden.scalar_type()==::tyloi::kFloat16 || hidden.scalar_type()==::tyloi::kFloat32)) {
            const auto sparse_gate=::tyloi::cuda::options::qsa_sparse_gate_fused();
            if(!sparse_gate || *sparse_gate)
                gated=mfq_qwen4_exp::sparse_gqa_attention_gate(p.query,graph_keys,graph_values,selected,
                    p.gate,hidden.scalar_type()==::tyloi::kFloat16,hot);
            else gated=mfq_qwen4_exp::attention_gate(
                mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys,graph_values,selected,hot),
                p.gate,hidden.scalar_type()==::tyloi::kFloat16);
        } else {
            auto attended=mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys,graph_values,selected,hot);
            gated=(attended.to(::tyloi::kFloat32)*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32))).to(hidden.scalar_type());
        }
        return finish(std::move(gated));
    }
    struct GraphRow { State* state; DenseTensor positions, cache_positions; };
    DenseTensor forward_independent_graph(const DenseTensor& hidden, std::span<const GraphRow> rows) {
        if (!(hidden.dim()==3 && hidden.size(0)>0 && hidden.size(1)>0 &&
            rows.size()==std::size_t(hidden.size(0))))
            throw std::runtime_error("QSA independent graph batch mismatch");
        std::vector<DenseTensor> positions;
        positions.reserve(rows.size());
        for (const auto& row : rows) {
            if (!(row.state && row.positions.defined() && row.positions.dim()==1 &&
                row.positions.size(0)==hidden.size(1) && row.positions.device()==hidden.device() &&
                (row.positions.scalar_type()==::tyloi::kInt32 || row.positions.scalar_type()==::tyloi::kInt64) &&
                row.cache_positions.defined() && row.cache_positions.dim()==1 &&
                row.cache_positions.size(0)==hidden.size(1) && row.cache_positions.device()==hidden.device() &&
                row.cache_positions.scalar_type()==::tyloi::kInt64))
                throw std::runtime_error("QSA independent graph row geometry mismatch");
            if (!positions.empty()) {
                if (!(row.positions.scalar_type()==positions.front().scalar_type()))
                    throw std::runtime_error("QSA independent graph position types differ");
            }
            positions.push_back(row.positions);
        }
        auto axes=::tyloi::cuda::cat(positions,0).reshape({1,hidden.size(0),hidden.size(1)});
        auto all=project(hidden,axes);
        if(auto* window=DecodeWindow::recording())window->mark_current("qsa_projection_end");
        const auto slice=[](const DenseTensor& value,int64_t index) {
            return value.defined()?value.narrow(0,index,1):DenseTensor{};
        };
        std::vector<DenseTensor> outputs;
        outputs.reserve(rows.size());
        for (std::size_t i=0;i<rows.size();++i) {
            const auto index=static_cast<int64_t>(i);
            Projection projected{slice(all.gate,index),slice(all.query,index),slice(all.key,index),
                slice(all.value,index),slice(all.iq,index),slice(all.raw,index),false};
            outputs.push_back(forward_graph(*rows[i].state,hidden.narrow(0,index,1),rows[i].positions,
                rows[i].cache_positions,{},false,false,false,&projected));
        }
        return outputs.size()==1 ? outputs.front() : ::tyloi::cuda::cat(outputs,0);
    }
    void advance_graph(State& request,int64_t tokens,int64_t attention_rows) {
        if(request.adaptive) {
            MfqCudaGuard guard(c_.device);
            request.adaptive->advance_graph(tokens,attention_rows);
            return;
        }
        if(!request.offload){request.keys.advance_fixed(tokens);request.values.advance_fixed(tokens);}
        request.index.advance_fixed(tokens);
    }
    void truncate(State& request,int64_t keep) {
        if (request.adaptive) {
            MfqCudaGuard guard(c_.device);
            request.adaptive->truncate(keep);
            return;
        }
        if (request.offload) {
            if (!(keep >= 0 && keep <= position(request)))
                throw std::runtime_error("invalid offloaded QSA truncation");
            request.offloaded_kv->truncate(keep);
            if(managed_memory(request))request.index.truncate(keep);
            else {request.offloaded_index->truncate(keep);request.offloaded_pooled->truncate(keep/c_.pool);}
            return;
        }
        if (!(keep >= 0 && keep <= request.keys.position() && keep <= request.values.position() &&
                              keep <= request.index.position()))
            throw std::runtime_error("invalid Qwen4 QSA cache truncation");
        request.keys.truncate(keep);
        request.values.truncate(keep);
        request.index.truncate(keep);
    }
    DenseTensor forward(State& request, const DenseTensor &hidden,
                   const DenseTensor &current_positions, const DenseTensor &full_positions, bool use_cache,
                   std::vector<DenseTensor> *selection_trace = nullptr,
                   std::vector<DenseTensor> projections = {}, bool project_output = true,
                   bool history_only = false) {
        if (!(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0 &&
                              hidden.size(1) > 0))
            throw std::runtime_error("Qwen4 QSA requires nonempty [B,T,H] input");
        if (!(!history_only || use_cache))
            throw std::runtime_error("QSA history priming requires a cache");
        if (request.adaptive && use_cache) {
            if (!(!selection_trace))
                throw std::runtime_error("adaptive QSA does not expose a selection trace");
            return forward_adaptive(*request.adaptive,hidden,current_positions,full_positions,projections,
                project_output,history_only);
        }
        activate_memory(request);
        auto graph_keys = (request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{});
        auto graph_values = (request.graph_values.defined()?request.graph_values.dense():DenseTensor{});
        if(use_cache && hidden.size(1)>8 && !selection_trace && !history_only) {
            // Index scores grow with both query rows and the accumulated
            // context. Bound that workspace independently of the FFN tile.
            // Keep per-head products, reduction and top-k scratch below the
            // same budget even at the final chunk of a long prompt.
            const auto offset=position(request),length=offset+hidden.size(1);
            const auto pools=std::max<int64_t>(1,length/c_.pool);
            const auto row_bytes=std::size_t(hidden.size(0))*pools*(c_.index_heads+4)*sizeof(float);
            const auto rows=std::max<int64_t>(32,int64_t((128ULL<<20)/row_bytes)/32*32);
            if(hidden.size(1)>rows) {
                std::vector<DenseTensor> outputs;
                for(int64_t begin=0;begin<hidden.size(1);begin+=rows) {
                    const auto count=std::min(rows,hidden.size(1)-begin);
                    std::vector<DenseTensor> projected;
                    for(const auto& value:projections)projected.push_back(value.narrow(1,begin,count));
                    outputs.push_back(forward(request,hidden.narrow(1,begin,count),
                        current_positions.narrow(-1,begin,count),full_positions.narrow(-1,0,offset+begin+count),
                        true,nullptr,std::move(projected),project_output));
                }
                return ::tyloi::cuda::cat(outputs,1);
            }
        }
        if (request.offload && use_cache) {
            if(managed_memory(request))request.memory->ensure(position(request)+hidden.size(1));
            return forward_offloaded(request, hidden, current_positions, full_positions, selection_trace,
                                     std::move(projections),project_output,history_only);
        }
        const auto b = hidden.size(0), t = hidden.size(1), offset = use_cache ? position(request) : 0;
        const auto prefill_option=::tyloi::cuda::options::qsa_prefill_fused();
        // Scalar decode and verification use the same grouped projections and
        // normalization/RoPE preparation as captured execution. Keeping short
        // eager calls on separate norms introduces extra FP16 rounding steps.
        const bool prepared_inputs=!projections.empty() || (!prefill_option || *prefill_option!=0);
        if (!(t <= c_.maximum - offset && full_positions.size(-1) == offset + t &&
                (!use_cache || (offset == request.values.position() && offset == request.index.position()))))
            throw std::runtime_error("Qwen4 QSA cache/position geometry mismatch");
        auto project_inputs = [&] {
                if(prepared_inputs)
                    return project(hidden,current_positions,!history_only && offset+t>c_.budget,{},{},{},std::move(projections));
                auto pair = this->project_query(hidden).reshape({b, t, c_.heads, 2 * c_.width});
                auto gate = pair.narrow(-1, c_.width, c_.width);
                auto query = ::tyloi::cuda::attention_ops::rms_norm(pair.narrow(-1, 0, c_.width),
                                      w_->query_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps)
                                 .permute({0, 2, 1, 3});
                auto key =
                    ::tyloi::cuda::attention_ops::rms_norm(::tyloi::ops::project(hidden,w_->input_projection[1]).dense().reshape({b, t, c_.kv_heads, c_.width}),
                             w_->key_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps)
                        .permute({0, 2, 1, 3});
                auto value = ::tyloi::ops::project(hidden,w_->input_projection[2]).dense().reshape({b, t, c_.kv_heads, c_.width});
                query = rotary_->forward(query, current_positions);
                key = rotary_->forward(key, current_positions).permute({0, 2, 1, 3});
                auto iqk = ::tyloi::ops::project(hidden,w_->input_projection[3]).dense();
                auto iq = ::tyloi::cuda::attention_ops::rms_norm(iqk.narrow(-1, 0, c_.index_heads * c_.index_width)
                                       .reshape({b, t, c_.index_heads, c_.index_width}),
                                   w_->index_query_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps)
                              .permute({0, 2, 1, 3});
                iq = rotary_->forward(iq, current_positions).permute({0, 2, 1, 3});
                auto raw = iqk.narrow(-1, c_.index_heads * c_.index_width, c_.index_width);
                return Projection{gate, query, key, value, iq, raw};
            };
        auto append_inputs = [&](Projection &p) {
                if(managed_memory(request)) {
                    if (!(b==1))
                        throw std::runtime_error("preallocated QSA expects one sequence per cache");
                    request.memory->ensure(offset+t);
                    if(request.memory->hot.exclusive())request.memory->hot.append(encode_key(p.key).permute({0,2,1,3}),
                        encode_value(p.value).permute({0,2,1,3}),::tyloi::cuda::arange(offset,offset+t,hidden.options().dtype(::tyloi::kInt64)));
                    else {
                        graph_keys.narrow(2,offset,t).copy_(encode_key(p.key).permute({0,2,1,3}));
                        graph_values.narrow(2,offset,t).copy_(encode_value(p.value).permute({0,2,1,3}));
                    }
                    if(request.memory->hot.defined()) {
                        if(!request.memory->hot.exclusive())request.memory->hot.publish(graph_keys,graph_values,::tyloi::cuda::arange(offset,offset+t,hidden.options().dtype(::tyloi::kInt64)));
                        p.key=graph_keys.permute({0,2,1,3});p.value=graph_values.permute({0,2,1,3});
                    }else {
                        p.key=graph_keys.narrow(2,0,offset+t).permute({0,2,1,3});
                        p.value=graph_values.narrow(2,0,offset+t).permute({0,2,1,3});
                    }
                    request.keys.advance_fixed(t);request.values.advance_fixed(t);request.graph_prepared=false;
                } else {
                    p.key = request.keys.append(encode_key(p.key).reshape({b, t, -1}))
                                .reshape({b, offset + t, c_.kv_heads, key_width()});
                    p.value = request.values.append(encode_value(p.value).reshape({b, t, -1}))
                                  .reshape({b, offset + t, c_.kv_heads, value_width()});
                }
                p.raw = request.index.append(p.raw);
            };
        if(history_only) {
            auto p=project_history(hidden,current_positions);
            try {append_inputs(p);}catch(...) {truncate(request,offset);throw;}
            return {};
        }
        const bool short_gate=prepared_inputs && fused_projection_ && rotary_->fused() &&
            t<=8 && !quantization_.enabled() &&
            (hidden.scalar_type()==::tyloi::kFloat16 || hidden.scalar_type()==::tyloi::kFloat32);
        const auto attend_projected=[&](const Projection& p,const DenseTensor& selected) {
            auto key=p.key.permute({0,2,1,3}),value=p.value.permute({0,2,1,3});
            if(!short_gate)return attend(request,p.query,key,value,selected,offset);
            const auto positions=::tyloi::cuda::arange(offset,offset+t,hidden.options().dtype(::tyloi::kInt64));
            const auto hot=prepare_hot(request,selected,positions,offset+t);
            // Match the decode graph's fused gate and FP32 attention
            // accumulation for short eager/correction calls as well.
            if(selected.defined())return mfq_qwen4_exp::sparse_gqa_attention_gate(
                p.query,key,value,selected,p.gate,hidden.scalar_type()==::tyloi::kFloat16,hot);
            return mfq_qwen4_exp::causal_gqa_attention_gate(p.query,key,value,positions,
                p.gate,offset+t,hidden.scalar_type()==::tyloi::kFloat16,hot);
        };
        const auto dense = [&](const Projection &p) {
                return attend_projected(p,{});
            };
        const auto pool = [&](const Projection &p) {
                const auto &raw = p.raw;
                const auto pools = (offset + t) / c_.pool;
                auto pooled = raw.narrow(1, 0, pools * c_.pool)
                                  .reshape({b, pools, c_.pool, c_.index_width})
                                  .to(::tyloi::kFloat32)
                                  .mean(2)
                                  .to(raw.scalar_type());
                pooled = ::tyloi::cuda::attention_ops::rms_norm(pooled, w_->index_key_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps);
                auto starts = ::tyloi::cuda::arange(pools, raw.options().dtype(::tyloi::kInt64)) * c_.pool;
                auto positions = full_positions.index_select(-1, starts);
                pooled = rotary_->forward(pooled.unsqueeze(1), positions).squeeze(1);
                return pooled;
            };
        const auto score = [](const Projection &p, const DenseTensor &pooled) {
                return mfq_qwen4_exp::block_scores(p.iq, pooled);
            };
        const auto select = [&](const DenseTensor &scores, const Projection &p, const DenseTensor &pooled) {
                auto ids =
                    tyloi::cuda::ops::select_pooled_blocks(
                        scores, offset, offset + t, c_.pool, c_.budget, true);
                if (selection_trace)
                    *selection_trace = {scores, ids, p.iq, pooled};
                return ids;
            };
        const auto sparse = [&](const Projection &p, DenseTensor ids) {
                return attend_projected(p,ids);
            };
        const auto output = [&](DenseTensor attended, const Projection &p) {
                auto gated = short_gate?std::move(attended):
                    attended.to(::tyloi::kFloat32) * ::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32));
                auto result = gated.reshape({b, t, c_.heads * c_.width}).to(hidden.scalar_type());
                return project_output ? this->project_output(result) : result;
            };
        auto qkv=project_inputs();
        try {
            if(use_cache)append_inputs(qkv);
            auto attended=[&] {
                if(offset+t<=c_.budget)return dense(qkv);
                auto pooled=pool(qkv);
                auto scores=score(qkv,pooled);
                auto selected=select(scores,qkv,pooled);
                return sparse(qkv,std::move(selected));
            }();
            return output(std::move(attended),qkv);
        } catch (...) {
            if(use_cache)truncate(request,offset);
            throw;
        }
    }

  private:
    HotKvView prepare_hot(State& request,const DenseTensor& selected,const DenseTensor& positions={},int64_t columns=0,bool epoch_ready=false) const {
        if(!managed_memory(request) || !request.memory->hot.defined())return {};
        auto graph_keys = (request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{});
        auto graph_values = (request.graph_values.defined()?request.graph_values.dense():DenseTensor{});
        request.memory->hot.prepare(graph_keys,graph_values,selected,positions,columns,true,epoch_ready);
        return request.memory->hot.view();
    }
    DenseTensor attend(State& request,const DenseTensor& query, DenseTensor keys, DenseTensor values, const DenseTensor& selected, int64_t offset) const {
        if(managed_memory(request) && request.memory->hot.defined()) {
            const auto positions=::tyloi::cuda::arange(offset,offset+query.size(2),query.options().dtype(::tyloi::kInt64));
            const auto hot=offset+query.size(2)<=request.memory->hot.keys.size(2)?request.memory->hot.view():prepare_hot(request,selected,positions,offset+query.size(2));
            if(quantization_.enabled())return cuda_kv_attention(query,keys,values,quantization_,selected,positions,offset,{},0,{},{},0,hot);
            if(selected.defined())return mfq_qwen4_exp::sparse_gqa_attention(query,keys,values,selected,hot);
            return mfq_qwen4_exp::dense_gqa_attention(query,keys,values,offset,hot);
        }
        if (keys.scalar_type() == ::tyloi::kInt32) {
            // Large query tiles amortize one decode per KV row and use the
            // existing matrix attention kernels. Persistent caches stay packed;
            // floating workspace is bounded and lives only for this layer.
            const auto floating_bytes = std::size_t(keys.size(0)) * keys.size(1) * keys.size(2) * c_.width * 4;
            if (query.size(2) <= 8 || floating_bytes > (128ULL << 20))
                return cuda_kv_attention(query, keys, values, quantization_, selected, {}, offset);
            keys = cuda_kv_decode(keys, int(c_.width), quantization_.key_bits(), false);
            values = cuda_kv_decode(values, int(c_.width), quantization_.value_bits(), true);
        }
        if (selected.defined())
            return mfq_qwen4_exp::sparse_gqa_attention(query, keys, values, selected);
        return mfq_qwen4_exp::dense_gqa_attention(query, keys, values, offset);
    }
    int64_t key_width() const { return quantization_.enabled()?cuda_kv_packed_width(int(c_.width),quantization_.key_bits()):c_.width; }
    int64_t value_width() const { return quantization_.enabled()?cuda_kv_packed_width(int(c_.width),quantization_.value_bits()):c_.width; }
    ::tyloi::ScalarType cache_dtype() const { return quantization_.enabled()?::tyloi::kInt32: ::tyloi::kFloat16; }
    DenseTensor encode_key(const DenseTensor& x) const { return quantization_.enabled()?cuda_kv_encode(x,quantization_.key_bits(),false):x.to(::tyloi::kFloat16); }
    DenseTensor encode_value(const DenseTensor& x) const { return quantization_.enabled()?cuda_kv_encode(x,quantization_.value_bits(),true):x.to(::tyloi::kFloat16); }
    struct Projection {DenseTensor gate,query,key,value,iq,raw;bool cache_written=false;};
    DenseTensor pool_index(const DenseTensor& raw,const DenseTensor& starts) {
        auto pooled=raw.reshape({1,raw.size(1)/c_.pool,c_.pool,c_.index_width}).mean(2);
        pooled=project_norm(pooled,w_->index_key_norm.dense());
        return rotary_->forward(pooled.unsqueeze(1),starts).squeeze(1);
    }
    DenseTensor forward_adaptive(QsaTqCache& cache,const DenseTensor& hidden,const DenseTensor& positions,
            const DenseTensor& full_positions,const std::vector<DenseTensor>& projections,
            bool project_output,bool history_only) {
        const auto start=cache.position(),tokens=hidden.size(1);
        if (!(hidden.size(0)==1 && tokens<=c_.maximum-start &&
            positions.size(-1)==tokens && full_positions.size(-1)==start+tokens))
            throw std::runtime_error("adaptive QSA cache/position geometry mismatch");
        const auto rows=cache.prefill_rows(tokens);
        const auto type=hidden.scalar_type()==::tyloi::kFloat16 ? ::tyloi::kFloat16 : ::tyloi::kFloat32;
        auto output=history_only ? DenseTensor{} : ::tyloi::cuda::empty({1,tokens,c_.heads*c_.width},
            hidden.options().dtype(type));
        const auto pool=[this](const DenseTensor& raw,const DenseTensor& starts) { return pool_index(raw,starts); };
        try {
            for (int64_t begin=0;begin<tokens;begin+=rows) {
                const auto count=std::min(rows,tokens-begin);
                auto h=hidden.narrow(1,begin,count),pos=positions.narrow(-1,begin,count);
                std::vector<DenseTensor> projected;
                for (const auto& value:projections) projected.push_back(value.narrow(1,begin,count));
                auto p=history_only?project_history(h,pos):project(h,pos,true,{},{},{},std::move(projected));
                auto buffer=history_only?DenseTensor{}:output.narrow(1,begin,count);
                cache.forward(p.query,p.key,p.value,p.iq,p.raw,pos,pool,history_only,p.gate,std::move(buffer),true,
                    full_positions.narrow(-1,0,start+begin+count));
            }
            if (tokens>8) cache.release_prefill_workspace();
            if (history_only) return {};
            output=output.to(hidden.scalar_type());
            return project_output?this->project_output(output):output;
        } catch (...) {
            cache.truncate(start);
            if (tokens>8) cache.release_prefill_workspace();
            throw;
        }
    }
    DenseTensor streamed_graph_attention(State& request,const Projection& p,const DenseTensor& selected,::tyloi::ScalarType output_type) {
        auto* window=DecodeWindow::recording();
        if (!(window && selected.size(1)<=8))
            throw std::runtime_error("streamed QSA requires a decode window of at most eight rows");
        struct Stage {
            DenseTensor raw,host_ids,ids;
            std::shared_ptr<::tyloi::ops::KvGatherWorkspace> gather;
        };
        const auto tokens=selected.size(1),columns=selected.size(2);
        const auto key_row=c_.kv_heads*key_width(),value_row=c_.kv_heads*value_width();
        const auto stage=window->workspace<Stage>(&request,[&] {
            ResidentAllocationScope resident;
            auto result=std::make_shared<Stage>();
            result->raw=::tyloi::cuda::empty({tokens,key_row+value_row},p.key.options().dtype(cache_dtype()));
            result->host_ids=::tyloi::cuda::empty(selected.sizes(),selected.options().device(::tyloi::kCPU).pinned_memory(true));
            result->ids=::tyloi::cuda::zeros(selected.sizes(),selected.options());
            // Every layer and MTP width shares the same bounded I/O workspace.
            // A later segment may reuse it only after the prior consumer ends.
            result->gather=request.offloaded_kv->gather_workspace(8*columns,c_.kv_heads,key_width());
            return result;
        });
        stage->raw.copy_(::tyloi::cuda::cat({encode_key(p.key).reshape({tokens,key_row}),encode_value(p.value).reshape({tokens,value_row})},1));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->host_ids.data_ptr(),selected.data_ptr(),selected.nbytes(),
            cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        window->host_barrier([this,&request,stage] {
            const auto start=position(request);
            try {
                request.offloaded_kv->append(stage->raw);
                auto* ids=stage->host_ids.data_ptr<int32_t>();
                std::unordered_map<int32_t,int32_t> compact;
                std::vector<int64_t> requested;
                requested.reserve(stage->host_ids.numel());compact.reserve(stage->host_ids.numel());
                for(int64_t i=0;i<stage->host_ids.numel();++i)if(ids[i]>=0) {
                    auto [item,added]=compact.emplace(ids[i],int32_t(requested.size()));
                    if(added)requested.push_back(ids[i]);ids[i]=item->second;
                }
                request.offloaded_kv->gather_into(requested,*stage->gather);
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->ids.data_ptr(),ids,stage->ids.nbytes(),
                    cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            } catch(...) {request.offloaded_kv->truncate(start);throw;}
        });
        const auto rows=stage->gather->rows.size(0);
        const auto& gathered=stage->gather->rows.dense();
        auto flat=gathered.reshape({-1});
        auto keys=flat.narrow(0,0,rows*key_row).reshape({1,c_.kv_heads,rows,key_width()});
        auto values=flat.narrow(0,rows*key_row,rows*value_row).reshape({1,c_.kv_heads,rows,value_width()});
        if(!quantization_.enabled() && fused_projection_ && rotary_->fused() && (output_type==::tyloi::kFloat16 || output_type==::tyloi::kFloat32))
            return mfq_qwen4_exp::sparse_gqa_attention_gate(p.query,keys,values,stage->ids,p.gate,output_type==::tyloi::kFloat16);
        auto attended=attend(request,p.query,keys,values,stage->ids,rows);
        return (attended.to(::tyloi::kFloat32)*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32))).to(output_type);
    }
    DenseTensor forward_offloaded(State& request, const DenseTensor& hidden,
        const DenseTensor& positions, const DenseTensor& full_positions, std::vector<DenseTensor>* trace,
        std::vector<DenseTensor> projections={},bool project_output=true,bool history_only=false) {
        const auto offset = position(request), tokens = hidden.size(1), length = offset + tokens;
        auto graph_pooled = (request.graph_pooled.defined()?request.graph_pooled.dense():DenseTensor{});
        const auto pool_position=[&]{return managed_memory(request)?request.index.position()/c_.pool:request.offloaded_pooled->position();};
        const auto pool_range=[&](int64_t start,int64_t count,::tyloi::Tensor& owner) -> DenseTensor {
            if(managed_memory(request)) return graph_pooled.narrow(1,start,count).reshape({count,c_.index_width});
            owner=request.offloaded_pooled->range(start,count);
            return owner.dense();
        };
        if (!(hidden.size(0) == 1 && length <= c_.maximum && full_positions.size(-1) == length))
            throw std::runtime_error("streamed QSA requires one sequence within its configured context");
        const auto key_row = c_.kv_heads * key_width(), value_row = c_.kv_heads * value_width();
        const auto max_keys = std::min(length, c_.budget + c_.pool - 1);
        const auto row_bytes = std::size_t(key_row+value_row) * (quantization_.enabled()?4:2);
        if (std::size_t(max_keys) * row_bytes > request.offload->config().buffer_bytes / 4)
            throw std::invalid_argument("QSA I/O buffer cannot hold one selected context; increase the resident KV budget");
        auto p = history_only?project_history(hidden,positions):
            project( hidden, positions, length > c_.budget,{},{},{},std::move(projections));
        try {
            request.offloaded_kv->append(::tyloi::cuda::cat({encode_key(p.key).reshape({tokens,key_row}), encode_value(p.value).reshape({tokens,value_row})}, 1));
            const auto first_pool = pool_position();
            if(managed_memory(request))request.index.append(p.raw);
            else request.offloaded_index->append(p.raw.reshape({tokens,c_.index_width}));
            const auto new_pools = length / c_.pool - first_pool;
            if (new_pools) {
                ::tyloi::Tensor raw_owner;
                DenseTensor raw;
                if(managed_memory(request))
                    raw=request.index.storage().dense().narrow(1,first_pool*c_.pool,new_pools*c_.pool).reshape({new_pools*c_.pool,c_.index_width});
                else {
                    raw_owner=request.offloaded_index->range(first_pool * c_.pool, new_pools * c_.pool);
                    raw=raw_owner.dense();
                }
                auto pooled = raw.reshape({1, new_pools, c_.pool, c_.index_width}).to(::tyloi::kFloat32).mean(2).to(::tyloi::kFloat16);
                pooled = ::tyloi::cuda::attention_ops::rms_norm(pooled, w_->index_key_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps);
                auto starts = (::tyloi::cuda::arange(new_pools, positions.options().dtype(::tyloi::kInt64)) + first_pool) * c_.pool;
                pooled = rotary_->forward(pooled.unsqueeze(1), full_positions.index_select(-1, starts)).squeeze(1);
                if(managed_memory(request))graph_pooled.narrow(1,first_pool,new_pools).copy_(pooled);
                else request.offloaded_pooled->append(pooled.reshape({new_pools, c_.index_width}));
            }
            if(history_only)return {};
            // When one layer fits in staging, reuse that layer across the
            // entire prefill tile instead of gathering the same rows per token.
            // Other layers remain under the GPU/RAM/SSD eviction policy.
            if (tokens > 8 && std::size_t(length) * row_bytes <= request.offload->config().buffer_bytes / 2 &&
                    std::size_t(length / c_.pool) * c_.index_width * 2 <= request.offload->config().buffer_bytes / 4) {
                auto packed_owner = request.offloaded_kv->range(0, length);
                const auto& packed = packed_owner.dense();
                auto key = packed.narrow(1,0,key_row).reshape({1,length,c_.kv_heads,key_width()}).permute({0,2,1,3});
                auto value = packed.narrow(1,key_row,value_row).reshape({1,length,c_.kv_heads,value_width()}).permute({0,2,1,3});
                DenseTensor ids;
                if (length > c_.budget) {
                    ::tyloi::Tensor pooled_owner;
                    auto pooled = pool_range(0,length / c_.pool,pooled_owner).reshape({1,length / c_.pool,c_.index_width});
                    auto scores = mfq_qwen4_exp::block_scores(p.iq, pooled);
                    ids = tyloi::cuda::ops::select_pooled_blocks(
                        scores, offset, length, c_.pool, c_.budget, true);
                    if (trace) *trace = {scores,ids,p.iq,pooled};
                }
                auto attended = attend(request,p.query, key, value, ids, offset);
                auto gated = attended.to(::tyloi::kFloat32) * ::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32));
                auto result = gated.reshape({1,tokens,c_.heads*c_.width}).to(hidden.scalar_type());
                return project_output ? this->project_output(result) : result;
            }
            const auto queries = std::max<int64_t>(1, std::min<int64_t>(32,
                request.offload->config().buffer_bytes / 4 / (std::size_t(max_keys) * row_bytes)));
            std::vector<DenseTensor> outputs;
            for (int64_t begin = 0; begin < tokens; begin += queries) {
                const auto count = std::min(queries, tokens - begin);
                auto cache_positions = ::tyloi::cuda::arange(offset + begin, offset + begin + count, positions.options().dtype(::tyloi::kInt64));
                DenseTensor selected;
                if (length <= c_.budget) {
                    auto ids = ::tyloi::cuda::arange(max_keys, cache_positions.options()).reshape({1, 1, max_keys}).expand({1, count, max_keys});
                    selected = ::tyloi::cuda::where(ids <= cache_positions.reshape({1, count, 1}), ids, ::tyloi::cuda::full_like(ids, -1)).to(::tyloi::kInt32).contiguous();
                } else {
                    const auto pools = pool_position(), keep = std::min(c_.budget / c_.pool, pools);
                    DenseTensor best_scores, best_ids;
                    std::vector<DenseTensor> traced_scores, traced_pools;
                    const auto pool_chunk = std::max<int64_t>(16, std::min<int64_t>(1024,
                        request.offload->config().buffer_bytes / 8 / (c_.index_width * 2 + count * 4)));
                    for (int64_t start = 0; start < pools; start += pool_chunk) {
                        const auto n = std::min(pool_chunk, pools - start);
                        ::tyloi::Tensor pooled_owner;
                        auto pooled = pool_range(start, n, pooled_owner).reshape({1, n, c_.index_width});
                        auto scores = mfq_qwen4_exp::block_scores(p.iq.narrow(1, begin, count), pooled);
                        auto ids = ::tyloi::cuda::arange(start, start + n, cache_positions.options());
                        auto visible = (ids * c_.pool + c_.pool - 1).reshape({1, 1, n}) <= cache_positions.reshape({1, count, 1});
                        auto ranked = ::tyloi::cuda::where(visible, scores.to(::tyloi::kFloat32), ::tyloi::cuda::full_like(scores, -1e30).to(::tyloi::kFloat32));
                        auto top = ::tyloi::cuda::topk(ranked, std::min(keep, n), -1, true, false);
                        auto values = std::get<0>(top), indices = std::get<1>(top) + start;
                        if (best_scores.defined()) {
                            values = ::tyloi::cuda::cat({best_scores, values}, -1);
                            indices = ::tyloi::cuda::cat({best_ids, indices}, -1);
                            top = ::tyloi::cuda::topk(values, std::min(keep, values.size(-1)), -1, true, false);
                            best_scores = std::get<0>(top); best_ids = indices.gather(-1, std::get<1>(top));
                        } else { best_scores = values; best_ids = indices; }
                        if (trace) { traced_scores.push_back(scores); traced_pools.push_back(pooled); }
                    }
                    selected = mfq_qwen4_exp::qsa_selected_tokens(best_ids, cache_positions, c_.pool, c_.budget);
                    if (trace && begin == 0) *trace = {::tyloi::cuda::cat(traced_scores, -1), selected, p.iq.narrow(1, begin, count), ::tyloi::cuda::cat(traced_pools, 1)};
                }
                auto cpu_ids = selected.to(::tyloi::kCPU).contiguous();
                auto* mapping = static_cast<int32_t*>(cpu_ids.data_ptr());
                std::unordered_map<int32_t, int32_t> compact;
                std::vector<int64_t> requested;
                for (int64_t index = 0; index < cpu_ids.numel(); ++index) {
                    if (mapping[index] < 0) continue;
                    auto [found, added] = compact.emplace(mapping[index], static_cast<int32_t>(requested.size()));
                    if (added) requested.push_back(mapping[index]);
                    mapping[index] = found->second;
                }
                auto packed_owner = request.offloaded_kv->gather(requested);
                const auto& packed = packed_owner.dense();
                const auto n = static_cast<int64_t>(requested.size());
                auto key = packed.narrow(1,0,key_row).reshape({1,n,c_.kv_heads,key_width()}).permute({0,2,1,3}).contiguous();
                auto value = packed.narrow(1,key_row,value_row).reshape({1,n,c_.kv_heads,value_width()}).permute({0,2,1,3}).contiguous();
                // IDs now address a compact union; selection already contains the causal mask.
                outputs.push_back(attend(request,p.query.narrow(2,begin,count),key,value,cpu_ids.to(selected.device()),n));
            }
            auto attended = ::tyloi::cuda::cat(outputs, 1);
            auto gated = attended.to(::tyloi::kFloat32) * ::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32));
            auto result=gated.reshape({1, tokens, c_.heads * c_.width}).to(hidden.scalar_type());
            return project_output?this->project_output(result):result;
        } catch (...) { truncate(request,offset); throw; }
    }
    DenseTensor project_norm(const DenseTensor& value,const DenseTensor& weight) {
        if(value.scalar_type()==::tyloi::kFloat16 || value.scalar_type()==::tyloi::kFloat32)
            return grouped_rms_norm_cuda(value.contiguous(),weight.to(::tyloi::kFloat32).contiguous(),value.size(-1),c_.eps,1.0);
        return ::tyloi::cuda::attention_ops::rms_norm(value,weight.to(::tyloi::kFloat32)+1,c_.eps);
    }
    Projection project_history(const DenseTensor& hidden,const DenseTensor& positions) {
        const auto b=hidden.size(0),t=hidden.size(1);
        auto source=::tyloi::ops::project(hidden,w_->input_projection[1]).dense().reshape({b,t,c_.kv_heads,c_.width});
        DenseTensor key;
        if(fused_projection_ && rotary_->fused() &&
                (source.scalar_type()==::tyloi::kFloat16 || source.scalar_type()==::tyloi::kFloat32))
            key=rotary_->forward_normalized_grouped({source},{w_->key_norm.dense()},positions,c_.eps,1)[0].permute({0,2,1,3});
        else key=rotary_->forward(project_norm(source,w_->key_norm.dense()).permute({0,2,1,3}),positions).permute({0,2,1,3});
        auto value=::tyloi::ops::project(hidden,w_->input_projection[2]).dense().reshape({b,t,c_.kv_heads,c_.width});
        auto raw=::tyloi::ops::project(hidden,w_->input_projection[3]).dense().narrow(-1,c_.index_heads*c_.index_width,c_.index_width);
        return {{},{},std::move(key),std::move(value),{},std::move(raw)};
    }
    Projection project(const DenseTensor& hidden,const DenseTensor& positions,
                       bool index_query_needed=true,const DenseTensor& cache_positions={},
                       const DenseTensor& key_cache={},const DenseTensor& value_cache={},
                       std::vector<DenseTensor> projections={}) {
        const auto b=hidden.size(0),t=hidden.size(1);
        if(projections.empty()) {
            if(tyloi::cuda::Executor::current().compute().attention_grouped.value_or(grouped_projection_) && ::tyloi::ops::grouped(w_->input_projection))
                projections=project_inputs(hidden);
            else projections={project_query(hidden),
                ::tyloi::ops::project(hidden,w_->input_projection[1]).dense(),
                ::tyloi::ops::project(hidden,w_->input_projection[2]).dense(),
                ::tyloi::ops::project(hidden,w_->input_projection[3]).dense()};
        }
        if (!(projections.empty() || projections.size()==4))
            throw std::runtime_error("QSA input projection count mismatch");
        auto pair=projections[0].reshape({b,t,c_.heads,2*c_.width});
        auto gate=pair.narrow(-1,c_.width,c_.width);
        auto key_source=projections[1].reshape({b,t,c_.kv_heads,c_.width});
        auto value=projections[2].reshape({b,t,c_.kv_heads,c_.width});
        const auto supported=[](const DenseTensor& x) { return x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32; };
        const bool fused=fused_projection_ && rotary_->fused() && supported(pair) && supported(key_source) && supported(value);
        const bool cache_written=fused && key_cache.defined();
        auto iqk=projections[3];
        DenseTensor query,key,iq;
        if(fused) {
            std::vector<DenseTensor> inputs{pair.narrow(-1,0,c_.width),key_source};
            std::vector<DenseTensor> weights{w_->query_norm.dense(),w_->key_norm.dense()};
            if(index_query_needed) {
                auto source=iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width});
                if(supported(source)){inputs.push_back(source);weights.push_back(w_->index_query_norm.dense());}
            }
            auto normalized=rotary_->forward_normalized_grouped(inputs,weights,positions,c_.eps,1,
                cache_written?key_cache:DenseTensor{},cache_written?cache_positions:DenseTensor{},
                cache_written?value:DenseTensor{},cache_written?value_cache:DenseTensor{});
            query=normalized[0];key=normalized[1];
            if(key.defined())key=key.permute({0,2,1,3});
            if(normalized.size()==3)iq=normalized[2].permute({0,2,1,3});
        } else {
            query=project_norm(pair.narrow(-1,0,c_.width),w_->query_norm.dense()).permute({0,2,1,3});
            key=project_norm(key_source,w_->key_norm.dense()).permute({0,2,1,3});
            query=rotary_->forward(query,positions);key=rotary_->forward(key,positions).permute({0,2,1,3});
        }
        if(index_query_needed && !iq.defined()) {
            auto source=iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width});
            if(fused && supported(source))iq=rotary_->forward_normalized(source,w_->index_query_norm.dense(),positions,c_.eps).permute({0,2,1,3});
            else {
                iq=project_norm(source,w_->index_query_norm.dense()).permute({0,2,1,3});
                iq=rotary_->forward(iq,positions).permute({0,2,1,3});
            }
        }
        return {gate,query,key,value,iq,iqk.narrow(-1,c_.index_heads*c_.index_width,c_.index_width),cache_written};
    }
    std::shared_ptr<const QsaWeights> w_;
    QsaConfig c_;
    tyloi::state::KvQuantization quantization_=c_.quantization;
    std::shared_ptr<RotaryEmbedding> rotary_;
    const bool fused_projection_;
    const bool grouped_projection_;
};
} // namespace tyloi::ops::detail

namespace tyloi::ops {
namespace {
Tensor hold(DenseTensor value) { return Tensor(std::move(value)); }
std::vector<DenseTensor> views(const std::vector<Tensor>& values) {
    std::vector<DenseTensor> result;
    result.reserve(values.size());
    for (const auto& value : values) {
        if (value.defined() && !value.representation<DenseTensor>())
            throw std::invalid_argument("QSA activation must be dense");
        result.push_back(value.representation<DenseTensor>() ? value.dense() : DenseTensor{});
    }
    return result;
}
void publish(std::vector<Tensor>& owners, const std::vector<DenseTensor>& values) {
    std::vector<Tensor> result;
    result.reserve(values.size());
    for (const auto& value : values) result.push_back(hold(value));
    owners = std::move(result);
}
} // namespace

struct Qsa::Impl {
    Device device;
    detail::QsaCore core;
    Impl(QsaWeights weights, QsaConfig config)
        : device(config.device),
          core(std::make_shared<const QsaWeights>(std::move(weights)), config,
               std::make_shared<cuda::RotaryEmbedding>(config.rotary, config.maximum, config.rope_base,
                   config.sections, config.interleaved, cuda::options::rotary_fusion().value_or(true), config.yarn),
               true, true) {}
};
struct Qsa::State::Impl {
    detail::QsaCore::State cache;
    // Retain failed trace bridge allocations until the request has drained and resets.
    std::vector<DenseTensor> failed_trace;
    DenseTensor failed_output;
    explicit Impl(const detail::QsaCore& model) : cache(model) {}
};

struct Qsa::HostLoan::Impl {
    std::vector<cuda::QsaTqCache*> caches;
    std::vector<cuda::QsaRamCache*> pools;
    std::vector<std::shared_ptr<void>> holds;
};

Qsa::HostLoan::HostLoan(std::span<State* const> layers, std::size_t base_bytes,
                       std::int64_t end, std::int64_t rows) : bytes_(base_bytes) {
    std::size_t extra = 0;
    for (auto* layer : layers) {
        auto* cache = layer->impl_->cache.adaptive.get();
        if (!cache) continue;
        if (!impl_) impl_ = std::make_unique<Impl>();
        impl_->caches.push_back(cache);
        auto* ram = cache->ram();
        if (!ram) continue;
        extra = std::max(extra, cache->prefill_host_bytes(end, rows));
        const auto pool = ram->workspace_pool();
        if (std::none_of(impl_->pools.begin(), impl_->pools.end(),
                [pool](const auto* other) { return other->workspace_pool() == pool; }))
            impl_->pools.push_back(ram);
    }
    bytes_ = engine::prefill_budget_add(base_bytes, extra);
    if (impl_)
        for (auto* pool : impl_->pools) impl_->holds.push_back(pool->hold_workspace(bytes_));
}

Qsa::HostLoan::~HostLoan() = default;

void Qsa::HostLoan::prepare() {
    if (!impl_ || impl_->pools.empty()) return;
    for (auto* cache : impl_->caches) cache->wait();
    cuda::QsaRamCache::prepare_workspace(impl_->pools, bytes_);
}

void Qsa::HostLoan::collect() {
    if (!impl_) return;
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    for (auto* cache : impl_->caches) cache->collect();
}

Qsa::Qsa(QsaWeights weights, QsaConfig config)
    : impl_(std::make_unique<Impl>(std::move(weights), std::move(config))) {}
Qsa::~Qsa() = default;
Qsa::Qsa(Qsa&&) noexcept = default;
Qsa& Qsa::operator=(Qsa&&) noexcept = default;
Qsa::State::State(const Qsa& model) {
    std::optional<MfqCudaGuard> guard;
    if (model.impl_->device.is_cuda()) guard.emplace(model.impl_->device);
    impl_ = std::make_unique<Impl>(model.impl_->core);
}
Qsa::State::~State() = default;
Qsa::State::State(State&&) noexcept = default;
Qsa::State& Qsa::State::operator=(State&&) noexcept = default;

std::int64_t Qsa::position(const State& state) const { return impl_->core.position(state.impl_->cache); }
bool Qsa::managed_memory(const State& state) const { return impl_->core.managed_memory(state.impl_->cache); }
std::int64_t Qsa::graph_capacity(const State& state) const { return impl_->core.graph_capacity(state.impl_->cache); }
std::vector<Tensor*> Qsa::graph_state(State& state) const { return impl_->core.graph_state(state.impl_->cache); }
void Qsa::activate_memory(State& state) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    impl_->core.activate_memory(state.impl_->cache);
}
std::size_t Qsa::growth_bytes(const State& state, std::int64_t tokens) const {
    return impl_->core.growth_bytes(state.impl_->cache, tokens);
}
std::size_t Qsa::allocation_bytes(const State& state, std::int64_t batch, std::int64_t tokens, bool reset) const {
    return impl_->core.allocation_bytes(state.impl_->cache, batch, tokens, reset);
}
std::size_t Qsa::exit_growth(const State& state, std::int64_t tokens) const {
    return impl_->core.exit_growth(state.impl_->cache, tokens);
}
std::size_t Qsa::graph_exit_growth_bytes(const State& state) const {
    return impl_->core.graph_exit_growth_bytes(state.impl_->cache);
}
std::size_t Qsa::graph_growth_bytes(const State& state, std::int64_t capacity) const {
    return impl_->core.graph_growth_bytes(state.impl_->cache, capacity);
}
bool Qsa::supports_session(const State& state) const { return impl_->core.supports_session(state.impl_->cache); }
void Qsa::validate_session(const State& state, const Snapshot& snapshot, std::int64_t count, Device device) const {
    impl_->core.validate_session(state.impl_->cache, snapshot, count, device);
}
std::shared_ptr<void> Qsa::reserve_session(State& state, const tyloi::state::qsa::Session& snapshot) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    if (!state.impl_->cache.adaptive)
        throw std::runtime_error("incompatible Qwen4 QSA RAM session state");
    return state.impl_->cache.adaptive->reserve_session_restore(snapshot);
}
std::size_t Qsa::fresh_session_growth(const State& state, std::int64_t count) const {
    return impl_->core.fresh_session_growth(state.impl_->cache, count);
}
void Qsa::stage_session(State& state, const Snapshot& snapshot, std::int64_t count, Device device) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    impl_->core.stage_session(state.impl_->cache, snapshot, count, device);
}
std::size_t Qsa::session_bytes(const State& state, std::int64_t count) const {
    return impl_->core.session_bytes(state.impl_->cache, count);
}
void Qsa::capture_session(const State& state, Snapshot& snapshot, std::int64_t count, std::size_t& bytes) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    impl_->core.capture_session(state.impl_->cache, snapshot, count, bytes);
}
void Qsa::restore_session(State& state, const Snapshot& snapshot, std::int64_t count) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    impl_->core.restore_session(state.impl_->cache, snapshot, count);
}
std::vector<Tensor> Qsa::resident_cache_tensors(const State& state) const {
    return impl_->core.resident_cache_tensors(state.impl_->cache);
}
std::vector<Tensor> Qsa::storage(const State& state) const {
    std::vector<Tensor> result;
    const auto add = [&](const Tensor& value) {
        if (value.defined()) result.push_back(hold(value.dense()));
    };
    const auto& cache = state.impl_->cache;
    if (cache.adaptive) result = impl_->core.resident_cache_tensors(cache);
    add(cache.keys.storage()); add(cache.values.storage()); add(cache.index.storage());
    add(cache.graph_keys); add(cache.graph_values); add(cache.graph_pooled); add(cache.graph_index);
    if (cache.memory)
        for (const auto& value : cache.memory->storage()) add(value);
    for (const auto& value : state.impl_->failed_trace) if (value.defined()) result.push_back(hold(value));
    if (state.impl_->failed_output.defined()) result.push_back(hold(state.impl_->failed_output));
    return result;
}
void Qsa::leave_graph(State& state) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    impl_->core.leave_graph(state.impl_->cache);
}
void Qsa::reset(State& state) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    impl_->core.reset(state.impl_->cache);
    state.impl_->failed_trace.clear();
    state.impl_->failed_output = {};
}
void Qsa::prepare_graph(State& state, const Tensor& positions, std::int64_t capacity, std::int64_t pending_tokens) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    if (positions.defined() && !positions.representation<DenseTensor>())
        throw std::invalid_argument("QSA positions must be dense");
    impl_->core.prepare_graph(state.impl_->cache,
        positions.representation<DenseTensor>() ? positions.dense() : DenseTensor{}, capacity, pending_tokens);
}
void Qsa::advance_graph(State& state, std::int64_t tokens, std::int64_t attention_rows) const {
    impl_->core.advance_graph(state.impl_->cache, tokens, attention_rows);
}
void Qsa::truncate(State& state, std::int64_t keep) const { impl_->core.truncate(state.impl_->cache, keep); }
std::vector<Tensor> Qsa::project_inputs(const Tensor& input) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    std::vector<Tensor> result;
    result.reserve(4);
    for (int i=0; i<4; ++i) result.emplace_back(DenseTensor{});
    auto values = impl_->core.project_inputs(input.dense());
    if (values.size()!=result.size()) throw std::logic_error("QSA projection count changed");
    for (std::size_t i=0; i<values.size(); ++i) result[i].dense()=std::move(values[i]);
    return result;
}
Tensor Qsa::project_output(const Tensor& input) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    Tensor result{DenseTensor{}};
    result.dense()=impl_->core.project_output(input.dense());
    return result;
}
Tensor Qsa::forward(State& state, const Tensor& hidden, const Tensor& positions, const Tensor& full_positions,
        bool cache, std::vector<Tensor>* selection_trace, std::vector<Tensor> projections,
        bool project_output, bool history_only) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    auto trace = selection_trace ? views(*selection_trace) : std::vector<DenseTensor>{};
    DenseTensor result;
    try {
        result = impl_->core.forward(state.impl_->cache, hidden.dense(), positions.dense(), full_positions.dense(),
            cache, selection_trace ? &trace : nullptr, views(projections), project_output, history_only);
        if (selection_trace) publish(*selection_trace, trace);
        return hold(result);
    } catch (...) {
        state.impl_->failed_trace = std::move(trace);
        state.impl_->failed_output = std::move(result);
        throw;
    }
}
Tensor Qsa::forward_graph(State& state, const Tensor& hidden, const Tensor& positions, const Tensor& cache_positions,
        std::vector<Tensor> projections, bool project_output, bool history_only, bool query_tail,
        const Tensor& pool_positions) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    Tensor result{DenseTensor{}};
    result.dense()=impl_->core.forward_graph(state.impl_->cache, hidden.dense(), positions.dense(), cache_positions.dense(),
        views(projections), project_output, history_only, query_tail, nullptr,
        pool_positions ? pool_positions.dense() : DenseTensor{});
    return result;
}
Tensor Qsa::forward_independent_graph(const Tensor& hidden, std::span<const GraphRow> rows) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->device.is_cuda()) guard.emplace(impl_->device);
    std::vector<detail::QsaCore::GraphRow> values;
    values.reserve(rows.size());
    for (const auto& row : rows) {
        if (!row.state)
            throw std::runtime_error("QSA independent graph row has no state");
        values.push_back({&row.state->impl_->cache, row.positions.dense(), row.cache_positions.dense()});
    }
    return hold(impl_->core.forward_independent_graph(hidden.dense(), values));
}
struct Qsa::Graph::Impl {
    std::vector<Request> requests;
    std::vector<Row> rows;
    bool independent, admitted = false, materialized = false, clear = false, hidden = false, laid_out = false, adaptive = false;
    std::int64_t tokens = 0, required = 0;
    std::shared_ptr<cuda::KvMemoryManager> manager;
    std::vector<std::pair<Device, std::size_t>> unmanaged_growth;
    Impl(std::vector<Request> values, bool batch)
        : requests(std::move(values)), rows(requests.size()), independent(batch) {}
};
Qsa::Graph::Graph(std::vector<Request> requests, bool independent)
    : impl_(std::make_unique<Impl>(std::move(requests), independent)) {}
Qsa::Graph::~Graph() = default;
Qsa::Graph::Graph(Graph&&) noexcept = default;
Qsa::Graph& Qsa::Graph::operator=(Graph&&) noexcept = default;
bool Qsa::Graph::eligible(bool allow_pending) const {
    bool pending = impl_->requests.empty() && cuda::qsa_tq_cache_enabled(), ordinary = false;
    for (const auto& request : impl_->requests) for (const auto& layer : request.layers) {
        if (const auto& cache = layer.second->impl_->cache.adaptive) {
            if (!cache->pending_supported()) return false;
            pending = true;
        } else ordinary = true;
    }
    if (pending) return allow_pending && !ordinary && mfq_cuda_graph_capture_supported();
    auto& executor = cuda::Executor::current();
    const auto offload = executor.kv_offload();
    if (offload) {
        const auto* option = cuda::env_lookup("TYLOINFER_KV_STREAM_GRAPHS");
        const auto manager = executor.kv_memory();
        if (impl_->independent || !manager || !manager->offload() ||
                (option && option[0] == '0')) return false;
    }
    return !impl_->independent || mfq_cuda_graph_capture_supported();
}
bool Qsa::Graph::reusable() const {
    if (impl_->requests.size() != 1 || impl_->requests.front().layers.empty()) return false;
    const auto& layers = impl_->requests.front().layers;
    return std::all_of(layers.begin(), layers.end(), [](const auto& layer) {
        return layer.first->graph_capacity(*layer.second) > 0;
    });
}
bool Qsa::Graph::warmup_supported() const {
    if (!eligible()) return false;
    for (const auto& request : impl_->requests)
        for (const auto& layer : request.layers)
            if (!layer.first->graph_capacity(*layer.second)) return false;
    return true;
}
void Qsa::Graph::admit(std::int64_t tokens, std::int64_t previous_capacity, bool hidden) {
    auto& plan = *impl_;
    if (plan.admitted) throw std::logic_error("QSA graph plan was already admitted");
    std::vector<std::pair<Device, std::vector<cuda::QsaRamCache*>>> ram;
    const auto admit_pending = [&](const Request& request, Row& row) {
        row.capacity = request.maximum;
        for (const auto& layer : request.layers) {
            const auto& cache = layer.second->impl_->cache.adaptive;
            if (!(cache && cache->pending_supported() && cache->position()==request.position))
                throw std::runtime_error("incompatible adaptive QSA graph state");
            const auto kind = cache->device_graph_kind(tokens);
            if (!(kind>0 && kind<4 && (!row.pending_kind || row.pending_kind==kind)))
                throw std::runtime_error("invalid adaptive QSA graph append");
            row.pending_kind = kind;
            for (const auto& growth : cache->graph_growth(tokens)) {
                plan.unmanaged_growth.push_back(growth);
                if (plan.independent) {
                    if (growth.second > std::numeric_limits<std::size_t>::max() - row.growth)
                        throw std::overflow_error("QSA graph row growth overflow");
                    row.growth += growth.second;
                }
            }
            if (auto* staging = cache->ram()) {
                const auto device = layer.first->impl_->device;
                auto entry = std::find_if(ram.begin(), ram.end(), [&](const auto& item) { return item.first == device; });
                if (entry == ram.end()) ram.emplace_back(device, std::vector<cuda::QsaRamCache*>{staging});
                else entry->second.push_back(staging);
            }
        }
    };
    if (plan.independent) {
        if (tokens != 1) throw std::invalid_argument("independent QSA graph requires one token");
        for (const auto& request : plan.requests)
            for (const auto& layer : request.layers)
                if (layer.second->impl_->cache.adaptive) plan.adaptive = true;
        for (std::size_t i = 0; i < plan.requests.size(); ++i) {
            const auto& request = plan.requests[i];
            if (!(request.position >= 0 && request.maximum > 0 && request.position < request.maximum))
                throw std::runtime_error("Flash-Next batch graph exceeds max_context");
            plan.required = std::max(plan.required, request.position + 1);
            if (plan.adaptive) {
                if (request.layers.empty())
                    throw std::runtime_error("adaptive QSA graph requires attention layers");
                admit_pending(request, plan.rows[i]);
            }
        }
        if (!plan.adaptive) plan.manager = cuda::Executor::current().kv_memory();
    } else {
        if (plan.requests.size() != 1) throw std::invalid_argument("single QSA graph requires one request");
        const auto& request = plan.requests.front();
        if (!(tokens > 0 && request.position >= 0 && request.maximum > 0 &&
            request.position <= request.maximum && tokens <= request.maximum - request.position))
            throw std::runtime_error("Qwen4 graph exceeds context capacity");
        plan.required = request.position + tokens;
        auto capacity = previous_capacity;
        if (!request.layers.empty()) {
            const auto& first = request.layers.front();
            if (first.second->impl_->cache.adaptive) {
                plan.adaptive = true;
                admit_pending(request, plan.rows.front());
                capacity = plan.rows.front().capacity;
            } else if (first.first->graph_capacity(*first.second)) {
                plan.manager = cuda::Executor::current().kv_memory();
                capacity = first.first->graph_capacity(*first.second);
            }
        }
        plan.rows.front().capacity = capacity;
        plan.clear = plan.required > capacity;
    }
    for (const auto& [device, caches] : ram) {
        const auto [gpu, host] = cuda::QsaRamCache::graph_growth(caches);
        if (gpu) plan.unmanaged_growth.emplace_back(device, gpu);
        if (host) plan.unmanaged_growth.emplace_back(Device(kCPU), host);
    }
    plan.tokens = tokens;
    plan.hidden = hidden;
    plan.admitted = true;
    plan.laid_out = plan.independent ? plan.adaptive : !plan.clear;
}
int Qsa::Graph::pending_kind() const {
    if (!impl_->admitted) throw std::logic_error("QSA graph is not admitted");
    return impl_->rows.size() == 1 ? impl_->rows.front().pending_kind : 0;
}
std::vector<std::pair<Device, std::size_t>> Qsa::Graph::growth() const {
    const auto& plan = *impl_;
    if (!plan.admitted) throw std::logic_error("QSA graph plan is not admitted");
    auto result = plan.manager ? plan.manager->growth(plan.required) : std::vector<std::pair<Device, std::size_t>>{};
    for (const auto& [device, bytes] : plan.unmanaged_growth) {
        auto entry = std::find_if(result.begin(), result.end(), [&](const auto& item) { return item.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else {
            if (bytes > std::numeric_limits<std::size_t>::max() - entry->second)
                throw std::overflow_error("QSA graph growth overflow");
            entry->second += bytes;
        }
    }
    return result;
}
void Qsa::Graph::materialize() {
    auto& plan = *impl_;
    if (!plan.admitted) throw std::logic_error("QSA graph plan is not admitted");
    if (plan.materialized) throw std::logic_error("QSA graph plan was already materialized");
    if (plan.manager) plan.manager->ensure(plan.required);
    plan.materialized = true;
}
bool Qsa::Graph::clear_required() const {
    if (!impl_->admitted) throw std::logic_error("QSA graph plan is not admitted");
    return impl_->clear;
}
void Qsa::Graph::layout() {
    auto& plan = *impl_;
    if (!plan.admitted) throw std::logic_error("QSA graph plan is not admitted");
    if (plan.independent && plan.adaptive) { plan.laid_out = true; return; }
    if (!plan.independent) {
        const auto& request = plan.requests.front();
        const auto required = request.position + plan.tokens;
        const auto planned = std::max(required, request.reserve);
        if (!(planned <= request.maximum))
            throw std::runtime_error("Qwen4 graph reserve exceeds context capacity");
        auto capacity = std::min<std::int64_t>(request.maximum,
            std::max<std::int64_t>(plan.hidden ? 2048 : 256, plan.rows.front().capacity));
        while (capacity < planned)
            capacity = std::min(request.maximum, capacity > request.maximum / 2 ? request.maximum : capacity * 2);
        plan.rows.front().capacity = capacity;
        plan.laid_out = true;
        if (const auto* trace = cuda::env_lookup("TYLOINFER_RUNTIME_TRACE_CUDA_GRAPH"); trace && trace[0] == '1')
            std::cerr << "runtime_qwen4_graph_capacity position=" << request.position
                      << " required=" << required << " planned=" << planned << " capacity=" << capacity << '\n';
        return;
    }
    const auto batch = static_cast<std::int64_t>(plan.requests.size());
    plan.unmanaged_growth.clear();
    for (std::size_t i = 0; i < plan.requests.size(); ++i) {
        const auto& request = plan.requests[i];
        const auto required = request.position + 1;
        if (!(required <= request.maximum))
            throw std::runtime_error("Flash-Next batch graph exceeds max_context");
        auto capacity = std::min<std::int64_t>(batch > 8 ? cuda::options::continuous_batch_kv_initial_capacity() : 256,
            request.maximum);
        if (batch > 16 && capacity < required) {
            const auto chunks = 1 + (required - 1) / 128;
            capacity = chunks > request.maximum / 128 ? request.maximum : chunks * 128;
        } else while (capacity < required)
            capacity = std::min(request.maximum, capacity > request.maximum / 2 ? request.maximum : capacity * 2);
        for (const auto& layer : request.layers)
            if (layer.first->graph_capacity(*layer.second)) capacity = layer.first->graph_capacity(*layer.second);
        auto& row = plan.rows[i];
        row.capacity = capacity;
        row.growth = 0;
        for (const auto& layer : request.layers) {
            // Managed mappings are planned once through the manager, which can
            // also own allocations outside this batch.
            if (layer.first->managed_memory(*layer.second)) continue;
            const auto bytes = layer.first->graph_growth_bytes(*layer.second, capacity);
            if (bytes > std::numeric_limits<std::size_t>::max() - row.growth)
                throw std::overflow_error("QSA graph row growth overflow");
            row.growth += bytes;
            if (!bytes) continue;
            const auto device = layer.first->impl_->device;
            auto entry = std::find_if(plan.unmanaged_growth.begin(), plan.unmanaged_growth.end(),
                                      [&](const auto& item) { return item.first == device; });
            if (entry == plan.unmanaged_growth.end()) plan.unmanaged_growth.emplace_back(device, bytes);
            else {
                if (bytes > std::numeric_limits<std::size_t>::max() - entry->second)
                    throw std::overflow_error("QSA graph growth overflow");
                entry->second += bytes;
            }
        }
    }
    plan.laid_out = true;
}
const std::vector<Qsa::Graph::Row>& Qsa::Graph::rows() const {
    if (!impl_->admitted) throw std::logic_error("QSA graph plan is not admitted");
    return impl_->rows;
}
std::vector<Tensor> Qsa::Graph::storage() const {
    std::vector<Tensor> result;
    for (std::size_t i = 0; i < impl_->requests.size(); ++i) {
        const auto kind = impl_->rows[i].pending_kind;
        if (!kind) continue;
        for (const auto& layer : impl_->requests[i].layers)
            for (auto& tensor : layer.second->impl_->cache.adaptive->graph_tensors(impl_->tokens, kind))
                result.emplace_back(std::move(tensor));
    }
    return result;
}
std::pair<int, std::vector<Tensor>> Qsa::Graph::pending(std::int64_t tokens, std::int64_t offset) const {
    if (impl_->independent || impl_->requests.size() != 1) return {};
    int kind = 0;
    std::vector<Tensor> result;
    for (const auto& layer : impl_->requests.front().layers) {
        const auto& cache = layer.second->impl_->cache.adaptive;
        if (!cache) return {};
        const auto value = cache->device_graph_kind(tokens, offset);
        if (value < 1 || value > 3 || (kind && kind != value)) return {};
        kind = value;
        for (auto& tensor : cache->graph_tensors(tokens, kind)) result.emplace_back(std::move(tensor));
    }
    return {kind, std::move(result)};
}
void Qsa::Graph::prepare(std::size_t request_index, std::size_t qsa_index, const Tensor& positions) const {
    if (!impl_->admitted || !impl_->laid_out || !impl_->materialized)
        throw std::logic_error("QSA graph layout is not materialized");
    const auto& request = impl_->requests.at(request_index);
    const auto& layer = request.layers.at(qsa_index);
    layer.first->prepare_graph(*layer.second, positions, impl_->rows.at(request_index).capacity,
        impl_->rows.at(request_index).pending_kind ? impl_->tokens : 0);
}
void Qsa::Graph::prepare(std::size_t request_index, const Tensor& positions) const {
    const auto& request = impl_->requests.at(request_index);
    for (std::size_t i = 0; i < request.layers.size(); ++i) prepare(request_index, i, positions);
}

} // namespace tyloi::ops
