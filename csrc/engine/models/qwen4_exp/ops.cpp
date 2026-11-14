#include "engine/models/plan.h"
#include "engine/models/session/snapshot.h"
#include "env.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_shared_gate.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "components/error.h"
#include "context.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/native/stream.h"
#include "ops.h"
#include "model.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/batch.h"
#include "predictor.h"
#include "engine/models/common/load.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/tq/source.h"
#include "engine/weights/tq/rows.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/ops/profile.h"
#include "engine/backends/cuda/kernels/moe/pipeline.h"
#include <iostream>
#include "engine/ops.h"
#include "engine/ops/residual.h"
#include "engine/ops/ffn.h"
#include "engine/ops/gdn.h"
#include "engine/ops/qsa.h"
#include "engine/generation/sequence.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "matmul_rows.h"
#include "engine/backends/cuda/native/graph.h"
#endif
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <exception>
#include <limits>

namespace tyloi::cuda {
namespace {

struct GraphCheckpoint {
    std::vector<::tyloi::Tensor> gdn_conv, gdn_state;
    std::vector<::tyloi::DenseTensor> ple_conv;
};

void add_growth(std::vector<std::pair<Device, std::size_t>>& growth, Device device,
        std::size_t bytes, std::size_t count = 1) {
    if (count && bytes > std::numeric_limits<std::size_t>::max() / count)
        throw std::overflow_error("Qwen4 graph growth overflow");
    bytes *= count;
    if (!bytes) return;
    auto found = std::find_if(growth.begin(), growth.end(),
        [device](const auto& entry) { return entry.first == device; });
    if (found == growth.end()) growth.emplace_back(device, bytes);
    else {
        if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("Qwen4 graph growth overflow");
        found->second += bytes;
    }
}

tyloi::ops::Residual::Result graph_attention(const qwen4_exp::Layer& weights, qwen4_exp::LayerState& state,
        ::tyloi::DenseTensor x, const ::tyloi::DenseTensor& ids, const ::tyloi::DenseTensor& positions,
        const ::tyloi::DenseTensor& cache_positions, const ::tyloi::DenseTensor& ple_rows, int layer,
        int64_t confirmed = 0, GraphCheckpoint* checkpoint = nullptr,
        const std::optional<tyloi::ops::Residual::Result>& pending_residual = {}) {
    if(auto* window=DecodeWindow::recording())window->set_profile_layer_kind(layer,weights.gdn?"gdn":"qsa");
    const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
    if(weights.ple)x=x+(confirmed
        ? weights.ple->forward_graph(state.ple, x,ids,ple_rows,confirmed,checkpoint->ple_conv)
        : weights.ple->forward_chunk(state.ple, x,ids,true,{},ple_rows));
    auto [attention_input,attention_residual,attention_gate]=pending_residual
        ? weights.attention_gr.pre_after(x,pending_residual->residual,pending_residual->gate,tyloi::ops::ResidualRole::attention)
        : weights.attention_gr.pre(x,tyloi::ops::ResidualRole::attention);
    mark("attention_core_begin");
    auto attention_output=weights.gdn
        ? (confirmed ? weights.gdn->forward_graph(state.gdn, attention_input,confirmed,checkpoint->gdn_conv,checkpoint->gdn_state)
                     : weights.gdn->forward_chunk(state.gdn, attention_input,true))
        : weights.qsa->forward_graph(*state.qsa, attention_input,positions,cache_positions);
    mark("attention_core_end");
    auto ffn_result=weights.ffn_gr.pre_after(attention_output,attention_residual,attention_gate,tyloi::ops::ResidualRole::ffn);
    mark("ffn_pre_end");
    attention_output={};
    return ffn_result;
}

} // namespace

static bool streamed_decode_graphs() {
    return tyloi::ops::Qsa::Graph({}, false).eligible();
}
static ::tyloi::DenseTensor join_request_outputs(
        const std::vector<::tyloi::DenseTensor>& outputs,int64_t axis) {
    const auto option=::tyloi::cuda::options::single_output_view();
    if(outputs.size()==1 && outputs.front().is_contiguous() && (!option || *option))
        return outputs.front();
    return ::tyloi::cuda::cat(outputs,axis);
}
static ::tyloi::DenseTensor append_decode_positions(
        const ::tyloi::DenseTensor& previous,const ::tyloi::DenseTensor& current,
        ::tyloi::DenseTensor& storage,int64_t maximum,
        const char* option_name="TYLOINFER_QWEN4_POSITION_BUFFER") {
    if(const auto option=::tyloi::cuda::options::position_buffer(option_name))if(!*option)
        return previous.defined()?::tyloi::cuda::cat({previous,current},-1):current.clone();
    TYLOINFER_RUNTIME_CHECK(current.dim()>=1 && (!previous.defined() || (previous.dim()==current.dim() &&
        previous.scalar_type()==current.scalar_type() && previous.device()==current.device())),
        "invalid decode position history");
    const int64_t count=previous.defined()?previous.size(-1):0,required=count+current.size(-1);
    TYLOINFER_RUNTIME_CHECK(required<=maximum,"decode position history exceeds context");
    for(int axis=0;previous.defined() && axis+1<current.dim();++axis)
        TYLOINFER_RUNTIME_CHECK(previous.size(axis)==current.size(axis),"decode position history axes changed");
    // Rollback retains a prefix view; restored histories receive a new arena.
    // The same last-axis append serves 1D target and 3D MTP rotary positions.
    const bool owns_prefix=storage.defined() && previous.defined() &&
        previous.data_ptr()==storage.data_ptr();
    if(!owns_prefix || storage.size(-1)<required) {
        int64_t capacity=owns_prefix?storage.size(-1):std::min<int64_t>(256,maximum);
        while(capacity<required)capacity=capacity>maximum/2?maximum:capacity*2;
        auto shape=current.sizes().vec();shape.back()=capacity;
        auto expanded=::tyloi::cuda::empty(shape,current.options());
        if(count)expanded.narrow(-1,0,count).copy_(previous);
        storage=std::move(expanded);
    }
    storage.narrow(-1,count,current.size(-1)).copy_(current);
    return storage.narrow(-1,0,required);
}
}

namespace tyloi::cuda::qwen4_exp {
using Config = tyloi::models::qwen4_exp::Config;

PredictorState::PredictorState(const Predictor& owner, bool initialize_layers)
    : MtpState(owner), owner(owner), positions(owner.layers.size()), lengths(owner.layers.size(),0) {
    if (initialize_layers)
        for (const auto& layer : owner.layers)
            layer_states.push_back(std::make_unique<LayerState>(*layer));
}
PredictorState::~PredictorState() = default;
int64_t PredictorState::position() const noexcept {
    return lengths.empty() ? 0 : lengths.front();
}
std::unique_ptr<tyloi::models::MtpState> Predictor::make_state() const {
    return std::make_unique<PredictorState>(*this);
}
Predictor::Predictor() = default;
Predictor::~Predictor() = default;
Predictor::Predictor(Predictor&&) noexcept = default;
Predictor& Predictor::operator=(Predictor&&) noexcept = default;

DenseTensor Predictor::embed(DenseTensor ids) const {
    auto normalized = ::tyloi::ops::contiguous(::tyloi::ops::copy(
        ::tyloi::ops::cast(ids, ::tyloi::kInt64), ::tyloi::Device(::tyloi::kCUDA, primary)));
    ::tyloi::ops::ProfileRange range("model.embed", normalized.device());
    auto output = ::tyloi::ops::embedding(normalized, embedding);
    range.finish();
    return output.dense();
}

DenseTensor Predictor::logits(DenseTensor hidden) const {
    ::tyloi::ops::ProfileRange range("model.lm_head", hidden.device());
    auto projected = ::tyloi::ops::project(hidden, output);
    range.finish();
    return ::tyloi::ops::softcap(projected, softcap).dense();
}

void Predictor::reset(tyloi::models::MtpState& value, int64_t next_batch) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(next_batch > 0, "Qwen4-Exp MTP batch must be positive");
    for (std::size_t i=0; i<layers.size(); ++i)
        request.layer_states[i]->reset(*layers[i]);
    for (auto &pos : request.positions)
        pos = {};
    std::fill(request.lengths.begin(), request.lengths.end(), 0);
    request.batch = next_batch;
}

std::pair<DenseTensor, DenseTensor> Predictor::evaluate(tyloi::models::MtpState& value, const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth, bool cache,
                                       const DenseTensor &supplied_positions,
                                       const DenseTensor &supplied_embeddings, bool history_only) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(!history_only || (cache && layers.size()==1),
        "cache-only MTP priming requires a single predictor layer");
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
#else
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
#endif
    const auto b = ids.size(0), t = ids.size(1);
    const auto capacity = config.maximum;
    const auto count = int64_t(request.lengths.size());
    tyloi::models::require_model(count > 0 && b > 0 && t > 0, "invalid predictor geometry");
    const auto layer = (depth % count + count) % count;
    if (cache && request.batch && request.batch != b)
        reset(value,b);
    const auto start = cache ? request.lengths[layer] : 0;
    tyloi::models::require_model(start >= 0 && t <= capacity - start,
                                 "predictor history exceeds context capacity");
    auto embeds =
        supplied_embeddings.defined() ? supplied_embeddings : embed(ids);
    TYLOINFER_RUNTIME_CHECK(embeds.sizes().vec() ==
                          std::vector<int64_t>({b, t, config.hidden}),
                      "Qwen4-Exp MTP embedding shape mismatch");
    std::array<DenseTensor, 2> pos;
    {
        // Full positions persist across predictor priming tiles and must
        // not occupy expert storage borrowed by this prefill phase.
        tyloi::cuda::ResidentAllocationScope persistent_positions;
        auto current = supplied_positions.defined()
                           ? supplied_positions
                           : ::tyloi::cuda::arange(start, start + t,
                                        ids.options().dtype(::tyloi::kInt32));
        if (current.dim() == 1)
            current = current.reshape({1, 1, t}).expand({3, 1, t}).contiguous();
        else if (current.dim() == 2)
            current = current.unsqueeze(1);
        if (current.dim() == 3 && current.size(0) == 4)
            current = current.narrow(0, 1, 3);
        TYLOINFER_RUNTIME_CHECK(current.dim() == 3 && current.size(0) == 3 &&
                              current.size(-1) == t &&
                              (current.size(1) == 1 || current.size(1) == b),
                          "Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
        current = current.to(::tyloi::kInt32).expand({3, b, t}).contiguous();
        auto full = cache && request.positions[layer].defined()
                        ? ::tyloi::cuda::cat({request.positions[layer].dense(), current}, -1)
                        : current;
        pos = {current, full};
    }
    DenseTensor x;
    {
        DenseTensor e;
        {
            e = ::tyloi::ops::project(rms_norm(embeds, embedding_norm.dense() + 1, config.eps),
                                      embedding_fusion).dense();
        }
        // Release prior-stage tensors before later projections allocate.
        embeds = {};
        DenseTensor streams;
        streams = ::tyloi::ops::project(rms_norm(hidden, hidden_norm.dense() + 1, config.eps)
                                           .reshape({b, t, config.streams, config.hidden}), hidden_fusion).dense();
        x = (streams + e.unsqueeze(-2)).reshape({b, t, config.streams * config.hidden});
    }
    DenseTensor multi;
    {
        auto &block = *layers[layer];
        // Teacher-forced rows each receive the target hidden state. Only
        // the attention input contributes persistent predictor history;
        // attention readout, FFN and final mixer have no consumer here.
        if(history_only) {
            auto mix=block.attention_gr.pre(x);
            block.qsa->forward(*request.layer_states[layer]->qsa,mix.mixed,pos[0],pos[1],true,nullptr,{},true,true);
            multi = DenseTensor{};
        } else {
            auto [attention_input,attention_residual,attention_gate]=block.attention_gr.pre(x);
            auto attention_output=block.qsa->forward(*request.layer_states[layer]->qsa,attention_input,pos[0],pos[1],cache);
            x=block.attention_gr.post(attention_output,attention_residual,attention_gate).dense();
            attention_output={};
            auto [ffn_input,ffn_residual,ffn_gate]=block.ffn_gr.pre(x);
            auto ffn_output=tyloi::ops::ffn(block.ffn,ffn_input);
            multi = block.ffn_gr.post(ffn_output,ffn_residual,ffn_gate).dense();
        }
    }
    x = {};
    auto result = history_only?DenseTensor{}:final_mixer->pre(multi).mixed.dense();
    if (cache) {
        request.positions[layer] = pos[1];
        request.batch = b;
        request.lengths[layer] = start + t;
    }
    return {std::move(result), std::move(multi)};
}

tyloi::models::MtpStep Predictor::step(tyloi::models::MtpState& value, const tyloi::Tensor& input_hidden, const tyloi::Tensor& input_ids,
                                      const tyloi::Tensor& input_positions,
                                      const tyloi::models::ForwardOptions&) const {
    const auto& hidden=input_hidden.dense();
    const auto& ids=input_ids.dense();
    auto positions=input_positions.defined()?input_positions.dense():DenseTensor{};
    if (positions.defined()) {
        if (positions.dim()==1) positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if (positions.dim()==2) positions=positions.unsqueeze(1);
        if (positions.dim()!=3 || positions.size(0)!=3 || positions.size(1)!=ids.size(0) || positions.size(2)!=ids.size(1))
            throw std::invalid_argument("Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
        positions=positions.to(::tyloi::kInt32).contiguous();
    }
    auto result = evaluate(value,hidden, ids, 0, true, positions);
    return {std::make_shared<DenseTensor>(std::move(result.first)),
            std::make_shared<DenseTensor>(std::move(result.second))};
}

void Predictor::history_step(tyloi::models::MtpState& value,DenseTensor hidden,DenseTensor ids,DenseTensor positions,bool scalar_chain) const {
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(positions.defined()) {
        if(positions.dim()==1)positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if(positions.dim()==2)positions=positions.unsqueeze(1);
        TYLOINFER_RUNTIME_CHECK(positions.dim()==3 && positions.size(0)==3 &&
            positions.size(1)==ids.size(0) && positions.size(2)==ids.size(1),"invalid MTP history positions");
        positions=positions.to(::tyloi::kInt32).contiguous();
    }
    if(scalar_chain && ids.size(1)>1) {
        for(int64_t row=0;row<ids.size(1);++row) {
            auto row_positions=positions.defined()?positions.narrow(-1,row,1):DenseTensor{};
            (void)step(value,hidden.narrow(1,row,1),ids.narrow(1,row,1),
                       row_positions.defined()?&row_positions:nullptr);
        }
    } else (void)step(value,hidden,ids,positions.defined()?&positions:nullptr);
}

void Predictor::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= request.position(),
                      "Qwen4-Exp MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index]->qsa)
        layers[index]->qsa->truncate(*request.layer_states[index]->qsa,position);
    if (request.positions[index].defined())
        request.positions[index] = request.positions[index].dense().narrow(-1, 0, position);
    request.lengths[index] = position;
}

bool Predictor::teacher_forced_prompt_prime() const noexcept { return true; }

bool Predictor::target_bootstrap_decode() const noexcept { return true; }

bool Predictor::preserve_output_dtype() const noexcept { return true; }
bool Predictor::stream_prompt_history(std::int64_t tokens) const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_STREAM_PROMPT_HISTORY");
    return tokens>16384 && (!option || option[0]!='0');
}
bool Predictor::host_prompt_history(std::int64_t tokens, std::size_t available,
                                    std::size_t raw_row_bytes) const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_HOST_PROMPT_HISTORY");
    return option ? option[0]!='0' : tyloi::models::Mtp::host_prompt_history(tokens,available,raw_row_bytes);
}
bool Predictor::bootstrap_before_publication() const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_EARLY_BOOTSTRAP");
    return !option || option[0]!='0';
}
std::size_t Predictor::predictor_history_flush_threshold() const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    return !option || option[0]!='0' ? 4 : 512;
}
std::size_t Predictor::predictor_parked_history_flush_threshold() const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    return !option || option[0]!='0' ? 16 : 512;
}
std::size_t Predictor::prefill_workspace_bytes(std::int64_t tokens) const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_CACHE_ONLY_PRIME");
    if (layers.size()!=1 || (option && option[0]=='0') || tokens<=128) return 0;
    return (std::size_t(64)<<20)+std::size_t(tokens)*config.hidden*config.streams*sizeof(float)*12;
}
tyloi::models::MtpStep Predictor::prefill(tyloi::models::MtpState& value,
        const tyloi::Tensor& hidden, const tyloi::Tensor& ids, const tyloi::Tensor& positions,
        const tyloi::models::ForwardOptions& options) const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_CACHE_ONLY_PRIME");
    if (layers.size()!=1 || (option && option[0]=='0'))
        return step(value,hidden,ids,positions,options);
    (void)evaluate(value,hidden.dense(),ids.dense(),0,true,
        positions ? positions.dense() : DenseTensor{}, {},true);
    return {};
}

std::unique_ptr<tyloi::engine::mtp::DepthPolicy> Predictor::depth_policy(
        int maximum_depth, tyloi::engine::mtp::PolicyState& state) const {
    const auto* option = tyloi::cuda::env_lookup("TYLOINFER_MTP_METAL_POLICY");
    const bool batch = !option || option[0] != '0';
    return std::make_unique<tyloi::engine::mtp::AdaptivePolicy>(maximum_depth, &state, true, batch);
}

bool Predictor::supports_session_state(const tyloi::models::MtpState& value) const noexcept {
    if (!value.belongs_to(*this) || layers.empty()) return false;
    const auto& request = static_cast<const PredictorState&>(value);
    if (request.layer_states.size() != layers.size()) return false;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (!request.layer_states[i] ||
            (layers[i]->qsa && (!request.layer_states[i]->qsa ||
                !layers[i]->qsa->supports_session(*request.layer_states[i]->qsa)))) return false;
    }
    return true;
}

MtpSessionState Predictor::capture_session_state(const tyloi::models::MtpState& value, int64_t target_position,const tyloi::Tensor& input_hidden) const {
    const auto& request=this->state<PredictorState>(value);
    const auto& hidden=input_hidden.dense();
    const auto position=target_position-1;
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(supports_session_state(value) && position>0 && position<=request.position() &&
        hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==1 && hidden.size(2)==hidden_norm.numel(),
        "Qwen4 MTP session boundary is unavailable");
#else
    TYLOINFER_RUNTIME_CHECK(supports_session_state(value) && position>0 && position<=request.position() &&
        hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==1 && hidden.size(2)==hidden_norm.numel(),
        "Qwen4 MTP session boundary is unavailable");
#endif
    MtpSessionState state;state.cache_pos=position;
    auto& snapshot=state.payload.emplace<tyloi::models::qwen4_exp::MtpSnapshot>();
    for(std::size_t i=0;i<layers.size();++i) {
        const auto count=std::min(position,request.lengths[i]);
        snapshot.lengths.push_back(count);
        snapshot.layers.push_back(count?capture_layer_session(*layers[i],*request.layer_states[i],count,state.bytes):tyloi::models::qwen4_exp::LayerSnapshot{});
        ::tyloi::Tensor pos;
        if(count) {
            pos=std::make_shared<DenseTensor>(request.positions[i].dense().narrow(-1,0,count).clone());
            state.bytes+=session_tensor_bytes(pos);
        }
        snapshot.positions.push_back(std::move(pos));
    }
    auto saved_hidden=hidden.clone();state.bytes+=session_tensor_bytes(saved_hidden);
    state.last_target_hidden=std::make_shared<DenseTensor>(std::move(saved_hidden));
    return state;
}

} // namespace tyloi::cuda::qwen4_exp

namespace tyloi::cuda {
using qwen4_exp::Model;
namespace {

::tyloi::DenseTensor qwen4_last_hidden(::tyloi::DenseTensor hidden) {
    return hidden.index({::tyloi::indexing::Slice(), -1, ::tyloi::indexing::Slice()});
}
::tyloi::DenseTensor qwen4_logits_from_hidden(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden) {
    const auto cap = model.metadata.final_logit_softcapping;
    auto logits = model.logits(hidden).dense();
    return cap > 0.0 ? ::tyloi::cuda::tanh(logits / cap) * cap : logits;
}
::tyloi::DenseTensor qwen4_finalize_hidden(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden,
                                           std::int64_t, std::int64_t) {
    return model.final_mixer->pre(hidden, tyloi::ops::ResidualRole::final).mixed.dense();
}

::tyloi::DenseTensor qwen4_next_token(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden) {
    auto* window=DecodeWindow::recording();
    if(window)window->mark(-1,"lm_head_begin");
    auto projected=model.logits(hidden).dense();
    if(window)window->mark(-1,"lm_head_projection_end");
    auto output=::tyloi::cuda::argmax(projected,-1).to(::tyloi::kInt64);
    if(window)window->mark(-1,"lm_head_sample_end");
    return output;
}

void qwen4_validate_forward(const qwen4_exp::Model& model, std::int64_t tokens,
                            std::int64_t position, bool cache_override) {
    tyloi::models::require_model(tokens > 0 &&
        position + tokens <= model.metadata.max_position_embeddings && !cache_override,
        "Qwen4 requires unpadded causal cache positions");
}

::tyloi::DenseTensor qwen4_forward(const qwen4_exp::Model& model, qwen4_exp::State& state,
                                   tyloi::models::Input input) {
    const auto primary = model.primary_device;
    tyloi::models::require_model(bool(input.ids), "Qwen4 forward requires token ids");
    auto ids = model.device_ids(input.ids.dense());
    auto embeddings = input.embeddings
        ? tyloi::ops::contiguous(tyloi::ops::copy(input.embeddings,
            ::tyloi::Device(::tyloi::kCUDA, primary))).dense()
        : model.embed_tokens(ids);
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                                     embeddings.size(1) == T && embeddings.size(2) == model.hidden_size(),
                                 "inputs_embeds shape must match [batch,tokens,hidden_size]");
    qwen4_validate_forward(model, T, state.cache_pos, bool(input.cache_positions));
    tyloi::models::require_model(state.speculative_start < 0 || state.speculative_suffix_forward,
                                 "commit or roll back the pending speculative pass before forwarding");
    const auto confirmed = input.confirmed;
    tyloi::models::require_model(
        confirmed >= 0 && (confirmed == 0 ||
            (confirmed < T && B == 1 && state.cache_pos > 0 &&
             !input.cache_positions &&
             model.supports_speculation())),
        "unsupported speculative backbone geometry");
    if (confirmed > 0) {
        tyloi::models::require_model(state.cache_pos + T <= model.max_position_embeddings(),
                                     "speculative pass exceeds context capacity");
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = confirmed;
    }
    if ((state.batch != 0 && state.batch != B) || state.cache_pos == 0)
        state.reset(B);
    auto cache_positions = ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T,
                                                 ::tyloi::TensorOptions()
                                                     .device(::tyloi::kCUDA)
                                                     .dtype(::tyloi::kInt64));
    auto current = input.positions
        ? model.device_ids(input.positions.dense())
        : (state.decode_position_delta == 0 ? cache_positions
                                            : cache_positions + state.decode_position_delta);
    if ((current.dim() == 2 || current.dim() == 3) && current.size(0) == 4)
        current = current.narrow(0, 1, 3);
    auto full_positions = state.positions.defined()
        ? ::tyloi::cuda::cat({state.positions.dense(), current}, -1)
        : current;
    tyloi::models::require_model((current.dim() == 1 || (current.dim() == 2 && current.size(0) == 3) ||
                                     (current.dim() == 3 && current.size(0) == 3 && current.size(1) == B)) &&
                                     current.size(-1) == T,
                                 "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    DenseTensor host_ids, cpu_positions;
    const bool pinned = model.host_tokens || std::any_of(model.layers.begin(), model.layers.end(),
        [](const auto& layer) { return layer->cpu_offloaded; });
    if (pinned) {
        host_ids = ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        cpu_positions = current.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        for (std::size_t i=0;i<model.layers.size();++i)
            if (model.layers[i]->ple)
                model.layers[i]->ple->prefetch_tokens(state.layers[i]->ple, host_ids);
    }
    auto hidden = model.repeat_hidden(
        tyloi::ops::contiguous(tyloi::ops::copy(embeddings,
            ::tyloi::Device(::tyloi::kCUDA, primary))).dense(), model.metadata.hc_mult);
    state.begin_forward();
    const auto trace_layer = [&] {
        if (!input.trace) return;
        input.trace->push_back(
            std::make_shared<DenseTensor>(tyloi::ops::copy(hidden,
                ::tyloi::Device(::tyloi::kCUDA, primary)).dense().to(::tyloi::kFloat32).clone()));
    };
    if (input.trace)
        input.trace->push_back(std::make_shared<DenseTensor>(hidden.to(::tyloi::kFloat32).clone()));
    for (std::size_t i=0;i<model.layers.size();++i) {
        auto& block = model.layers[i];
        MfqCudaGuard guard(block->cuda_device);
        auto local_ids = block->cpu_offloaded
            ? host_ids
            : tyloi::ops::copy(ids, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        auto local_pos = block->cpu_offloaded
            ? cpu_positions
            : tyloi::ops::copy(current, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        auto local_full_positions = tyloi::ops::copy(full_positions,
            ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        hidden = block->cpu_offloaded
            ? hidden.to(::tyloi::kCPU).contiguous()
            : tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        hidden = qwen4_exp::forward(*block, *state.layers[i],
                                std::move(hidden), local_ids, local_pos, local_full_positions,
                                confirmed, host_ids);
        trace_layer();
    }
    state.finish_forward(full_positions, B, T);
    state.cache_pos += T;
    hidden = tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, primary)).dense();
    if (input.raw_hidden) *input.raw_hidden = hidden;
    auto finalized = qwen4_finalize_hidden(model, std::move(hidden), B, T);
    return finalized;
}

} // namespace
bool Model::prefetch_token_rows(State& state,const ::tyloi::Tensor& host_ids) {
    if (!state.reuse_decode_graphs || !state.graph_registry) return false;
    bool issued=false;
    for(std::size_t i=0;i<layers.size();++i) {
        const auto& block=layers[i];
        if(block->ple && block->ple->graph_supported()) {
            block->ple->prefetch_tokens(state.layers[i]->ple,host_ids.dense(),true);
            issued=true;
        }
    }
    return issued;
}

struct Qwen4DecodeGraph {
    bool startup_warmup=false;
    struct PleStage {
        qwen4_exp::Ple* ple;
        qwen4_exp::Ple::State* state;
        ::tyloi::DenseTensor rows;
        std::shared_ptr<RowStage> transfer;
        std::vector<qwen4_exp::NgramEmbedding::Context> prefix_context;
        std::unique_ptr<DecodeWindow::Signal> signal;
        PleStage(qwen4_exp::Ple* p,qwen4_exp::Ple::State& request,int tokens,int hidden,DecodeWindow& window)
            :ple(p),state(&request),signal(window.signal()) {
            transfer=ple->graph_stage(*state,tokens,::tyloi::Device{::tyloi::kCUDA,mfq_current_cuda_device()});
            signal->set_mapped_rows(transfer->mapped_rows());
            rows=transfer->output().dense().reshape({1,tokens,hidden});
        }
    };
    struct Cell {
        ::tyloi::DenseTensor ids,positions,cache_positions,host_ids,output,raw_hidden,logits,greedy_host,compact_host;
        unsigned used_kinds=0;
        std::vector<std::unique_ptr<PleStage>> ple;
        std::vector<GraphCheckpoint> checkpoints;
        DecodeWindow window;
        Cell(int tokens,int hidden,const std::vector<std::unique_ptr<qwen4_exp::Layer>>& weights,
                const std::vector<std::unique_ptr<qwen4_exp::LayerState>>& layers,int confirmed,
                std::vector<GraphCheckpoint>& shared_checkpoints)
            :window(::tyloi::cuda::options::layer_timings()) {
            const auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64);
            ids=window.buffer({1,tokens},options,true).dense();
            positions=window.buffer({tokens},options,true).dense();
            cache_positions=window.buffer({tokens},options,true).dense();
            host_ids=window.buffer({1,tokens},options.device(::tyloi::kCPU),true).dense();
            greedy_host=window.buffer({tokens},options.device(::tyloi::kCPU).pinned_memory(true)).dense();
            compact_host=window.buffer({tokens,64},options.device(::tyloi::kCPU).pinned_memory(true)).dense();
            if(confirmed && shared_checkpoints.empty())shared_checkpoints.resize(weights.size());
            std::size_t layer=0;
            for(const auto& block:weights) {
                auto& b=*block;
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),layers[layer]->ple,tokens,hidden,window));
                if(confirmed) {
                    auto& shared=shared_checkpoints[layer];
                    GraphCheckpoint checkpoint;
                    if(b.gdn) {
                        const auto state=b.gdn->graph_state(layers[layer]->gdn);
                        while(shared.gdn_conv.size()<std::size_t(tokens-confirmed)) {
                            shared.gdn_conv.push_back(::tyloi::cuda::empty_like(state[0]->dense()));
                            shared.gdn_state.push_back(::tyloi::cuda::empty_like(state[1]->dense()));
                        }
                        checkpoint.gdn_conv.assign(shared.gdn_conv.begin(),shared.gdn_conv.begin()+tokens-confirmed);
                        checkpoint.gdn_state.assign(shared.gdn_state.begin(),shared.gdn_state.begin()+tokens-confirmed);
                    }
                    if(b.ple) {
                        while(shared.ple_conv.size()<std::size_t(tokens-confirmed))
                            shared.ple_conv.push_back(::tyloi::cuda::empty_like(b.ple->conv_state(layers[layer]->ple).dense()));
                        checkpoint.ple_conv.assign(shared.ple_conv.begin(),shared.ple_conv.begin()+tokens-confirmed);
                    }
                    checkpoints.push_back(std::move(checkpoint));
                }
                ++layer;
            }
        }
    };
    DecodeStream stream;
    std::map<std::tuple<int,int,int>,std::unique_ptr<Cell>> cells;
    // Verification widths execute serially and commit/rollback before the next
    // window. Their snapshots can share stable allocations across graph shapes.
    std::vector<GraphCheckpoint> shared_checkpoints;
    qwen4_exp::State& request;
    std::vector<std::pair<const tyloi::ops::Qsa*,tyloi::ops::Qsa::State*>> qsa;
    ::tyloi::DenseTensor position_storage;
    int64_t kv_capacity=0;
    bool failed=false;
    explicit Qwen4DecodeGraph(const qwen4_exp::Model& model,qwen4_exp::State& state)
        :request(state) {
        for(std::size_t i=0;i<model.layers.size();++i)
            if(auto* attention=model.layers[i]->qsa.get())qsa.emplace_back(attention,request.layers[i]->qsa.get());
    }
    void restore_legacy() {
        cells.clear();shared_checkpoints.clear();
        for(auto [attention,state]:qsa)attention->leave_graph(*state);
        for(auto& layer:request.layers)layer->retain_graph_storage=false;
    }
    bool reusable() const {
        return !failed && !qsa.empty() && std::all_of(qsa.begin(),qsa.end(),
            [](const auto& attention){return attention.first->graph_capacity(*attention.second)>0;});
    }

    tyloi::StepSequence<tyloi::models::Decoded> run(const qwen4_exp::Model& model,
            tyloi::models::Input input, tyloi::models::DecodeOptions options) {
        auto ids=model.device_ids(input.ids.dense());
        const auto tokens=static_cast<int>(ids.size(1));
        const auto confirmed=static_cast<int>(input.confirmed);
        const bool want_raw=input.raw_hidden!=nullptr;
        const auto projection=options.projection;
        int kind=projection==tyloi::models::DecodeOptions::Projection::greedy ? 2 :
            projection==tyloi::models::DecodeOptions::Projection::logits ? 1 :
            projection==tyloi::models::DecodeOptions::Projection::hidden ? 4 : 3;
        const bool want_logits=kind==4;
        tyloi::models::Decoded result;
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        stream.require_current();
        // These consumers use the same single-row backbone. Keep its hidden
        // state, logits and selected token in one graph instead of reserving
        // a complete set of layer buffers for each output representation.
        const int result_kind=kind;
        if(tokens==1 && kind>=1 && kind<=4)kind=4;
        co_yield tyloi::GraphPhase::ready;
        const auto growth=[&](std::int64_t capacity) {
            std::vector<std::pair<tyloi::Device,std::size_t>> result;
            for(std::size_t i=0;i<model.layers.size();++i) {
                const auto& block=*model.layers[i];
                if(!block.qsa)continue;
                const auto bytes=block.qsa->graph_growth_bytes(*request.layers[i]->qsa,capacity);
                if(!bytes)continue;
                const tyloi::Device device(tyloi::kCUDA,block.cuda_device);
                auto it=std::find_if(result.begin(),result.end(),[&](const auto& item){return item.first==device;});
                if(it==result.end())result.emplace_back(device,bytes);
                else {
                    if(it->second>std::numeric_limits<std::size_t>::max()-bytes)
                        throw std::overflow_error("Qwen4 graph growth overflow");
                    it->second+=bytes;
                }
            }
            return result;
        };
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,growth(request.cache_pos+tokens)};
        tyloi::ops::Qsa::Graph graph({{qsa, request.cache_pos, model.max_position_embeddings(),
                                      request.decode_reserve_until}}, false);
        graph.admit(tokens, kv_capacity, want_raw);
        if (graph.clear_required()) {
            co_yield tyloi::GraphPhase::ready;
            co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,request.graph_exit_growth()};
            restore_legacy();
            graph.layout();
        }
        kv_capacity = graph.rows().front().capacity;
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,growth(kv_capacity)};
        const auto observed=[&] {
            std::vector<::tyloi::Tensor> result;
            for(std::size_t i=0;i<model.layers.size();++i) {
                auto& layer=*request.layers[i];const auto& weights=*model.layers[i];
                for(auto* tensor:layer.graph_state(weights))result.push_back(*tensor);
                if(weights.qsa)for(auto* tensor:weights.qsa->graph_state(*layer.qsa))result.push_back(*tensor);
            }
            return result;
        };
        auto& owner=cells[{tokens,kind,confirmed}];
        if(owner && owner->window.valid() && !owner->window.matches(observed()))owner.reset();
        if(!owner) {
            const Device device(kCUDA, model.primary_device);
            const auto ple_count = std::count_if(model.layers.begin(), model.layers.end(),
                [](const auto& layer) { return bool(layer->ple); });
            auto fixed = DecodeWindow::fixed_growth(device, ::tyloi::cuda::options::layer_timings(), kind == 4 && tokens > 1, ple_count);
            add_growth(fixed, device, std::size_t(tokens) * sizeof(int64_t), 3);
            add_growth(fixed, Device(kCPU), std::size_t(tokens) * sizeof(int64_t), 66);
            for (std::size_t i = 0; i < model.layers.size(); ++i) {
                const auto& layer = *model.layers[i];
                if (layer.ple) for (const auto& [target, bytes] : layer.ple->graph_stage_growth(tokens, device))
                    add_growth(fixed, target, bytes);
                if (!confirmed) continue;
                const GraphCheckpoint empty;
                const auto& shared = i < shared_checkpoints.size() ? shared_checkpoints[i] : empty;
                const auto count = std::size_t(tokens - confirmed);
                if (layer.gdn) {
                    const auto tensors = layer.gdn->graph_state(request.layers[i]->gdn);
                    if (shared.gdn_conv.size() < count)
                        add_growth(fixed, tensors[0]->device(), tensors[0]->nbytes(), count - shared.gdn_conv.size());
                    if (shared.gdn_state.size() < count)
                        add_growth(fixed, tensors[1]->device(), tensors[1]->nbytes(), count - shared.gdn_state.size());
                }
                if (layer.ple && shared.ple_conv.size() < count) {
                    const auto& tensor = layer.ple->conv_state(request.layers[i]->ple);
                    add_growth(fixed, tensor.device(), tensor.nbytes(), count - shared.ple_conv.size());
                }
            }
            co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(fixed)};
            owner=std::make_unique<Cell>(tokens,model.config.hidden,model.layers,request.layers,confirmed,shared_checkpoints);
        }
        owner->window.set_profile_geometry(want_raw?"mtp_target":"target_decode",tokens);
        auto& cell=*owner;
        if(kind==4 && tokens>1)cell.window.set_compact_sampling(options.compact_sampling,"target",tokens);
        // A retained graph's first replay can still pay cold transfer/cache
        // costs after prefill. Exclude that replay from steady MTP policy data.
        result.initialized=!cell.window.valid() || !(cell.used_kinds&(1u<<result_kind));
        if(input.host_ids) {
            TYLOINFER_RUNTIME_CHECK(!input.host_ids.dense().is_cuda() && input.host_ids.dense().scalar_type()==::tyloi::kInt64 &&
                input.host_ids.dense().sizes()==ids.sizes(),"Qwen4 graph host token geometry changed");
            cell.host_ids.copy_(input.host_ids.dense());
        } else cell.host_ids.copy_(ids.to(::tyloi::kCPU,::tyloi::kInt64));
        cell.ids.copy_(ids);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,request.cache_pos,request.decode_position_delta);
        try {
            // PLE depends only on token IDs and the committed ngram context.
            // Start its read before waiting for adaptive expert exchanges,
            // rather than leaving that CPU interval outside the I/O overlap.
            // Task reset repeats issue() idempotently; it never advances the
            // context, which is still committed by upload_graph().
            const auto* early_ple=tyloi::cuda::env_lookup("TYLOINFER_PLE_EARLY_ISSUE");
            if(!early_ple || early_ple[0]!='0')
                for(const auto& stage:cell.ple)stage->ple->prefetch_tokens(*stage->state,cell.host_ids);
            co_yield tyloi::GraphPhase::ready;
            // Prefix restoration and eager prefill update the same KV/index
            // storage; refresh pooled keys even when this graph is already valid.
            graph.prepare(0, request.positions);
            if(!cell.window.valid()) {
                co_yield tyloi::GraphPhase::prepare;
                struct Saved {::tyloi::Tensor* target;::tyloi::DenseTensor saved;const void* address;};
                std::vector<Saved> states;
                std::vector<std::pair<Device, std::size_t>> copy_growth;
                if (!startup_warmup) for (std::size_t i = 0; i < model.layers.size(); ++i)
                    for (auto* value : request.layers[i]->graph_state(*model.layers[i]))
                        add_growth(copy_growth, value->device(), value->nbytes());
                if (!copy_growth.empty())
                    co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(copy_growth)};
                if(!startup_warmup) for(std::size_t i=0;i<model.layers.size();++i)
                        for(auto* value:request.layers[i]->graph_state(*model.layers[i]))
                            states.push_back({value,value->dense().clone(),value->dense().data_ptr()});
                auto restore=[&] {
                    for(const auto& state:states) {
                        TYLOINFER_RUNTIME_CHECK(state.target->dense().data_ptr()==state.address,"MFQ capture changed recurrent storage");
                        state.target->dense().copy_(state.saved);
                    }
                };
                auto recording=cell.window.begin_capture();
                try { while(recording->next()) {
                    std::exception_ptr failure;
                    bool warm = false;
                    try {
                    cell.window.mark(-1,"window_begin");
                    auto hidden=model.repeat_hidden(model.embed_tokens(cell.ids),model.metadata.hc_mult);
                    std::size_t ple_index=0;int layer=0;
                    std::optional<tyloi::ops::Residual::Result> pending_residual;
                    const auto chain_setting=::tyloi::cuda::options::gr_chain_ffn_post();
                    const bool chain_post=(!chain_setting || *chain_setting) && tokens<=8;
                    for(const auto& block:model.layers) {
                        cell.window.mark(layer,"block_begin");
                        auto& b=*block;::tyloi::DenseTensor rows;
                        if(b.ple) {
                            cell.window.mark(layer,"ple_begin");
                            auto* stage=cell.ple[ple_index++].get();rows=stage->rows;
                            cell.window.enroll({stage,
                                [&cell,stage] {stage->signal->reset();stage->ple->prefetch_tokens(*stage->state,cell.host_ids);},
                                [this,&cell,stage,confirmed] {
                                    stage->signal->wait_route();
                                    auto guard=cell.window.transfer_scope();
                                    if(confirmed) {
                                        stage->prefix_context.clear();
                                        for(int i=confirmed;i<cell.ids.size(1);++i)
                                            stage->prefix_context.push_back(stage->ple->graph_prefix_context(*stage->state,cell.host_ids,i));
                                    }
                                    stage->ple->upload_graph(*stage->state,cell.host_ids,*stage->transfer);
                                    stage->signal->publish(stage->transfer->mapped_rows());
                                },{},[stage]{stage->signal->cancel();}});
                            stage->signal->wait_device();
                            stage->transfer->decode();
                            cell.window.mark(layer,"ple_end");
                        }
                        cell.window.mark(layer,"attention_begin");
                        std::optional<tyloi::ops::Residual::Result> next_residual;
                        bool defer=false;
                        if(chain_post && size_t(layer+1)<model.layers.size()) {
                            auto& next=*model.layers[layer+1];
                            defer=!next.ple && next.attention_gr.can_chain(pending_residual?pending_residual->residual:hidden);
                        }
                        {
                            auto& layer_state=*request.layers[layer];
                            auto ffn_result=graph_attention(b,layer_state,std::move(hidden),cell.ids,cell.positions,
                                cell.cache_positions,rows,layer,confirmed,confirmed?&cell.checkpoints[layer]:nullptr,pending_residual);
                            const auto& ffn_input=ffn_result.mixed;
                            auto ffn_output=tyloi::ops::ffn(b.ffn,ffn_input).dense();
                            if(defer) {
                                next_residual=std::move(ffn_result);
                                hidden=std::move(ffn_output);
                            } else hidden=b.ffn_gr.post(ffn_output,ffn_result.residual,ffn_result.gate).dense();
                        }
                        pending_residual=std::move(next_residual);
                        cell.window.mark(layer++,"block_end");
                    }
                    if(kind>=3)cell.raw_hidden=hidden.clone();
                    hidden=qwen4_finalize_hidden(model,std::move(hidden),1,tokens);
                    if(kind>=3) {
                        cell.output=std::move(hidden);
                        if(kind==4) {
                            cell.logits=qwen4_logits_from_hidden(model, cell.output);
                            cell.window.sample(cell.logits,cell.greedy_host,cell.compact_host,tokens);
                        }
                    }
                    else if(kind==2)cell.output=qwen4_next_token(model,qwen4_last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.logits(qwen4_last_hidden(std::move(hidden))).dense();
                    else cell.output=qwen4_logits_from_hidden(model, std::move(hidden));
                    cell.window.mark(-1,"window_end");
                    warm=recording->finish();
                    } catch (...) { failure = std::current_exception(); }
                    if (failure) {
                        if (!recording->retry(failure)) std::rethrow_exception(failure);
                        warm = true;
                    }
                    restore();
                    if(warm) {
                        cell.output={};cell.raw_hidden={};cell.logits={};
                        recording->release_warm_outputs();
                        recording->pause();
                        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, recording->growth()};
                        recording->resume();
                    }
                }
                recording->complete();
                cell.window.retain_result(cell.output);
                cell.window.retain_result(cell.raw_hidden);
                cell.window.retain_result(cell.logits);
                cell.window.retain(observed());
                } catch(...) {recording.reset();restore();throw;}
                ++request.decode_graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=target rows="<<tokens<<" kind="<<kind
                    <<" confirmed="<<confirmed<<" segments="<<cell.window.segments()<<" count="<<request.decode_graph_captures
                    <<" ms="<<cell.window.capture_ms()<<'\n';
            }
            cell.window.run();cell.used_kinds|=1u<<result_kind;
            if(confirmed) {
                // Captured copies save the confirmed prefix. Bind their owned
                // storage only after replay, so generic commit/rollback keeps
                // exactly the same transaction lifecycle as eager execution.
                std::size_t ple_index=0,layer=0;
                for(const auto& block:model.layers) {
                    auto& b=*block;
                    const auto& checkpoint=cell.checkpoints[layer];
                    auto& layer_state=*request.layers[layer++];
                    const auto position=request.cache_pos+confirmed;
                    if(b.gdn)b.gdn->bind_graph_checkpoint(layer_state.gdn,checkpoint.gdn_conv,checkpoint.gdn_state,position);
                    if(b.ple)b.ple->bind_graph_checkpoint(layer_state.ple,checkpoint.ple_conv,cell.ple[ple_index++]->prefix_context,position);
                }
                request.speculative_start=request.cache_pos;request.speculative_confirmed=confirmed;
                request.speculative_prefix_retention=true;
            }
            for(std::size_t i=0;i<model.layers.size();++i)
                if(model.layers[i]->qsa)model.layers[i]->qsa->advance_graph(*request.layers[i]->qsa,tokens);
            request.cache_pos+=tokens;
            request.positions=append_decode_positions(request.positions ? request.positions.dense() : DenseTensor{},cell.positions,position_storage,
                model.max_position_embeddings());request.batch=1;
            // MTP can retain history until a later probe. Each step must own its
            // raw residual rather than alias the next replay's capture buffer.
            if(want_raw)result.raw=cell.raw_hidden.clone();
            if(kind==4 && tokens>1 && options.compact_sampling && cell.logits.size(-1)>=64)
                result.compact=cell.compact_host;
            if(want_logits) {
                result.logits=cell.logits.clone();
                const auto* first=cell.greedy_host.data_ptr<int64_t>();
                result.tokens.assign(first,first+tokens);
            }
            if(tokens==1 && result_kind==1)result.logits=cell.logits.reshape({1,-1});
            else if(tokens==1 && result_kind==2)result.ids=cell.greedy_host;
            else result.hidden=cell.output;
            co_yield std::move(result);
        } catch(...) {failed=true;throw;}
    }
};
std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(
        const qwen4_exp::Model&, const qwen4_exp::State& state) {
    std::vector<std::pair<std::string,double>> result;
    if (state.graph_registry) for (const auto& [key,cell] : state.graph_registry->cells) {
        const auto prefix = std::string("cuda_graph_profile_target_rows_") +
            std::to_string(std::get<0>(key)) + "_kind_" + std::to_string(std::get<1>(key)) +
            "_confirmed_" + std::to_string(std::get<2>(key)) + "_";
        append_graph_profile_metrics(cell->window, prefix, result);
    }
    return result;
}

tyloi::StepSequence<tyloi::models::Decoded> qwen4_exp::Model::decode(
        tyloi::models::Input input, tyloi::models::State& value,
        const tyloi::models::ForwardOptions&, const tyloi::models::DecodeOptions& options) const {
    auto& state=dynamic_cast<qwen4_exp::State&>(value);
    if (!options.enabled || !streamed_decode_graphs() || !input.ids || input.embeddings ||
        input.positions || input.lengths || input.cache_positions || input.image_mask || input.suffix ||
        state.speculative_start>=0) co_return;
    auto ids=input.ids.dense();
    if(ids.dim()==1) { ids=ids.unsqueeze(0); input.ids=ids; }
    const auto kind=options.projection;
    const bool hidden=kind==tyloi::models::DecodeOptions::Projection::hidden ||
                      kind==tyloi::models::DecodeOptions::Projection::hidden_only;
    if(state.cache_pos==0 && (!hidden || !state.graph_registry || !state.graph_registry->reusable())) co_return;
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 ||
        (state.cache_pos!=0 && state.positions.dim()!=1) || input.confirmed<0 ||
        (input.confirmed && input.confirmed>=ids.size(1))) co_return;
    if(options.routed && !tq_graph_supports_tokens(ids.size(1)))co_return;
    for(const auto& block:layers)
        if((hidden || options.routed) && (!block->ffn.qwen_tq ||
            (block->ple && !block->ple->graph_supported()))) co_return;
    TYLOINFER_RUNTIME_CHECK(state.cache_pos+ids.size(1)<=max_position_embeddings(),"MFQ window exceeds context");
    state.decode_reserve_until=options.reserve_until;
    if(!state.graph_registry)state.graph_registry=std::make_unique<Qwen4DecodeGraph>(*this,state);
    auto job=state.graph_registry->run(*this,std::move(input),options);
    while(auto step=job.next()) co_yield std::move(step);
}

void qwen4_release_decode_graphs(const qwen4_exp::Model& model,qwen4_exp::State& state) {
    if(state.graph_registry)state.graph_registry->restore_legacy();
    state.graph_registry.reset();
    for(auto& layer:state.layers)layer->retain_graph_storage=false;
}

tyloi::StepSequence<tyloi::models::Decoded> qwen4_exp::Model::warmup(
        tyloi::models::State& value, const tyloi::models::DecodeOptions& options) const {
    auto& state=dynamic_cast<qwen4_exp::State&>(value);
    const auto* reuse=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_REUSE_GRAPHS");
    state.reuse_decode_graphs=!reuse || reuse[0]!='0';
    const auto* setting=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_GRAPHS");
    if(!state.reuse_decode_graphs || (setting && setting[0]=='0') || !options.enabled ||
        max_position_embeddings()<32 || !streamed_decode_graphs()) co_return;
    for(std::size_t i=0;i<layers.size();++i) {
        const auto& block=*layers[i];
        if(!block.ffn.qwen_tq || (block.qsa && !block.qsa->graph_capacity(*state.layers[i]->qsa)) ||
            (block.ple && !block.ple->graph_supported())) co_return;
    }
    struct Reset {
        qwen4_exp::State& state;
        ~Reset() {
            if(state.graph_registry) state.graph_registry->startup_warmup=false;
            try { state.reset(1); } catch (...) {}
        }
    } reset{state};
    const auto start=std::chrono::steady_clock::now();
    co_yield tyloi::GraphPhase::warmup;
    const auto tensor_options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64);
    std::vector<int64_t> warm_ids(16);
    for(std::size_t i=0;i<warm_ids.size();++i)warm_ids[i]=int64_t((i*7919+17)%metadata.vocab_size);
    auto ids=::tyloi::cuda::tensor(warm_ids,tensor_options.device(::tyloi::kCPU)).to(::tyloi::kCUDA).reshape({1,16});
    co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,state.reset_growth(1)};
    state.reset(1);
    tyloi::models::ForwardOptions forward_options;
    forward_options.primary=primary_device;
    for(int row=0;row<(options.routed?1:16);++row) {
        tyloi::models::Input input;
        input.ids=ids.narrow(1,row,1);
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit,nullptr,state.growth(input,forward_options)};
        (void)forward(std::move(input),state,forward_options);
        co_yield tyloi::StepState::advanced;
    }
    co_yield tyloi::GraphPhase::target;
    if(!state.graph_registry)state.graph_registry=std::make_unique<Qwen4DecodeGraph>(*this,state);
    state.graph_registry->startup_warmup=true;
    for(auto projection : {tyloi::models::DecodeOptions::Projection::greedy,
                           tyloi::models::DecodeOptions::Projection::logits}) {
        tyloi::models::Input input;
        input.ids=ids.narrow(1,0,1);
        auto policy=options;
        policy.projection=projection;
        auto job=decode(std::move(input),state,{},policy);
        while(auto step=job.next()) {
            if(step.graph || step.workspace || step.layer || !step.value) co_yield std::move(step);
        }
    }
    state.reset(1);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
    std::cerr<<"runtime_qwen4_graph_warmup target_captures="<<state.decode_graph_captures<<" ms="
        <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
    co_yield tyloi::GraphPhase::settle;
}

std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(
        const qwen4_exp::Model& model,const qwen4_exp::State& state,const tyloi::models::Mtp* mtp) {
    // Shared checkpoint views and recurrent state aliases count once. These
    // are existing cache tensors; collecting telemetry never copies GPU data.
#ifdef TYLOINFER_NATIVE_CUDA
    std::map<int,std::vector<::tyloi::DenseTensor>> allocations;
    const auto add=[&](const ::tyloi::DenseTensor& tensor) {
        if(tensor.defined() && tensor.is_cuda()) allocations[tensor.get_device()].push_back(tensor);
    };
#else
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add=[&](const ::tyloi::DenseTensor& tensor) {
        if(!tensor.defined() || !tensor.is_cuda())return;
        auto& bytes=allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(tensor.data_ptr())}];
        bytes=std::max(bytes,std::size_t(tensor.numel())*tensor.element_size());
    };
#endif
    const auto block_cache=[&](const qwen4_exp::Layer& block,const qwen4_exp::LayerState& layer) {
        if(block.qsa)for(const auto& tensor:block.qsa->resident_cache_tensors(*layer.qsa))add(tensor.dense());
        if(block.gdn) { add(layer.gdn.conv.dense()); add(layer.gdn.recurrent.dense()); }
        if(block.ple && layer.ple.conv.defined())add(layer.ple.conv.dense());
    };
    for(std::size_t i=0;i<model.layers.size();++i)block_cache(*model.layers[i],*state.layers[i]);
    if(state.positions) add(state.positions.dense());
    if(state.graph_registry)add(state.graph_registry->position_storage);
    if(state.graph_registry)for(const auto& [key,cell]:state.graph_registry->cells)
        for(const auto& checkpoint:cell->checkpoints) {
            for(const auto& tensor:checkpoint.gdn_conv)add(tensor.dense());
            for(const auto& tensor:checkpoint.gdn_state)add(tensor.dense());
            for(const auto& tensor:checkpoint.ple_conv)add(tensor);
        }
    if(const auto* predictor=dynamic_cast<const qwen4_exp::Predictor*>(mtp); predictor && state.predictor && state.predictor->belongs_to(*predictor)) {
        const auto& request=static_cast<const qwen4_exp::PredictorState&>(*state.predictor);
        for(std::size_t i=0;i<predictor->layers.size();++i)
            block_cache(*predictor->layers[i],*request.layer_states[i]);
        for(const auto& positions:request.positions)if(positions)add(positions.dense());
    }
    std::size_t bytes=0;
    std::map<int,std::size_t> devices;
#ifdef TYLOINFER_NATIVE_CUDA
    for(const auto& [device,tensors]:allocations) {
        std::vector<const ::tyloi::DenseTensor*> buffers;
        for(const auto& tensor:tensors) buffers.push_back(&tensor);
        const auto size=::tyloi::backing_bytes(buffers);
        bytes+=size; devices[device]=size;
    }
#else
    for(const auto& [address,size]:allocations) { bytes+=size; devices[address.first]+=size; }
#endif
    std::vector<std::pair<std::string,double>> result{{"kv_cache_bytes",double(bytes)},
        {"kv_cache_contexts",state.cache_pos>0?double(state.batch):0.0}};
    for(const auto& [device,size]:devices)result.emplace_back("gpu_"+std::to_string(device)+"_live_kv_bytes",double(size));
    return result;
}

} // namespace tyloi::cuda

namespace tyloi::cuda {
struct Qwen4BatchRequestState::Impl {
    const qwen4_exp::Model& model;
    std::vector<std::unique_ptr<qwen4_exp::LayerState>> layers;
    ::tyloi::DenseTensor positions, position_storage;
    int64_t cache_position = 0, decode_position_delta = 0;
    explicit Impl(const qwen4_exp::Model& owner) : model(owner) {}
    std::size_t growth_bytes(int64_t tokens) const {
        std::size_t bytes = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& weights = *model.layers[i];
            const auto& state = *layers[i];
            if (weights.qsa) bytes += weights.qsa->growth_bytes(*state.qsa,tokens);
            if (weights.gdn) bytes += weights.gdn->growth_bytes(state.gdn);
        }
        return bytes;
    }
};

Qwen4BatchRequestState::Qwen4BatchRequestState(qwen4_exp::Model& model)
    : state_(std::make_unique<Impl>(model)) {
    const auto primary = model.primary_device;
    for (const auto& block : model.layers) {
        TYLOINFER_RUNTIME_CHECK(!block->cpu_offloaded && block->cuda_device == primary,
                          "Flash-Next continuous batching requires one CUDA device");
        state_->layers.push_back(std::make_unique<qwen4_exp::LayerState>(*block));
    }
}
Qwen4BatchRequestState::~Qwen4BatchRequestState() = default;
int64_t Qwen4BatchRequestState::position() const { return state_->cache_position; }
std::size_t Qwen4BatchRequestState::initial_growth_bytes() const {
    std::size_t fixed=0;
    for(std::size_t i=0; i<state_->layers.size(); ++i) {
        const auto& weights = *state_->model.layers[i];
        const auto& state = *state_->layers[i];
        if(weights.gdn)fixed+=weights.gdn->request_cache_growth_bytes(state.gdn);
    }
    return fixed;
}
void Qwen4BatchRequestState::initialize() {
    for(std::size_t i=0; i<state_->layers.size(); ++i) {
        const auto& weights = *state_->model.layers[i];
        auto& state = *state_->layers[i];
        if(weights.gdn)weights.gdn->initialize_request_cache(state.gdn,false);
    }
}
std::size_t Qwen4BatchRequestState::prefill_growth_bytes(int64_t tokens) const {
    return state_->growth_bytes(tokens);
}
std::size_t Qwen4BatchRequestState::resident_bytes() const {
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add = [&](const ::tyloi::DenseTensor& tensor) {
        if (!tensor.defined() || !tensor.is_cuda()) return;
        auto& bytes = allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(tensor.data_ptr())}];
        bytes = std::max(bytes, std::size_t(tensor.numel()) * tensor.element_size());
    };
    add(state_->positions);
    add(state_->position_storage);
    for (std::size_t i = 0; i < state_->layers.size(); ++i) {
        const auto& weights = *state_->model.layers[i];
        const auto& state = *state_->layers[i];
        if (weights.qsa) for (const auto& tensor : weights.qsa->resident_cache_tensors(*state.qsa)) add(tensor.dense());
        if (weights.gdn) { add(state.gdn.conv.dense()); add(state.gdn.recurrent.dense()); }
        if (weights.ple && state.ple.conv.defined()) add(state.ple.conv.dense());
    }
    std::size_t bytes = 0;
    for (const auto& [address,size] : allocations) bytes += size;
    return bytes;
}
struct Qwen4BatchDecodeGraph::Impl {
    using DenseTensor = ::tyloi::DenseTensor;
    qwen4_exp::Model& model;
    DecodeStream stream;
    std::unique_ptr<DecodeWindow> window;
    std::vector<Qwen4BatchRequestState*> requests;
    std::vector<int64_t> capacities;
    std::vector<DenseTensor> positions, cache_positions;
    std::vector<DenseTensor> gdn_addresses;
    std::vector<std::vector<std::shared_ptr<RowStage>>> ple;
    DenseTensor ids, logits;
    bool greedy = false;
    int64_t captures = 0, replays = 0;
    explicit Impl(qwen4_exp::Model& owner)
        : model(owner) {}
    void clear() {
        if (requests.empty() && !window) return;
        window.reset();
        for (auto* request : requests) {
            for (std::size_t i = 0; i < request->state_->layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                auto& state = *request->state_->layers[i];
                if (weights.qsa) weights.qsa->leave_graph(*state.qsa);
            }
        }
        requests.clear(); capacities.clear(); positions.clear(); cache_positions.clear(); ple.clear();gdn_addresses.clear();
        ids = {}; logits = {};
    }
};
Qwen4BatchDecodeGraph::Qwen4BatchDecodeGraph(qwen4_exp::Model& model)
    : graph_(std::make_unique<Impl>(model)) {}
Qwen4BatchDecodeGraph::~Qwen4BatchDecodeGraph() = default;
void Qwen4BatchDecodeGraph::clear() { graph_->clear(); }
int64_t Qwen4BatchDecodeGraph::captures() const { return graph_->captures; }
int64_t Qwen4BatchDecodeGraph::replays() const { return graph_->replays; }
std::optional<Qwen4BatchDecodeGraph::Plan> Qwen4BatchDecodeGraph::plan(
        const std::vector<Qwen4BatchRequestState*>& requests, bool greedy) {
    auto& graph = *graph_;
    auto& model = graph.model;
    std::vector<tyloi::ops::Qsa::Graph::Request> rows;
    for (auto* request : requests) {
        tyloi::ops::Qsa::Graph::Request row;
        row.position = request->position();
        row.maximum = model.max_position_embeddings();
        for (std::size_t i = 0; i < model.layers.size(); ++i)
            if (model.layers[i]->qsa)
                row.layers.emplace_back(model.layers[i]->qsa.get(), request->state_->layers[i]->qsa.get());
        rows.push_back(std::move(row));
    }
    tyloi::ops::Qsa::Graph attention(std::move(rows), true);
    if (!attention.eligible()) return {};
    for (auto* request : requests) {
        for (std::size_t i = 0; i < request->state_->layers.size(); ++i) {
            const auto& weights = *model.layers[i];
            if (weights.ple && !weights.ple->graph_supported()) return {};
        }
    }
    const auto batch = static_cast<int64_t>(requests.size());
    attention.admit();
    auto stream_scope=graph.stream.enter();
    bool reuse = graph.requests == requests && graph.window && graph.greedy == greedy;
    if (reuse) {
        std::vector<::tyloi::Tensor> owners;
        for(auto* request:requests)for(std::size_t i=0;i<model.layers.size();++i) {
            auto& state=*request->state_->layers[i];const auto& weights=*model.layers[i];
            for(auto* tensor:state.graph_state(weights))owners.push_back(*tensor);
            if(weights.qsa)for(auto* tensor:weights.qsa->graph_state(*state.qsa))owners.push_back(*tensor);
        }
        reuse=graph.window->matches(owners);
    }
    if (reuse) for (int64_t row = 0; row < batch; ++row)
        reuse = reuse && requests[row]->position() + 1 <= graph.capacities[row];
    if (!reuse) {
        graph.clear();
    }
    Plan plan;
    plan.rebuild = !graph.window;
    if (plan.rebuild) {
        attention.layout();
        for (std::size_t row = 0; row < requests.size(); ++row) {
            plan.capacities.push_back(attention.rows()[row].capacity);
            plan.growth += attention.rows()[row].growth;
            for (std::size_t i = 0; i < model.layers.size(); ++i)
                if (model.layers[i]->gdn)
                    plan.growth += model.layers[i]->gdn->request_cache_growth_bytes(requests[row]->state_->layers[i]->gdn);
        }
    }
    return plan;
}

tyloi::StepSequence<::tyloi::Tensor> Qwen4BatchDecodeGraph::run(
        const std::vector<Qwen4BatchRequestState*>& requests,
        const ::tyloi::Tensor& ids_handle,const ::tyloi::Tensor& host_ids_handle,bool greedy,const Plan& plan,bool prepared) {
    const auto& ids = ids_handle.dense();
    const auto& host_ids = host_ids_handle.dense();
    auto& graph = *graph_;
    auto& model = graph.model;
    const auto batch = static_cast<int64_t>(requests.size());
    auto stream_scope=graph.stream.enter();
    graph.greedy = greedy;
    if (plan.rebuild) {
        auto fixed = DecodeWindow::fixed_growth(ids.device(), ::tyloi::cuda::options::layer_timings(), false, 0);
        add_growth(fixed, ids.device(), std::size_t(batch) * sizeof(int64_t), 3);
        for (std::size_t i = 0; i < model.layers.size(); ++i) {
            const auto& layer = *model.layers[i];
            if (layer.ple) for (const auto& [device, bytes] : layer.ple->graph_stage_growth(1, ids.device()))
                add_growth(fixed, device, bytes, std::size_t(batch));
            if (layer.gdn && batch > 1 && ::tyloi::cuda::options::batch_attention_projections() &&
                    ::tyloi::cuda::options::gdn_independent_batch()) {
                add_growth(fixed, ids.device(), std::size_t(batch) * sizeof(int64_t), 3);
                add_growth(fixed, Device(kCPU), std::size_t(batch) * sizeof(int64_t), 3);
            }
        }
        stream_scope.reset();
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(fixed)};
        stream_scope = graph.stream.enter();
        graph.requests = requests;
        graph.capacities = plan.capacities;
        graph.window = std::make_unique<DecodeWindow>(::tyloi::cuda::options::layer_timings(),true);
        graph.ids = graph.window->buffer(ids.sizes().vec(),ids.options()).dense();
        graph.ple.resize(batch);
        for (int64_t row = 0; row < batch; ++row) {
            auto& state = *requests[row]->state_;
            graph.positions.push_back(graph.window->buffer({1},ids.options()).dense());
            graph.cache_positions.push_back(graph.window->buffer({1},ids.options()).dense());
            graph.ple[row].resize(state.layers.size());
            for (std::size_t i = 0; i < state.layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                auto& layer = *state.layers[i];
                if (weights.gdn) weights.gdn->initialize_request_cache(layer.gdn);
                if (weights.qsa) weights.qsa->prepare_graph(*layer.qsa,state.positions,graph.capacities[row]);
                if (weights.ple) graph.ple[row][i] = weights.ple->graph_stage(layer.ple,1,ids.device());
            }
        }
        graph.gdn_addresses.resize(model.layers.size());
        if(batch>1 && ::tyloi::cuda::options::batch_attention_projections() && ::tyloi::cuda::options::gdn_independent_batch())
            for(std::size_t i=0;i<model.layers.size();++i) {
                std::vector<std::pair<const tyloi::ops::Gdn*,tyloi::ops::Gdn::State*>> rows;
                for(auto* request:requests) {
                    const auto& weights = *model.layers[i];
                    auto& state = *request->state_->layers[i];
                    if(weights.gdn) rows.emplace_back(weights.gdn.get(),&state.gdn);
                }
                if(rows.size()==requests.size())graph.gdn_addresses[i]=tyloi::ops::Gdn::prepare_independent_batch(rows).dense();
            }
        // Cached MFE owns its graph pool; dense execution needs its own.
        if (!prepared) graph.window->prepare_pool();
        graph.window->set_profile_geometry("batch_decode",static_cast<int>(batch));
    }
    graph.ids.copy_(ids);
    std::vector<std::tuple<qwen4_exp::Ple*,qwen4_exp::Ple::State*,qwen4_exp::NgramEmbedding::Context>> ple_contexts;
    for (int64_t row = 0; row < batch; ++row) {
        auto& state = *requests[row]->state_;
        mfq_qwen4_exp::fill_decode_positions(graph.cache_positions[row],graph.positions[row],
                                            state.cache_position,state.decode_position_delta);
        for (std::size_t i = 0; i < state.layers.size(); ++i) {
            const auto& weights = *model.layers[i];
            auto& layer = *state.layers[i];
            if (weights.ple) {
                if(!graph.window->valid())ple_contexts.emplace_back(weights.ple.get(),&layer.ple,weights.ple->session_context(layer.ple));
                weights.ple->upload_graph(layer.ple,host_ids.narrow(0,row,1),*graph.ple[row][i]);
            }
        }
    }
    if (!graph.window->valid()) {
        struct Saved { ::tyloi::Tensor* tensor; ::tyloi::DenseTensor snapshot; };
        std::vector<Saved> recurrent;
        std::vector<::tyloi::Tensor*> tensors;
        std::vector<std::pair<Device, std::size_t>> copy_growth;
        for (auto* request : requests) for (std::size_t i = 0; i < model.layers.size(); ++i)
            for (auto* tensor : request->state_->layers[i]->graph_state(*model.layers[i])) {
                tensors.push_back(tensor);
                add_growth(copy_growth, batch > 8 ? Device(kCPU) : tensor->device(), tensor->nbytes());
            }
        stream_scope.reset();
        co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, std::move(copy_growth)};
        stream_scope = graph.stream.enter();
        const auto save=[&](::tyloi::Tensor* tensor) {
            // Capture restores these values after warmup. They are never read
            // by replay, so large batches can keep this temporary copy in RAM.
            if(tensor->defined())recurrent.push_back({tensor,batch>8 ? tensor->dense().to(::tyloi::kCPU) : tensor->dense().clone()});
        };
        for (auto* tensor : tensors) save(tensor);
        try {
            auto recording=graph.window->begin_capture();
            while(recording->next()) {
            std::exception_ptr failure;
            bool warm = false;
            try {
            graph.window->mark(-1,"window_begin");
            auto hidden = model.repeat_hidden(model.embed_tokens(graph.ids),model.metadata.hc_mult);
            std::optional<tyloi::ops::Residual::Result> pending_residual;
            const auto chain_setting=::tyloi::cuda::options::gr_chain_ffn_post();
            const bool chain_post=(!chain_setting || *chain_setting) && batch<=8;
            for (std::size_t i = 0; i < model.layers.size(); ++i) {
                graph.window->mark(static_cast<int>(i),"block_begin");
                const auto& block = *model.layers[i];
                graph.window->set_profile_layer_kind(static_cast<int>(i),block.gdn?"gdn":"qsa");
                if (block.ple) {
                    graph.window->mark(static_cast<int>(i),"ple_begin");
                    std::vector<::tyloi::DenseTensor> positional;
                    for (int64_t row = 0; row < batch; ++row) {
                        auto x = hidden.narrow(0,row,1);
                        auto& layer = *requests[row]->state_->layers[i];
                        graph.ple[row][i]->decode();
                        auto rows = graph.ple[row][i]->output().dense().reshape({1,1,model.hidden_size()});
                        positional.push_back(x + block.ple->forward_chunk(
                            layer.ple,x,graph.ids.narrow(0,row,1),true,{},rows));
                    }
                    hidden = join_request_outputs(positional,0);
                    graph.window->mark(static_cast<int>(i),"ple_end");
                }
                auto mix = pending_residual
                    ? block.attention_gr.pre_after(hidden,pending_residual->residual,pending_residual->gate,tyloi::ops::ResidualRole::attention)
                    : block.attention_gr.pre(hidden,tyloi::ops::ResidualRole::attention);
                graph.window->mark(static_cast<int>(i),"attention_core_begin");
                const bool project_batch=batch>1 && ::tyloi::cuda::options::batch_attention_projections();
                const auto projections=project_batch ? (block.gdn ? block.gdn->project_inputs(mix.mixed)
                    : block.qsa->project_inputs(mix.mixed)) : std::vector<::tyloi::Tensor>{};
                std::vector<::tyloi::DenseTensor> attended;
                DenseTensor attention;
                if(project_batch && block.gdn && graph.gdn_addresses[i].defined())
                    attention=block.gdn->forward_independent_batch(requests[0]->state_->layers[i]->gdn,mix.mixed,projections,graph.gdn_addresses[i]).dense();
                if(!attention.defined())for (int64_t row = 0; row < batch; ++row) {
                    auto& layer = *requests[row]->state_->layers[i];
                    auto attention_input = mix.mixed.dense().narrow(0,row,1);
                    std::vector<::tyloi::Tensor> projected;
                    for(const auto& value:projections)projected.push_back(value.dense().narrow(0,row,1));
                    attended.push_back((block.gdn ? block.gdn->forward_chunk(layer.gdn,attention_input,true,std::move(projected),!project_batch)
                        : block.qsa->forward_graph(*layer.qsa,attention_input,graph.positions[row],graph.cache_positions[row],
                            std::move(projected),!project_batch)).dense());
                }
                if(!attention.defined())attention=join_request_outputs(attended,0);
                if(project_batch)attention=(block.gdn ? block.gdn->project_output(attention)
                    : block.qsa->project_output(attention)).dense();
                graph.window->mark(static_cast<int>(i),"attention_core_end");
                auto ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate,tyloi::ops::ResidualRole::ffn);
                graph.window->mark(static_cast<int>(i),"ffn_pre_end");
                auto ffn_output=tyloi::ops::batch_ffn(block.ffn,ffn_mix.mixed).dense();
                bool defer=false;
                if(chain_post && i+1<model.layers.size()) {
                    const auto& next=*model.layers[i+1];
                    defer=!next.ple && next.attention_gr.can_chain(ffn_mix.residual);
                }
                if(defer){hidden=std::move(ffn_output);pending_residual=std::move(ffn_mix);}
                else {hidden=block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate).dense();pending_residual.reset();}
                graph.window->mark(static_cast<int>(i),"block_end");
            }
            auto finalized=qwen4_finalize_hidden(model, std::move(hidden),batch,1);
            graph.logits = greedy ? qwen4_next_token(model,qwen4_last_hidden(std::move(finalized)))
                .reshape({batch,1}) : qwen4_logits_from_hidden(model, std::move(finalized)).reshape({batch,-1});
            graph.window->mark(-1,"window_end");
            warm=recording->finish();
            } catch (...) { failure = std::current_exception(); }
            if (failure) {
                if (!recording->retry(failure)) std::rethrow_exception(failure);
                warm = true;
            }
            for (const auto& saved : recurrent) saved.tensor->dense().copy_(saved.snapshot);
            if(warm) {
                graph.logits={};
                recording->release_warm_outputs();
                recording->pause();
                stream_scope.reset();
                co_yield tyloi::GraphRequest{tyloi::GraphPhase::admit, nullptr, recording->growth()};
                stream_scope=graph.stream.enter();
                recording->resume();
            }
            }
            recording->complete();
            std::vector<::tyloi::Tensor> owners;
            for(auto* request:requests)for(std::size_t i=0;i<model.layers.size();++i) {
                auto& state=*request->state_->layers[i];const auto& weights=*model.layers[i];
                for(auto* tensor:state.graph_state(weights))owners.push_back(*tensor);
                if(weights.qsa)for(auto* tensor:weights.qsa->graph_state(*state.qsa))owners.push_back(*tensor);
            }
            graph.window->retain_result(graph.logits);
            graph.window->retain(owners);
        }
        catch(const std::exception& error) {
            for (const auto& saved : recurrent) saved.tensor->dense().copy_(saved.snapshot);
            const std::string message=error.what();
            if(!dynamic_cast<const MfqResourceExhausted*>(&error) &&
               !message.starts_with("CUDA process VRAM limit exceeded:") &&
               !message.starts_with("CUDA VRAM limit exceeded:") &&
               !message.starts_with("CUDA allocation failed:") &&
               !message.starts_with("CUDA pinned allocation failed:"))throw;
            // Capture has not accepted a token. Retire its consumers before
            // restoring state, then let admission reject only the newest row.
            graph.clear();
            (void)cudaGetLastError();
            for(const auto& saved:recurrent)saved.tensor->dense().copy_(saved.snapshot);
            for(auto& [layer,layer_state,context]:ple_contexts)layer->restore_graph_context(*layer_state,std::move(context));
            throw;
        }
        ++graph.captures;
    }
    graph.window->run();
    ++graph.replays;
    for (std::size_t row=0;row<requests.size();++row) {
        auto& state = *requests[row]->state_;
        for (std::size_t i = 0; i < state.layers.size(); ++i)
            if (model.layers[i]->qsa) model.layers[i]->qsa->advance_graph(*state.layers[i]->qsa,1);
        state.positions=append_decode_positions(state.positions,graph.cache_positions[row],
            state.position_storage,model.max_position_embeddings());
        ++state.cache_position;
    }
    stream_scope.reset();
    co_yield ::tyloi::Tensor(graph.logits);
}

tyloi::StepSequence<std::vector<::tyloi::Tensor>> qwen4_batch_prefill(qwen4_exp::Model& model,
        const std::vector<Qwen4BatchRequestState*>& requests,
        const std::vector<::tyloi::Tensor>& chunk_handles, bool cache_pipeline) {
    std::vector<::tyloi::DenseTensor> chunks;
    chunks.reserve(chunk_handles.size());
    for (const auto& chunk : chunk_handles) chunks.push_back(chunk.dense());
    TYLOINFER_RUNTIME_CHECK(!requests.empty() && requests.size()==chunks.size(),"invalid packed prefill rows");
    const bool packed=requests.size()>1;
    std::vector<int64_t> offsets{0};
    std::vector<::tyloi::DenseTensor> current,full;
    for (std::size_t row=0;row<requests.size();++row) {
        const auto& state=*requests[row]->state_;
        const auto& ids=chunks[row];
        TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 &&
            ids.size(1)<=model.max_position_embeddings()-state.cache_position,"invalid packed prefill chunk");
        auto positions=::tyloi::cuda::arange(ids.size(1),ids.options())+state.cache_position;
        current.push_back(positions);
        full.push_back(state.positions.defined()?::tyloi::cuda::cat({state.positions,positions},-1):positions);
        offsets.push_back(offsets.back()+ids.size(1));
    }
    auto ids=::tyloi::cuda::cat(chunks,1);
    auto host_ids=ids.to(::tyloi::kCPU);
    const auto tokens=offsets.back();
    if(!packed && cache_pipeline) {
        for(std::size_t row=0;row<requests.size();++row) {
            const auto row_ids=host_ids.narrow(1,offsets[row],chunks[row].size(1));
            for(std::size_t i=0;i<requests[row]->state_->layers.size();++i) {
                const auto& weights=*model.layers[i];
                auto& state=*requests[row]->state_->layers[i];
                if(weights.ple)weights.ple->prefetch_tokens(state.ple,row_ids);
            }
        }
    }
    auto hidden=model.repeat_hidden(model.embed_tokens(ids),model.metadata.hc_mult);
    for (std::size_t i=0;i<model.layers.size();++i) {
        co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::begin,
            packed ? tyloi::LayerEvent::Mode::packed : tyloi::LayerEvent::Mode::single};
        const auto& block=*model.layers[i];
        if (block.ple) {
            std::vector<::tyloi::DenseTensor> positional;
            for (std::size_t row=0;row<requests.size();++row) {
                auto x=hidden.narrow(1,offsets[row],chunks[row].size(1));
                auto& layer=*requests[row]->state_->layers[i];
                std::optional<tyloi::ops::ProfileRange> range;
                if (!packed) range.emplace("prefill.ple");
                auto ple = block.ple->forward(layer.ple,x,chunks[row],true,0,
                    host_ids.narrow(1,offsets[row],chunks[row].size(1)));
                if (range) range->finish();
                positional.push_back(x+ple);
            }
            hidden=join_request_outputs(positional,1);
        }
        tyloi::ops::Residual::Result mix;
        {
            std::optional<tyloi::ops::ProfileRange> range;
            if (!packed) range.emplace("prefill.gr_attention");
            mix = block.attention_gr.pre(hidden);
            if (range) range->finish();
        }
        ::tyloi::DenseTensor attention;
        if(packed) {
            const auto projections=block.gdn?block.gdn->project_inputs(mix.mixed)
                                             :block.qsa->project_inputs(mix.mixed);
            std::vector<::tyloi::DenseTensor> attended;
            for (std::size_t row=0;row<requests.size();++row) {
                auto& layer=*requests[row]->state_->layers[i];
                auto attention_input=mix.mixed.dense().narrow(1,offsets[row],chunks[row].size(1));
                std::vector<::tyloi::Tensor> projected;
                for (const auto& value:projections)
                    projected.push_back(value.dense().narrow(1,offsets[row],chunks[row].size(1)));
                attended.push_back((block.gdn?block.gdn->forward_chunk(layer.gdn,attention_input,true,std::move(projected),false)
                    :block.qsa->forward(*layer.qsa,attention_input,current[row],full[row],true,nullptr,std::move(projected),false)).dense());
            }
            attention=join_request_outputs(attended,1);
            attention=(block.gdn?block.gdn->project_output(attention):block.qsa->project_output(attention)).dense();
        } else {
            auto& layer=*requests.front()->state_->layers[i];
            tyloi::ops::ProfileRange range(block.gdn ? "prefill.gdn" : "prefill.qsa");
            attention = block.gdn ? block.gdn->forward(layer.gdn,mix.mixed,true,0).dense()
                : block.qsa->forward(*layer.qsa,mix.mixed,current.front(),full.front(),true).dense();
            range.finish();
        }
        tyloi::ops::Residual::Result ffn_mix;
        {
            std::optional<tyloi::ops::ProfileRange> range;
            if (!packed) range.emplace("prefill.gr_ffn_pre");
            ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate);
            if (range) range->finish();
        }
        ::tyloi::DenseTensor ffn_output;
        {
            std::optional<tyloi::ops::ProfileRange> range;
            if (!packed) range.emplace("prefill.ffn");
            ffn_output = tyloi::ops::ffn(block.ffn,ffn_mix.mixed).dense();
            if (range) range->finish();
        }
        {
            std::optional<tyloi::ops::ProfileRange> range;
            if (!packed) range.emplace("prefill.gr_ffn_post");
            hidden = block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate).dense();
            if (range) range->finish();
        }
        co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::end};
    }
    hidden=qwen4_finalize_hidden(model, std::move(hidden),1,tokens);
    std::vector<::tyloi::Tensor> result;
    for (std::size_t row=0;row<requests.size();++row) {
        result.push_back(hidden.narrow(1,offsets[row+1]-1,1));
        requests[row]->state_->positions=std::move(full[row]);
        requests[row]->state_->cache_position+=chunks[row].size(1);
    }
    co_yield std::move(result);
}

::tyloi::Tensor qwen4_batch_decode(qwen4_exp::Model& model,
        const std::vector<Qwen4BatchRequestState*>& requests, ::tyloi::Tensor ids_handle,
        const ::tyloi::Tensor& host_ids_handle) {
    const auto& ids = ids_handle.dense();
    const auto& host_ids = host_ids_handle.dense();
    const auto batch = static_cast<int64_t>(requests.size());
    TYLOINFER_RUNTIME_CHECK(batch > 0 && ids.dim() == 2 && ids.size(0) == batch && ids.size(1) == 1,
                      "invalid Flash-Next batch input");
    std::vector<::tyloi::DenseTensor> current, full;
    for (const auto* request : requests) {
        const auto& state = *request->state_;
        TYLOINFER_RUNTIME_CHECK(state.cache_position < model.max_position_embeddings(), "batch request exceeds max_context");
        auto position = ::tyloi::cuda::full({1},state.cache_position,ids.options());
        current.push_back(position);
        full.push_back(state.positions.defined() ? ::tyloi::cuda::cat({state.positions,position},-1) : position);
    }
    auto hidden = model.repeat_hidden(model.embed_tokens(ids),model.metadata.hc_mult);
    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const auto& block = *model.layers[i];
        if (block.ple) {
            std::vector<::tyloi::DenseTensor> positional;
            for (int64_t row = 0; row < batch; ++row) {
                auto& layer = *requests[row]->state_->layers[i];
                auto x = hidden.narrow(0,row,1);
                positional.push_back(x + block.ple->forward(layer.ple,x,ids.narrow(0,row,1),true,0,
                                                           host_ids.narrow(0,row,1)));
            }
            hidden = join_request_outputs(positional,0);
        }
        auto mix = block.attention_gr.pre(hidden);
        const bool project_batch=batch>1 && block.gdn && ::tyloi::cuda::options::batch_attention_projections();
        const auto projections=project_batch ? block.gdn->project_inputs(mix.mixed) : std::vector<::tyloi::Tensor>{};
        std::vector<::tyloi::DenseTensor> attended;
        for (int64_t row = 0; row < batch; ++row) {
            auto& layer = *requests[row]->state_->layers[i];
            auto attention_input = mix.mixed.dense().narrow(0,row,1);
            std::vector<::tyloi::Tensor> projected;
            for(const auto& value:projections)projected.push_back(value.dense().narrow(0,row,1));
            attended.push_back((block.gdn ? block.gdn->forward_chunk(layer.gdn,attention_input,true,std::move(projected),!project_batch)
                : block.qsa->forward(*layer.qsa,attention_input,current[row],full[row],true)).dense());
        }
        auto attention=join_request_outputs(attended,0);
        if(project_batch)attention=block.gdn->project_output(attention).dense();
        auto ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate);
        // Tiles share the expert transfer cache across all eligible rows.
        hidden = block.ffn_gr.post(tyloi::ops::batch_ffn(block.ffn,ffn_mix.mixed),ffn_mix.residual,ffn_mix.gate).dense();
    }
    auto logits = qwen4_logits_from_hidden(model, qwen4_finalize_hidden(model, std::move(hidden),batch,1)).reshape({batch,-1});
    for (int64_t row = 0; row < batch; ++row) {
        requests[row]->state_->positions = std::move(full[row]);
        ++requests[row]->state_->cache_position;
    }
    return logits;
}

qwen4_exp::State::State(const Model& model, bool initialize_layers) : model_(model) {
    if (initialize_layers)
        for (const auto& layer : model_.layers)
            layers.push_back(std::make_unique<LayerState>(*layer));
}

qwen4_exp::State::~State() = default;

tyloi::engine::Metrics qwen4_exp::State::metrics() const {
    auto result=qwen4_cache_memory_metrics(model_,*this,model_.predictor());
    auto profile=qwen4_decode_graph_profile_metrics(model_,*this);
    result.insert(result.end(),profile.begin(),profile.end());
    result.emplace_back("decode_graph_count",graph_registry ? double(graph_registry->cells.size()) : 0.0);
    return result;
}

std::unique_ptr<tyloi::models::Batch> qwen4_exp::Model::make_batch(
        const tyloi::models::BatchConfig& config) const {
    return qwen4_exp::make_batch(const_cast<Model&>(*this),config);
}

void qwen4_exp::State::release_graphs() {
    qwen4_release_decode_graphs(model_, *this);
}

void qwen4_exp::State::publish_restore(State& candidate) noexcept {
    layers.swap(candidate.layers);
    graph_registry.swap(candidate.graph_registry);
    std::swap(positions, candidate.positions);
    cache_pos = candidate.cache_pos;
    decode_position_delta = candidate.decode_position_delta;
    batch = candidate.batch;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    speculative_suffix_journal = false;
    speculative_prefix_retention = false;
}

void qwen4_exp::State::reset(int64_t new_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    speculative_suffix_journal = false;
    speculative_prefix_retention = false;
    const bool retain=reuse_decode_graphs && new_batch==1 && graph_registry && graph_registry->reusable();
    for(auto& layer:layers)layer->retain_graph_storage=retain;
    if(!retain)graph_registry.reset();
    else for(auto& [key,cell]:graph_registry->cells)cell->used_kinds=0;
    positions = {};
    batch = new_batch;
    for(std::size_t i=0;i<layers.size();++i)layers[i]->reset(*model_.layers[i]);
}

void qwen4_exp::State::begin_forward() {
    for(auto& layer:layers)layer->retained_prefix=false;
    if(graph_registry) {
        if(reuse_decode_graphs && graph_registry->reusable()) {
            for(auto [attention,state]:graph_registry->qsa)attention->leave_graph(*state);
        } else {graph_registry->restore_legacy();graph_registry.reset();}
    }
}

void qwen4_exp::State::finish_forward(const ::tyloi::Tensor &full_positions,
                                      int64_t new_batch, int64_t tokens) {
    positions = full_positions;
    batch = new_batch;
    speculative_prefix_retention=tokens<=8 && !layers.empty() &&
        std::all_of(layers.begin(),layers.end(),[](const auto& layer) {
            return layer->retained_prefix;
        });
}

// Keep each compressed expert image resident while advancing all prompt chunks
// through that layer. Attention and PLE still advance in causal token order.
tyloi::models::PrefillPlan qwen4_exp::Model::prefill_plan(
        const tyloi::models::State& value, const tyloi::models::PrefillOptions& policy) const {
    const auto& state = dynamic_cast<const qwen4_exp::State&>(value);
    const auto enabled=::tyloi::cuda::options::moe_prefill_layer_major();
    const auto whole=::tyloi::cuda::options::moe_prefill_layer();
    if ((enabled && *enabled==0) || (whole && *whole==0) || (!policy.routed && !whole) ||
        policy.tokens<=128 || policy.tokens>16384 || state.speculative_start>=0 ||
        (!policy.routed && !host_tokens) || !policy.coordinated || policy.transformed ||
        policy.compute<=8 || policy.first!=0 || (policy.last!=0 && policy.last!=policy.tokens)) return {false,8192};
    for (const auto& block:layers)
        if (block->cpu_offloaded || block->cuda_device!=primary_device) return {false,8192};
    return {true,8192};
}

tyloi::StepSequence<tyloi::models::Prefill> qwen4_exp::Model::prefill(
        tyloi::models::Input input, tyloi::models::State& value,
        const tyloi::models::ForwardOptions&, const tyloi::models::PrefillOptions& policy) const {
    if (!prefill_plan(value, policy).layered) co_return;
    auto& state=dynamic_cast<qwen4_exp::State&>(value);
    const auto& model=*this;
    auto ids_handle=input.ids;
    auto compute_chunk=policy.compute;
    const bool raw_hidden=input.raw_hidden!=nullptr;
    const bool raw_hidden_host=policy.raw_host;
    using DenseTensor=::tyloi::DenseTensor;
    auto ids = ids_handle.dense();
    TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 &&
        prefill_plan(state,policy).layered,
        "layer-major prefill requires a bounded single-device Qwen4 chunk and whole-layer cache");
    TYLOINFER_RUNTIME_CHECK(compute_chunk>8,"layer-major compute chunk must exceed decode width");
    ids=model.device_ids(std::move(ids));
    const auto tokens=ids.size(1);
    const auto cache_start=state.cache_pos;
    auto capture = tyloi::ops::ProfileCapture::prefill_once(tokens);
    compute_chunk=std::min(compute_chunk,tokens);
    const auto reserve=std::size_t(1)<<30;
    const auto row_bytes=std::size_t(model.config.hidden)*model.config.streams*sizeof(float)*8;
    const auto sequence_reserve=std::size_t(tokens)*model.config.hidden*model.config.streams*sizeof(float)*3;
    capture.suspend();
    co_yield tyloi::Workspace{reserve+sequence_reserve, nullptr,
        tyloi::Workspace::Tile{tokens,row_bytes,&compute_chunk}};
    capture.resume();
    tyloi::cuda::ResidentAllocationScope persistent_preamble;
    qwen4_validate_forward(model,tokens,cache_start,false);
    if(cache_start==0)state.reset(1);
    state.begin_forward();
    auto current=::tyloi::cuda::arange(cache_start,cache_start+tokens,ids.options());
    if(state.decode_position_delta)current=current+state.decode_position_delta;
    auto prepared=std::move(current);
    if ((prepared.dim() == 2 || prepared.dim() == 3) && prepared.size(0) == 4)
        prepared = prepared.narrow(0,1,3);
    auto prepared_full=state.positions.defined()?
        ::tyloi::cuda::cat({state.positions.dense(),prepared},-1):prepared;
    tyloi::models::require_model((prepared.dim() == 1 || (prepared.dim() == 2 && prepared.size(0) == 3) ||
        (prepared.dim() == 3 && prepared.size(0) == 3 && prepared.size(1) == 1)) && prepared.size(-1) == tokens,
        "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    auto cpu_ids=ids.to(::tyloi::kCPU,::tyloi::kInt64).contiguous();
    persistent_preamble.finish();
    DenseTensor hidden=::tyloi::cuda::empty({1,tokens,model.metadata.hidden_size*model.metadata.hc_mult},
                           ids.options().dtype(::tyloi::kFloat16));
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
        const auto count=std::min(compute_chunk,tokens-begin);
        auto input=model.repeat_hidden(model.embed_tokens(ids.narrow(1,begin,count)),model.metadata.hc_mult);
        TYLOINFER_RUNTIME_CHECK(input.scalar_type()==hidden.scalar_type() &&
            input.size(-1)==hidden.size(-1),"layer-major embedding geometry changed");
        hidden.narrow(1,begin,count).copy_(input);
    }
    const auto trace_option=::tyloi::cuda::options::prefill_layer_major_trace();
    const bool trace=trace_option && *trace_option!=0;
    const auto batch_option=::tyloi::cuda::options::moe_prefill_expert_batch();
    const bool batch_experts=!batch_option || *batch_option!=0;
    const auto bulk_gr_option=::tyloi::cuda::options::gr_prefill_layer_batch();
    // Long prompts retain the established memory footprint unless explicitly
    // requested; the combined residual tensors have been verified to 16k.
    const bool bulk_gr=batch_experts && (bulk_gr_option?*bulk_gr_option:tokens<=16384);
    const auto bulk_gdn_option=::tyloi::cuda::options::gdn_prefill_layer_batch();
#ifdef TYLOINFER_NATIVE_CUDA
    // Allow 8k chat framing while keeping the extra whole-sequence QKV and
    // gate tensors out of the 16k service path, which also retains MTP weights.
    const bool bulk_gdn=bulk_gr && (bulk_gdn_option?*bulk_gdn_option:tokens<=9216);
#else
    const bool bulk_gdn=bulk_gr && bulk_gdn_option && *bulk_gdn_option;
#endif
    const auto reference_rows=batch_experts?std::min<int64_t>(1024,compute_chunk):0;
    const auto batch_tokens_option=::tyloi::cuda::options::moe_prefill_expert_batch_tokens();
    const int64_t batch_tokens=batch_tokens_option?*batch_tokens_option:8192;
    const auto append_tail_option=::tyloi::cuda::options::moe_prefill_append_tail();
    const bool append_tail=append_tail_option && *append_tail_option;
    const auto expert_batch_count=[&](int64_t remaining) {
        // Chat framing can leave a few tokens beyond the batch boundary.
        // Absorb at most one reference chunk instead of running a second
        // almost-empty routed GEMM for every layer.
        return append_tail && remaining<=batch_tokens+1024?remaining:std::min(batch_tokens,remaining);
    };
    TYLOINFER_RUNTIME_CHECK(!batch_experts || batch_tokens>=1024,"expert prefill batch must cover at least 1024 tokens");
    const auto audit_option=::tyloi::cuda::options::prefill_expert_batch_audit();
    const bool audit_batch=audit_option && *audit_option!=0;
    const auto started=std::chrono::steady_clock::now();
    int layer=0;
    for(auto& block:model.layers) {
        auto& layer_state=*state.layers[layer];
        capture.suspend(); co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::begin, tyloi::LayerEvent::Mode::tiled}; capture.resume();
        DenseTensor next_hidden=hidden;
        DenseTensor ffn_inputs,ffn_gate;
        tyloi::ops::Residual::Result attention_result;
        DenseTensor attention_output;
        if(bulk_gr) {
            auto& qwen_block=*block;
            DenseTensor source=hidden;
            if(qwen_block.ple) {
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
                    const auto count=std::min(compute_chunk,tokens-begin);
                    auto chunk_ids=ids.narrow(1,begin,count);
                    auto host_ids=cpu_ids.narrow(1,begin,count);
                    qwen_block.ple->prefetch_tokens(layer_state.ple, host_ids);
                    auto input=qwen4_exp::prefill_ple(qwen_block,layer_state,
                        hidden.narrow(1,begin,count),chunk_ids,host_ids);
                    if(begin==0)source=::tyloi::cuda::empty(hidden.sizes(),input.options());
                    source.narrow(1,begin,count).copy_(input);
                }
            }
            {
                tyloi::ops::ProfileRange range("prefill.gr_attention");
                attention_result = qwen_block.attention_gr.pre(source);
                range.finish();
            }
            if(bulk_gdn && qwen_block.gdn) {
                capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
                const tyloi::ops::Gdn::ProjectionScope projection_rows(compute_chunk);
                tyloi::ops::ProfileRange range("prefill.gdn");
                attention_output = qwen_block.gdn->forward(layer_state.gdn,attention_result.mixed,true,0).dense();
                range.finish();
            }
        }
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
            if(bulk_gdn && block->gdn)continue;
            const auto count=std::min(compute_chunk,tokens-begin);
            capture.resume();
            auto chunk_ids=ids.narrow(1,begin,count);
            auto host_ids=cpu_ids.narrow(1,begin,count);
            auto positions=prepared.narrow(-1,begin,count);
            auto full_positions=prepared_full.narrow(-1,0,cache_start+begin+count);
            if (block->ple)
                block->ple->prefetch_tokens(layer_state.ple, host_ids);
            auto input=hidden.narrow(1,begin,count);
            DenseTensor output;
            if(bulk_gr) {
                auto& qwen_block=*block;
                auto attention_chunk=qwen4_exp::prefill_attention_core(qwen_block,layer_state,
                    attention_result.mixed.dense().narrow(1,begin,count),positions,full_positions);
                if(!attention_output.defined()) {
                    auto shape=attention_chunk.sizes().vec();shape[1]=tokens;
                    attention_output=::tyloi::cuda::empty(shape,attention_chunk.options());
                }
                attention_output.narrow(1,begin,count).copy_(attention_chunk);
                continue;
            } else if(batch_experts) {
                auto& qwen_block=*block;
                auto mix=qwen4_exp::prefill_attention(qwen_block,layer_state,
                    input,chunk_ids,host_ids,positions,full_positions);
                if(!ffn_inputs.defined()) {
                    ffn_inputs=::tyloi::cuda::empty({1,tokens,model.hidden_size()},mix.mixed.dense().options().dtype(::tyloi::kFloat16));
                    ffn_gate=::tyloi::cuda::empty({1,tokens,mix.gate.size(-1)},mix.gate.dense().options());
                }
                // The MoE casts the router/FFN source to half at this boundary.
                ffn_inputs.narrow(1,begin,count).copy_(mix.mixed.dense().to(::tyloi::kFloat16));
                ffn_gate.narrow(1,begin,count).copy_(mix.gate.dense());
                output=mix.residual.dense();
            } else output=qwen4_exp::forward(*block,layer_state,
                input,chunk_ids,positions,full_positions,0,host_ids);
            // Retain the original rounding at every layer boundary. The next
            // chunk is independent of this input once its cache writes retire.
            TYLOINFER_RUNTIME_CHECK(output.sizes()==input.sizes(),"layer-major decoder changed hidden shape");
            if(begin==0 && output.scalar_type()!=hidden.scalar_type()) {
                next_hidden=::tyloi::cuda::empty(hidden.sizes(),output.options());
                if(trace)std::cout<<"prefill_layer_major_hidden layer="<<layer
                    <<" input_bytes="<<hidden.nbytes()<<" output_bytes="<<next_hidden.nbytes()<<std::endl;
            }
            TYLOINFER_RUNTIME_CHECK(output.scalar_type()==next_hidden.scalar_type(),
                "layer-major decoder output dtype changed within a layer");
            next_hidden.narrow(1,begin,count).copy_(output);
        }
        // Attention's decoded matrices are no longer needed by FFN. Keep
        // reuse within each phase, without retaining both sets at peak memory.
        if(tokens>9216) { capture.suspend(); co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::clear}; capture.resume(); }
        if(bulk_gr) {
            auto& qwen_block=*block;
            tyloi::ops::Residual::Result mix;
            {
                tyloi::ops::ProfileRange range("prefill.gr_ffn_pre");
                mix = qwen_block.ffn_gr.pre_after(attention_output,attention_result.residual,attention_result.gate);
                range.finish();
            }
            ffn_inputs=mix.mixed.dense().to(::tyloi::kFloat16);ffn_gate=mix.gate.dense();
            next_hidden=mix.residual.dense();
            attention_result={};attention_output={};
        }
        if(tokens>9216) { capture.suspend(); co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::clear}; capture.resume(); }
        // The new residual owns the next layer's values. Prior hidden readers
        // have already been submitted before this stream-ordered release.
        if(tokens>9216)hidden={};
        if(batch_experts) {
            capture.suspend(); co_yield tyloi::StepState::advanced; capture.resume();
            auto& qwen_block=*block;
            capture.resume();
            DenseTensor ffn_output;
            {
                tyloi::ops::ProfileRange range("prefill.ffn");
                if(expert_batch_count(tokens)==tokens) {
                    ffn_output=tyloi::ops::ffn(qwen_block.ffn,ffn_inputs,{}, {}, reference_rows).dense();
                } else {
                    for(int64_t begin=0;begin<tokens;) {
                        auto count=expert_batch_count(tokens-begin);
                        const auto tail=tokens-begin-count;
                        // Keep tiny tails in the prefill path to retain shared prefill images.
                        if(tail>0 && tail<=8)count-=16-tail;
                        auto part=tyloi::ops::ffn(qwen_block.ffn,ffn_inputs.narrow(1,begin,count),{}, {}, reference_rows).dense();
                        if(!ffn_output.defined()) {
                            auto shape=part.sizes().vec();shape[1]=tokens;
                            ffn_output=::tyloi::cuda::empty(shape,part.options());
                        }
                        ffn_output.narrow(1,begin,count).copy_(part);
                        begin+=count;
                    }
                }
                range.finish();
            }
            TYLOINFER_RUNTIME_CHECK(ffn_output.defined(),"expert prefill produced no output");
            if(audit_batch) {
                auto expected=::tyloi::cuda::empty(ffn_output.sizes(),ffn_output.options());
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    expected.narrow(1,begin,count).copy_(
                        tyloi::ops::ffn(qwen_block.ffn,ffn_inputs.narrow(1,begin,count),{}, {}, reference_rows).dense());
                }
                auto a=ffn_output.to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
                auto b=expected.to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
                double error=0,scale=0,maximum=0;
                for(int64_t i=0;i<a.numel();++i) {
                    const double delta=double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i];
                    error+=delta*delta;scale+=double(b.data_ptr<float>()[i])*b.data_ptr<float>()[i];
                    maximum=std::max(maximum,std::abs(delta));
                }
                std::cout<<"prefill_expert_batch_layer_oracle layer="<<layer<<" tokens="<<tokens
                    <<" relative_rms="<<std::sqrt(error/std::max(1e-30,scale))
                    <<" maximum_absolute="<<maximum<<std::endl;
                // Diagnostic only: continue with the reference so later
                // layers are tested with their original input distribution.
                ffn_output=std::move(expected);
            }
            DenseTensor post_hidden=next_hidden;
            for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                const auto count=std::min(compute_chunk,tokens-begin);
                auto output=qwen_block.ffn_gr.post(ffn_output.narrow(1,begin,count),
                    next_hidden.narrow(1,begin,count),ffn_gate.narrow(1,begin,count));
                if(begin==0 && output.scalar_type()!=next_hidden.scalar_type())
                    post_hidden=::tyloi::cuda::empty(next_hidden.sizes(),output.options());
                TYLOINFER_RUNTIME_CHECK(output.scalar_type()==post_hidden.scalar_type(),
                    "expert-batched residual dtype changed");
                post_hidden.narrow(1,begin,count).copy_(output.dense());
            }
            next_hidden=std::move(post_hidden);
        }
        hidden=std::move(next_hidden);
        capture.suspend(); co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::end}; capture.resume();
        ++layer;
        if(trace && (layer%4==0 || layer==int(model.layers.size()))) {
            tyloi::models::Prefill report;
            report.phase=tyloi::models::Prefill::Phase::progress;
            report.progress=tyloi::models::Prefill::Progress{
                ::tyloi::Device{::tyloi::kCUDA,model.primary_device},tokens,compute_chunk,
                layer,static_cast<std::int64_t>(model.layers.size()),started};
            capture.suspend(); co_yield std::move(report); capture.resume();
        }
    }
    capture.resume();
    const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
    auto finalized=qwen4_finalize_hidden(model, hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
    finalized=qwen4_last_hidden(std::move(finalized)).unsqueeze(1);
    if(raw_hidden && raw_hidden_host)hidden=hidden.to(::tyloi::kCPU);
    tyloi::models::Prefill result;
    result.mode=tyloi::models::Prefill::Mode::layered;
    result.phase=tyloi::models::Prefill::Phase::computed;
    capture.suspend(); co_yield result; capture.resume();
    result.hidden=std::move(finalized);
    if(raw_hidden)result.raw=std::move(hidden);
    else hidden={};
    capture.suspend();
    co_yield tyloi::Workspace{0,&result.hidden,std::nullopt,raw_hidden && !raw_hidden_host ? &result.raw : nullptr};
    capture.resume();
    state.finish_forward(prepared_full,1,tokens);
    state.cache_pos=cache_start+tokens;
    result.phase=tyloi::models::Prefill::Phase::complete;
    capture.suspend(); co_yield std::move(result); capture.resume();
}

namespace qwen4_exp {

::tyloi::Tensor Model::embedding(const ::tyloi::Tensor& ids, int) const {
    return std::make_shared<::tyloi::DenseTensor>(embed_tokens(device_ids(ids.dense())));
}

::tyloi::Tensor Model::normalize(const ::tyloi::Tensor& hidden, std::int64_t, std::int64_t) const {
    return hidden;
}

::tyloi::Tensor Model::softcap(const ::tyloi::Tensor& logits) const {
    return logits;
}

::tyloi::Tensor Model::logits(const ::tyloi::Tensor& hidden, bool) const {
    tyloi::ops::ProfileRange range("model.lm_head", hidden.device());
    auto output = tyloi::ops::project(hidden, lm_head);
    range.finish();
    return output;
}

::tyloi::Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                                    const tyloi::models::ForwardOptions&) const {
    auto& state = static_cast<State&>(state_value);
    if (input.suffix) {
        tyloi::models::require_model(state.speculative_start >= 0 && state.speculative_confirmed == 0 &&
                                         !state.speculative_suffix_forward,
                                     "speculative suffix is not active");
    }
    struct SuffixScope {
        State& state;
        bool active;
        ~SuffixScope() { if (active) state.speculative_suffix_forward = false; }
    } scope{state, input.suffix};
    if (input.suffix) state.speculative_suffix_forward = true;
    return std::make_shared<::tyloi::DenseTensor>(qwen4_forward(*this, state, std::move(input)));
}

void State::rollback_speculative(std::int64_t accepted) {
    tyloi::models::require_model(speculative_start >= 0 && accepted >= 0,
                                 "no speculative transaction");
    tyloi::models::require_model(!speculative_suffix_journal || accepted == 0,
                                 "suffix checkpoint cannot retain an accepted prefix");
    const auto keep = speculative_start + speculative_confirmed + accepted;
    speculative_prefix_retention=false;
    for (std::size_t i=0;i<model_.layers.size();++i)
        layers[i]->rollback(*model_.layers[i],keep);
    if (positions.defined()) positions=positions.dense().narrow(-1,0,keep);
    cache_pos = keep;
    speculative_suffix_journal=false;
    speculative_start = -1;
    speculative_confirmed = 0;
}

void Model::load_config(std::string_view payload) {
    auto& config = this->config;
    auto& metadata = this->metadata;
    config = CausalConfig::from_json(payload);
    metadata.vocab_size = config.vocab;
    metadata.hidden_size = config.hidden;
    metadata.num_hidden_layers = config.layers;
    metadata.num_attention_heads = config.heads;
    metadata.num_key_value_heads = config.kv_heads;
    metadata.head_dim = config.width;
    metadata.max_position_embeddings = config.maximum;
    metadata.rotary_dim = config.rotary;
    metadata.num_experts = config.experts;
    metadata.hc_mult = config.streams;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.eps;
    metadata.tie_word_embeddings = config.tied_embeddings;
    metadata.multi_axis_positions = true;
    metadata.flash_next = true;
    metadata.model_type = "qwen4_exp";
    metadata.layer_types = config.layer_types;
}


} // namespace qwen4_exp
} // namespace tyloi::cuda
