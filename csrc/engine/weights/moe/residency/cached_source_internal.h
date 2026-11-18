#pragma once

#include "engine/weights/moe/prepare/load.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/moe/prepare/projection.h"

#include "engine/backends/cuda/ops/ffn/experts/launch.h"
#include "engine/weights/moe/residency/expert_cache_internal.h"
#include "engine/weights/moe/host/dispatch_plan.h"

struct MoeCachedCohort {
    int index = -1;
    const MixedMoePool * cpu = nullptr;
    MoeGpuArena * arena = nullptr;
    std::vector<::tyloi::DenseTensor> cpu_fields;
    std::vector<const uint8_t *> mapped_fields;
    std::vector<int64_t> bytes_per_expert;
    std::vector<int32_t> expert_to_local;
    std::vector<int32_t> host_map;
    // Field-major cold experts inside one immutable, contiguous layer image.
    std::vector<int32_t> prefill_ram_experts;
    std::vector<int64_t> prefill_ram_fields;
    std::shared_ptr<tyloi::cuda::MfeMxfp4ExpertStore> range_store;
    bool map_dirty = false;
    MixedMoePool active;
};

class MoeCachedSource {
public:
    const ::tyloi::MfeTensor& source_tensor() const { return *cpu_; }
    tyloi::weights::Placement placement() const {
        tyloi::weights::Placement result;
        result.devices.emplace_back(tyloi::DeviceType::cuda, cache_->owner_device_);
        if (quant_source_) result.devices.emplace_back(tyloi::DeviceType::cpu);
        else result.caller = tyloi::DeviceType::cuda;
        return result;
    }
    bool pipeline_enabled() const { return cache_->options_.policy.pipeline; }
    void add_bundle(const MoeCachedSource& up, const MoeCachedSource& down);
    void begin_routes(const MoeRoutePlan& route);
    void finish_routes(const MoeRoutePlan& route);
    bool two_stage_compatible() const {
        return std::all_of(cohorts_.begin(),cohorts_.end(),[](const auto& cohort) {
            const auto& p=*cohort.cpu;
            if(p.family==MixedMoeFamily::Nint)return p.nint.gs>0 && p.nint.gs<=64;
            if(p.family==MixedMoeFamily::Mxfp4)return p.mxfp4.neuron_len%32==0;
            return p.family==MixedMoeFamily::Nvq && p.nvq.gs==24 &&
                p.nvq.kernel_format>=1 && p.nvq.kernel_format<=17;
        });
    }
    MoeCachedSource(
            MoeExpertCache * cache,
            int id,
            std::string name,
            std::shared_ptr<::tyloi::MfeTensor> cpu,
            int minimum_slots,
            int layer_id,
            std::string projection_role,
            std::shared_ptr<tyloi::cuda::MfeMxfp4ExpertStore> range_store,
            std::shared_ptr<MoeQuantRangeSource> quant_source = {})
        : cache_(cache),
          id_(id),
          name_(std::move(name)),
          layer_id_(layer_id),
          projection_role_(std::move(projection_role)),
          cpu_(std::move(cpu)),
          range_store_(std::move(range_store)),
          quant_source_(std::move(quant_source)),
          expert_to_cohort_(
              static_cast<size_t>(cpu_->n_experts), -1),
          expert_to_local_(
              static_cast<size_t>(cpu_->n_experts), -1) {
        if (minimum_slots <= 0) {
            throw std::runtime_error(
                "MoE cache source minimum slots must be positive");
        }
        cohorts_.reserve(cpu_->pools.size());
        if (quant_source_) {
            const auto& store=quant_source_->store();
            if (cpu_->pools.size()!=store.pool_count() || cpu_->n_experts!=store.num_experts() ||
                cpu_->out_per_expert!=store.out_per_expert() || cpu_->neuron_len!=store.neuron_len())
                throw std::runtime_error("quantized range cache geometry mismatch");
            for (std::size_t index=0; index<cpu_->pools.size(); ++index) {
                auto& pool=cpu_->pools[index];
                auto prototype=pool;
                prototype.local_experts=1;
                auto layouts=moe_cache_field_layouts(prototype);
                MoeCachedCohort cohort;
                cohort.index=static_cast<int>(index); cohort.cpu=&pool;
                cohort.arena=cache_->register_cohort_layout(prototype,cpu_->out_per_expert,cpu_->neuron_len,
                    minimum_slots,pool.local_experts,std::move(layouts));
                for (const auto& layout:cohort.arena->layouts) cohort.bytes_per_expert.push_back(layout.elements*layout.element_size);
                cohort.mapped_fields.assign(cohort.bytes_per_expert.size(),nullptr);
                cohort.host_map.assign(cpu_->n_experts,-1);
                cohort.expert_to_local.assign(cpu_->n_experts,-1);
                const auto& ids=store.pool_expert_ids(index);
                for (std::size_t local=0; local<ids.size(); ++local) {
                    expert_to_cohort_[ids[local]]=static_cast<int>(index);
                    expert_to_local_[ids[local]]=static_cast<int>(local);
                    cohort.expert_to_local[ids[local]]=static_cast<int>(local);
                }
                // Retain only the shared table and geometry after registration.
                // Cache misses materialize one expert's fields through the source.
                if (pool.family==MixedMoeFamily::Nint) {
                    pool.nint.q_packed={}; pool.nint.row_q_bits={}; pool.nint.row_q_bit_offsets={};
                    pool.nint.sub_scale={}; pool.nint.sub_min={}; pool.nint.neuron_scale={}; pool.nint.neuron_min={};
                } else if(pool.family==MixedMoeFamily::Mxfp4) {
                    pool.mxfp4.values={};pool.mxfp4.scales={};
                } else {
                    pool.nvq.indices_packed={}; pool.nvq.aux_packed={}; pool.nvq.sub_scale_packed={}; pool.nvq.neuron_scale={};
                }
                cohorts_.push_back(std::move(cohort));
            }
        } else if (range_store_) {
            if (cpu_->pools.size() != 1 ||
                    cpu_->pools.front().family != MixedMoeFamily::Mxfp4 ||
                    cpu_->n_experts != range_store_->num_experts() ||
                    cpu_->out_per_expert != range_store_->out_per_expert() ||
                    cpu_->neuron_len != range_store_->neuron_len() ||
                    range_store_->values_bytes_per_expert() >
                        static_cast<uint64_t>(
                            std::numeric_limits<int64_t>::max()) ||
                    range_store_->scales_bytes_per_expert() >
                        static_cast<uint64_t>(
                            std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error(
                    "invalid exact-range MXFP4 cache metadata");
            }
            const auto & pool = cpu_->pools.front();
            MoeCachedCohort cohort;
            cohort.index = 0;
            cohort.cpu = &pool;
            cohort.range_store = range_store_;
            const int64_t values = static_cast<int64_t>(
                range_store_->values_bytes_per_expert());
            const int64_t scales = static_cast<int64_t>(
                range_store_->scales_bytes_per_expert());
            cohort.arena = cache_->register_cohort_layout(
                pool,
                cpu_->out_per_expert,
                cpu_->neuron_len,
                minimum_slots,
                cpu_->n_experts,
                {
                    {
                        ::tyloi::kUInt8,
                        {cpu_->out_per_expert, cpu_->neuron_len / 2},
                        values,
                        1,
                    },
                    {
                        ::tyloi::kUInt8,
                        {cpu_->out_per_expert, cpu_->neuron_len / 32},
                        scales,
                        1,
                    },
                });
            cohort.bytes_per_expert = {values, scales};
            cohort.mapped_fields = {nullptr, nullptr};
            cohort.expert_to_local.resize(
                static_cast<size_t>(cpu_->n_experts));
            cohort.host_map.assign(
                static_cast<size_t>(cpu_->n_experts), -1);
            for (int expert = 0; expert < cpu_->n_experts; ++expert) {
                cohort.expert_to_local[static_cast<size_t>(expert)] = expert;
                expert_to_cohort_[static_cast<size_t>(expert)] = 0;
                expert_to_local_[static_cast<size_t>(expert)] = expert;
            }
            cohorts_.push_back(std::move(cohort));
        } else {
            for (int cohort_index = 0;
                 cohort_index < static_cast<int>(cpu_->pools.size());
                 ++cohort_index) {
                const auto & pool =
                    cpu_->pools.at(static_cast<size_t>(cohort_index));
                MoeCachedCohort cohort;
                cohort.index = cohort_index;
                cohort.cpu = &pool;
                cohort.arena = cache_->register_cohort(
                    pool, cpu_->out_per_expert, cpu_->neuron_len,
                    minimum_slots);
                cohort.cpu_fields = moe_cache_fields(pool);
                cohort.expert_to_local.assign(
                    static_cast<size_t>(cpu_->n_experts), -1);
                cohort.host_map.assign(
                    static_cast<size_t>(cpu_->n_experts), -1);
                const auto * local =
                    pool.expert_local.data_ptr<int32_t>();
                for (int expert = 0; expert < cpu_->n_experts; ++expert) {
                    const int local_index = local[expert];
                    cohort.expert_to_local[
                        static_cast<size_t>(expert)] = local_index;
                    if (local_index < 0) continue;
                    if (expert_to_cohort_[
                            static_cast<size_t>(expert)] >= 0) {
                        throw std::runtime_error(
                            "MoE cache source has duplicate expert ownership");
                    }
                    expert_to_cohort_[
                        static_cast<size_t>(expert)] = cohort_index;
                    expert_to_local_[
                        static_cast<size_t>(expert)] = local_index;
                }
                for (const auto & field : cohort.cpu_fields) {
                    const int64_t nbytes = tensor_nbytes(field);
                    if (nbytes % pool.local_experts != 0) {
                        throw std::runtime_error(
                            "MoE cache field byte count is not expert aligned");
                    }
                    cohort.bytes_per_expert.push_back(
                        nbytes / pool.local_experts);
                    cohort.mapped_fields.push_back(
                        cache_->register_mapped_field(field));
                }
                cohorts_.push_back(std::move(cohort));
            }
        }
        if (std::any_of(
                expert_to_cohort_.begin(),
                expert_to_cohort_.end(),
                [](int value) { return value < 0; })) {
            throw std::runtime_error(
                "MoE cache source does not cover every expert");
        }
    }

    int id() const noexcept {
        return id_;
    }

    const std::string & name() const noexcept {
        return name_;
    }

    int layer_id() const noexcept {
        return layer_id_;
    }

    const std::string & projection_role() const noexcept {
        return projection_role_;
    }

    int n_experts() const noexcept {
        return cpu_->n_experts;
    }

    MoeGpuArena * arena_for_expert(int expert) const {
        if (expert < 0 || expert >= cpu_->n_experts) {
            throw std::out_of_range(
                "MoE cache profile expert is out of range");
        }
        const int cohort = expert_to_cohort_.at(
            static_cast<size_t>(expert));
        return cohorts_.at(static_cast<size_t>(cohort)).arena;
    }

    int64_t bytes_for_expert(int expert) const {
        return arena_for_expert(expert)->slot_bytes;
    }

    void touch_expert(int expert) {
        const int cohort = expert_to_cohort_.at(
            static_cast<size_t>(expert));
        auto & arena = *cohorts_.at(
            static_cast<size_t>(cohort)).arena;
        const int slot = arena.book->slot_for(
            {id_, cohort, expert});
        if (slot < 0) {
            throw std::runtime_error(
                "prewarmed MoE expert is absent from its arena");
        }
        arena.book->touch(slot);
    }

    int64_t host_bytes() const {
        return static_cast<int64_t>(cpu_->backing_nbytes()) +
            (quant_source_ ? quant_source_->store().index_nbytes() : 0);
    }

    void finalize() {
        const MfqCudaGuard guard(cache_->device());
        if (active_) return;
        active_ = std::make_shared<::tyloi::MfeTensor>();
        active_->n_experts = cpu_->n_experts;
        active_->out_per_expert = cpu_->out_per_expert;
        active_->neuron_len = cpu_->neuron_len;
        active_->pools.reserve(cohorts_.size());
        for (auto & cohort : cohorts_) {
            auto & source = *cohort.cpu;
            auto & arena = *cohort.arena;
            MixedMoePool pool = source;
            pool.local_experts = arena.slots;
            pool.expert_local = ::tyloi::cuda::full(
                {cpu_->n_experts}, -1,
                ::tyloi::TensorOptions()
                    .device(cache_->device())
                    .dtype(::tyloi::kInt32));
            size_t field = 0;
            if (pool.family == MixedMoeFamily::Nint) {
                pool.nint.aligned_q8 = false;
                pool.nint.q_packed = arena.fields.at(field++);
                pool.nint.row_q_bits = arena.fields.at(field++);
                pool.nint.row_q_bit_offsets = arena.fields.at(field++);
                pool.nint.sub_scale = arena.fields.at(field++);
                pool.nint.sub_min = arena.fields.at(field++);
                pool.nint.neuron_scale = arena.fields.at(field++);
                pool.nint.neuron_min = arena.fields.at(field++);
                pool.nint.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.nint.shape.empty()) {
                    pool.nint.shape[0] = pool.nint.out;
                }
            } else if (pool.family == MixedMoeFamily::Nint8Zero) {
                pool.q8_zero.q_packed = arena.fields.at(field++);
                pool.q8_zero.q8_zero_scale = arena.fields.at(field++);
                pool.q8_zero.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.q8_zero.shape.empty()) {
                    pool.q8_zero.shape[0] = pool.q8_zero.out;
                }
            } else if (pool.family == MixedMoeFamily::Mxfp4) {
                pool.mxfp4.values = arena.fields.at(field++);
                pool.mxfp4.scales = arena.fields.at(field++);
                pool.mxfp4.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
            } else if (pool.family == MixedMoeFamily::Nvq) {
                pool.nvq.indices_packed = arena.fields.at(field++);
                pool.nvq.aux_packed = arena.fields.at(field++);
                pool.nvq.sub_scale_packed = arena.fields.at(field++);
                pool.nvq.neuron_scale = arena.fields.at(field++);
                pool.nvq.codebook =
                    copy_cpu_weight_to_cuda(source.nvq.codebook);
                pool.nvq.out =
                    static_cast<int64_t>(arena.slots) *
                    cpu_->out_per_expert;
                if (!pool.nvq.shape.empty()) {
                    pool.nvq.shape[0] = pool.nvq.out;
                }
            } else {
                pool.nepq.indices_packed = arena.fields.at(field++);
                pool.nepq.aux_packed = arena.fields.at(field++);
                pool.nepq.state_packed = arena.fields.at(field++);
                pool.nepq.neuron_scale = arena.fields.at(field++);
                pool.nepq.bank_ids = arena.fields.at(field++);
                pool.nepq.table_pool =
                    copy_cpu_weight_to_cuda(source.nepq.table_pool);
                pool.nepq.grouped_table_pool =
                    copy_cpu_weight_to_cuda(
                        source.nepq.grouped_table_pool);
                pool.nepq.rotation_signs =
                    copy_cpu_weight_to_cuda(
                        source.nepq.rotation_signs);
                if (pool.nepq.residual) {
                    pool.nepq.residual_codebook =
                        copy_cpu_weight_to_cuda(
                            source.nepq.residual_codebook);
                    pool.nepq.residual_first =
                        arena.fields.at(field++);
                    pool.nepq.residual_second =
                        arena.fields.at(field++);
                }
                pool.nepq.n_experts = arena.slots;
            }
            if (field != arena.fields.size()) {
                throw std::runtime_error(
                    "MoE cache active field count does not match its arena");
            }
            cohort.active = std::move(pool);
            active_->pools.push_back(cohort.active);
        }
    }

    void invalidate(int cohort_index, int expert, int slot) {
        if (cohort_index < 0 ||
            cohort_index >= static_cast<int>(cohorts_.size()) ||
            expert < 0 || expert >= cpu_->n_experts) {
            throw std::runtime_error(
                "invalid MoE cache eviction key");
        }
        auto & cohort =
            cohorts_.at(static_cast<size_t>(cohort_index));
        if (cohort.host_map[static_cast<size_t>(expert)] == slot) {
            cohort.host_map[static_cast<size_t>(expert)] = -1;
            cohort.map_dirty = true;
        }
    }

    std::vector<int32_t> route_experts(
            const MoeRoutePlan & route) const {
        cache_->finish_pipeline_exchanges();
        if (!route.host_unique_experts) {
            auto ids = cache_->read_route_ids(route, cpu_->n_experts);
            cache_->observe_route(id_, layer_id_, route, ids, cpu_->n_experts);
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            route.host_unique_experts =
                std::make_shared<std::vector<int32_t>>(std::move(ids));
        }
        return *route.host_unique_experts;
    }

    void begin_prefetch(const MoeRoutePlan & route) {
        if (cache_->complete_residency()) return;
        if (use_full_projection(route)) return;
        cache_->begin_route_experts(route, cpu_->n_experts);
    }

    bool use_full_projection(const MoeRoutePlan & route) const {
        return !range_store_ && !quant_source_ && route.ids.size(0) > 8;
    }

    tyloi::weights::PreparedExperts prepare(
            const MoeRoutePlan& route, ::tyloi::Device input_device,
            tyloi::weights::ExpertProjection projection, bool heterogeneous);

    void prefetch(const MoeRoutePlan & route) {
        if (cache_->complete_residency()) return;
        if (use_full_projection(route)) return;
        (void)cache_->prepare(
            *this, route_experts(route), true);
    }

    bool supports_clamped_swiglu() const {
        return tyloi::cuda::supports_clamp(*cpu_);
    }

    static bool prefetch_bundle(
            const std::vector<std::shared_ptr<MoeCachedSource>> & sources,
            const MoeRoutePlan & route) {
        if (sources.empty()) return false;
        auto & first = *sources.front();
        if (first.cache_->complete_residency()) return false;
        if (first.use_full_projection(route)) return false;
        std::vector<MoeCachedSource *> raw_sources;
        raw_sources.reserve(sources.size());
        for (const auto & source : sources) {
            if (!source || source->cache_ != first.cache_ ||
                    source->n_experts() != first.n_experts() ||
                    source->layer_id() != first.layer_id() ||
                    source->use_full_projection(route)) {
                return false;
            }
            raw_sources.push_back(source.get());
        }
        if (raw_sources.size() == 3 &&
                raw_sources[0]->projection_role_ == "gate" &&
                raw_sources[1]->projection_role_ == "up" &&
                raw_sources[2]->projection_role_ == "down" &&
                raw_sources[2]->range_store_) {
            return first.cache_->prepare_bundle_deferred(
                {raw_sources[0], raw_sources[1]},
                *raw_sources[2],
                first.route_experts(route));
        }
        if (raw_sources.size() == 2 &&
                raw_sources[0]->projection_role_ == "gate_up" &&
                raw_sources[1]->projection_role_ == "down" &&
                raw_sources[1]->range_store_) {
            return first.cache_->prepare_bundle_deferred(
                {raw_sources[0]},
                *raw_sources[1],
                first.route_experts(route));
        }
        return first.cache_->prepare_bundle(
            raw_sources, first.route_experts(route));
    }

private:
    friend class MoeExpertCache;
    friend class MoeResidencyManager;
    friend struct MoeResidencyTestAccess;
    friend struct tyloi::weights::PreparedExperts;

    std::shared_ptr<::tyloi::MfeTensor> fallback_runtime() {
        if (quant_source_) throw std::runtime_error("quantized range cache cannot stage a full projection");
        if (!range_store_) return cpu_;
        std::lock_guard<std::mutex> guard(fallback_mutex_);
        if (!fallback_cpu_) {
            fallback_cpu_ = make_mfe(
                tyloi::mfe::unpack(range_store_->read_blob()), false);
        }
        return fallback_cpu_;
    }

    std::shared_ptr<const tyloi::MfeTensor> stage_fallback_runtime(
            bool heterogeneous) {
        auto runtime = fallback_runtime();
        return stage_cpu_mixed_moe(runtime, heterogeneous);
    }

    MoeExpertCache * cache_ = nullptr;
    int id_ = -1;
    std::string name_;
    int layer_id_ = -1;
    std::string projection_role_;
    std::shared_ptr<::tyloi::MfeTensor> cpu_;
    std::shared_ptr<tyloi::cuda::MfeMxfp4ExpertStore> range_store_;
    std::shared_ptr<MoeQuantRangeSource> quant_source_;
    std::mutex fallback_mutex_;
    std::shared_ptr<::tyloi::MfeTensor> fallback_cpu_;
    std::vector<MoeCachedCohort> cohorts_;
    std::vector<int> expert_to_cohort_;
    std::vector<int> expert_to_local_;
    std::shared_ptr<::tyloi::MfeTensor> active_;
};
