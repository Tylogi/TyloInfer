#include "run.h"
#include "cli.h"
#include "minicpmo45.h"
#include "engine/program.h"
#include "engine/models/causal_model.h"
#include "engine/models/source.h"
#include "engine/models/source/model_config.h"
#include "engine/models/prepare.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/weights/cache.h"
#include "diagnostics/backend_checks.h"
#include "diagnostics/model_checks.h"
#include "diagnostics/operator_checks.h"
#include "diagnostics/engines.h"
#include "diagnostics/source.h"
#include "diagnostics/text.h"
#include "diagnostics/vision.h"
#include "diagnostics/mtp.h"
#include "diagnostics/mtp_bench.h"
#include "diagnostics/qwen_mtp.h"
#include "diagnostics/qwen_decode.h"
#include "diagnostics/qwen_variants.h"
#include "diagnostics/prefill.h"
#include "diagnostics/batch.h"
#include "diagnostics/trace.h"
#include "diagnostics/policy.h"
#include "diagnostics/compare.h"
#include "diagnostics/tokens.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tyloi::cuda {

struct DiagnosticsCommandOptions : tyloi::engine::LoadOptions, TokenInputOptions {
    std::string check_linear, check_linear_cpu, check_linear_gate;
    std::string check_tp_linear, check_ep_moe;
    std::string check_tp_axis_arg = "output";
    std::string check_linear_group, check_q8_embedding;
    std::string check_gdn_input, check_gdn_output, check_gdn_state;
    std::string check_linear_conv_input, check_linear_conv_output;
    std::string check_mfe_tensor, check_dsv4_output_a;
    std::string prefill_sweep_arg;
    std::string check_moe_tokens = "1,2,4,8,16,32,64,128,256";
    std::string block_trace_reference, block_trace_output;
    std::string check_tokenizer_text =
        "MFQ tokenizer check: hello, world! <think>";
    std::string bench_qwen35_mtp;
    std::string kl_mmq_arg = "default";
    int bench_qwen35_mtp_reps = 3;
    int64_t minicpmo_eval_vision_batch_size = 16;
    int64_t block_trace_start = 0;
    int64_t block_trace_count = 0;
    int prefill_repeat = 0;
    int prefill_sweep_reps = 5;
    int check_linear_m = 1;
    int check_linear_reps = 200;
    int check_tp_m = 1;
    int check_ep_moe_tokens = 1;
    int check_ep_moe_routes = 2;
    int check_gdn_tokens = 512;
    int check_gdn_q_heads = 16;
    int check_gdn_v_heads = 32;
    int check_gdn_head_dim = 128;
    int check_linear_conv_tokens = 512;
    int check_linear_conv_q_heads = 16;
    int check_linear_conv_v_heads = 32;
    int check_linear_conv_key_dim = 128;
    int check_linear_conv_value_dim = 128;
    int check_linear_conv_kernel = 4;
    int check_gemma_geglu_layer = -1;
    int check_gemma_geglu_reps = 100;
    int check_moe_layer = -1;
    int check_moe_reps = 100;
    int check_mfe_tokens = 1;
    int check_mfe_routes = 2;
    int check_mfe_reps = 100;
    int check_mfe_split_width = 0;
    int check_dsv4_output_a_batch = 1;
    int check_dsv4_output_a_reps = 200;
    int check_attention_decode = 0;
    int check_attention_reps = 200;
    int check_attention_head_dim = 256;
    int check_attention_window = 4096;
    int compare_mma_decode_steps = 1;
    int compare_mma_decode_planned_len = 0;
    bool profile = false;
    bool ignore_eos = false;
    bool compare_ffn_transfer_phases = false;
    bool compare_router_lookahead = false;
    bool compare_rotary_fusion = false;
    bool compare_attention_grouping = false;
    bool compare_mhc_fusion = false;
    bool compare_mhc_shared_norm = false;
    bool compare_mhc_dense_fusion = false;
    bool compare_mhc_packed_groups = false;
    bool compare_mhc_fixed_formats = false;
    bool compare_mhc_async_normalized = false;
    bool compare_gdn_fixed_geometry = false;
    bool compare_gdn_inplace_state = false;
    bool compare_qsa_fusion = false;
    bool compare_mfe_parallel_gu = false;
    bool compare_mfe_gpu_bundle = false;
    bool compare_mfe_default_math = false;
    bool compare_mfe_e8_narrow = false;
    bool compare_mfe_shared_first = false;
    bool compare_mfe_early_gu = false;
    bool compare_cpu_transfer_budget = false;
    bool compare_cpu_background_calibration = false;
    bool compare_router_topk = false;
    bool compare_route_poll = false;
    bool compare_window_flush = false;
    bool compare_early_no_cpu_ready = false;
    bool compare_gpu_only_cpu_wait = false;
    bool compare_fixed_gpu_timing = false;
    bool compare_window_input_views = false;
    bool compare_route_spin = false;
    bool compare_moe_plan_reuse = false;
    bool compare_shared_gate_fusion = false;
    bool compare_graph_optimizations = false;
    int comparison_rounds = 1;
    bool comparison_phase_timings = false;
    bool compare_host_pin = false;
    bool compare_resident_overlap = false;
    bool compare_dma_graph_batch = false;
    bool compare_ram_field_packing = false;
    bool compare_warp_multi_sum = false;
    int warmup_prefill = -1;
    std::string router_lookahead_audit;
    std::string decode_inputs_file;
    std::vector<int64_t> decode_inputs;
    bool check_backend_bf16_add = false;
    bool check_backend_argmax = false;
    bool compare_mma_attention = false;
    bool compare_decode_splitk = false;
    bool compare_mma_decode = false;
    bool compare_nvq_vec4 = false;
    bool check_gemma4_swa = false;
    bool check_glm_dsa = false;
    bool check_dsv4_attention = false;
    bool check_dsv4_hc = false;
    bool check_deepseek_v41 = false;
    bool check_text_session_state = false;
    bool check_qwen35_mtp = false;
    bool check_continuous_batching = false;
    int check_continuous_batching_requests = 2;
    bool check_engine_isolation = false;
    bool check_flash_next = false;
    bool check_qwen4_decode_hidden = false;
    bool check_flash_next_mtp = false;
    bool check_nemotron_h = false;
    bool check_nemotron_h_mtp = false;
    bool check_nemotron_h_vision = false;
    bool compare_dsv4_hc_ops = false;
    bool compare_dsv4_hc_model = false;
    bool check_attention_swa_decode = false;
    bool check_mfe_routed_input = false;
    bool check_mfe_benchmark_only = false;
    bool check_runtime_assets = false;
    bool check_mfq_container = false;
    bool minicpmo_eval_batch = false;

    int runtime_comparison_modes() const {
        return int(compare_ffn_transfer_phases) + int(compare_router_lookahead) +
               int(compare_rotary_fusion) + int(compare_attention_grouping) + int(compare_mhc_fusion) + int(compare_mhc_shared_norm) + int(compare_mhc_dense_fusion) + int(compare_mhc_packed_groups) + int(compare_mhc_fixed_formats) + int(compare_mhc_async_normalized) + int(compare_gdn_fixed_geometry) + int(compare_gdn_inplace_state) + int(compare_qsa_fusion) + int(compare_mfe_parallel_gu) + int(compare_mfe_gpu_bundle) + int(compare_mfe_default_math) + int(compare_mfe_e8_narrow) + int(compare_mfe_shared_first) + int(compare_mfe_early_gu) + int(compare_cpu_transfer_budget) + int(compare_cpu_background_calibration) +
               int(compare_router_topk) + int(compare_route_poll) + int(compare_window_flush) + int(compare_early_no_cpu_ready) + int(compare_gpu_only_cpu_wait) + int(compare_fixed_gpu_timing) + int(compare_window_input_views) + int(compare_route_spin) + int(compare_moe_plan_reuse) + int(compare_shared_gate_fusion) + int(compare_graph_optimizations) + int(compare_host_pin) + int(compare_resident_overlap) + int(compare_dma_graph_batch) + int(compare_ram_field_packing) + int(compare_warp_multi_sum);
    }
    const char* runtime_comparison_name() const {
        if (compare_route_poll) return "route_poll_comparison";
        if (compare_window_flush) return "window_flush_comparison";
        if (compare_early_no_cpu_ready) return "early_no_cpu_ready_comparison";
        if (compare_gpu_only_cpu_wait) return "gpu_only_cpu_wait_comparison";
        if (compare_fixed_gpu_timing) return "fixed_gpu_timing_comparison";
        if (compare_window_input_views) return "window_input_views_comparison";
        if (compare_route_spin) return "route_spin_comparison";
        if (compare_moe_plan_reuse) return "moe_plan_reuse_comparison";
        if (compare_mfe_default_math) return "mfe_default_math_comparison";
        if (compare_mfe_e8_narrow) return "mfe_e8_narrow_comparison";
        if (compare_shared_gate_fusion) return "shared_gate_fusion_comparison";
        if (compare_graph_optimizations) return "graph_optimizations_comparison";
        if (compare_host_pin) return "host_pin_comparison";
        if (compare_resident_overlap) return "resident_overlap_comparison";
        if (compare_dma_graph_batch) return "dma_graph_batch_comparison";
        if (compare_ram_field_packing) return "ram_field_packing_comparison";
        if (compare_warp_multi_sum) return "warp_multi_sum_comparison";
        if (compare_router_topk) return "router_topk_comparison";
        if (compare_mhc_fusion) return "mhc_fusion_comparison";
        if (compare_mhc_shared_norm) return "mhc_shared_norm_comparison";
        if (compare_mhc_dense_fusion) return "mhc_dense_fusion_comparison";
        if (compare_mhc_packed_groups) return "mhc_packed_groups_comparison";
        if (compare_mhc_fixed_formats) return "mhc_fixed_formats_comparison";
        if (compare_mhc_async_normalized) return "mhc_async_normalized_comparison";
        if (compare_gdn_fixed_geometry) return "gdn_fixed_geometry_comparison";
        if (compare_gdn_inplace_state) return "gdn_inplace_state_comparison";
        if (compare_qsa_fusion) return "qsa_fusion_comparison";
        if (compare_mfe_parallel_gu) return "mfe_parallel_gu_comparison";
        if (compare_mfe_gpu_bundle) return "mfe_gpu_bundle_comparison";
        if (compare_mfe_shared_first) return "mfe_shared_first_comparison";
        if (compare_mfe_early_gu) return "mfe_early_gate_up_comparison";
        if (compare_cpu_background_calibration) return "cpu_background_calibration_comparison";
        if (compare_cpu_transfer_budget) return "cpu_transfer_budget_comparison";
        if (compare_attention_grouping) return "attention_grouping_comparison";
        if (compare_rotary_fusion) return "rotary_fusion_comparison";
        if (compare_router_lookahead) return "router_lookahead_comparison";
        return "ffn_transfer_comparison";
    }
};

} // namespace tyloi::cuda

using namespace tyloi::cuda::internal;

using namespace tyloi::cuda::diagnostics;

namespace {
struct DiagnosticsCommand : tyloi::cuda::DiagnosticsCommandOptions {
    explicit DiagnosticsCommand(tyloi::cuda::DiagnosticsCommandOptions options)
        : tyloi::cuda::DiagnosticsCommandOptions(std::move(options)) {}
    int run() { return with_command_errors([&]() -> int {
        if (check_engine_isolation) {
            tyloi::engine::EngineOptions options;
            static_cast<tyloi::engine::LoadOptions&>(options) = *this;
            return run_cuda_engine_isolation_check(std::move(options));
        }
        if (runtime_comparison_modes() > 1)
            throw std::runtime_error("select only one runtime comparison");
        if (comparison_phase_timings && !runtime_comparison_modes())
            throw std::runtime_error("comparison phase timings require a runtime comparison");
        if (compare_ffn_transfer_phases || compare_router_lookahead || compare_mfe_parallel_gu ||
            compare_mfe_gpu_bundle || compare_mfe_default_math || compare_mfe_e8_narrow ||
            compare_mfe_shared_first || compare_mfe_early_gu || compare_cpu_transfer_budget ||
            compare_cpu_background_calibration || compare_early_no_cpu_ready || compare_gpu_only_cpu_wait ||
            compare_fixed_gpu_timing || compare_moe_plan_reuse || compare_graph_optimizations ||
            compare_resident_overlap || compare_dma_graph_batch || compare_ram_field_packing)
            throw std::runtime_error(std::string(runtime_comparison_name()) + " is not available with Program execution");
        if (!router_lookahead_audit.empty())
            throw std::runtime_error("--router-lookahead-audit is not available with Program execution");
        if (compare_nvq_vec4)
            throw std::runtime_error("--compare-nvq-vec4 is not available with Program execution");
        if (context_size < 0) throw std::runtime_error("--ctx-size must be positive");
        if (check_continuous_batching) {
            TYLOINFER_RUNTIME_CHECK(!check_qwen35_mtp && !check_flash_next && !check_flash_next_mtp &&
                bench_qwen35_mtp.empty() && ids_arg.empty() && ids_file.empty() && prefill_sweep_arg.empty(),
                "continuous batching check cannot combine execution modes");
            if (context_size == 0) context_size = 512;
            TYLOINFER_RUNTIME_CHECK(context_size >= 32, "continuous batching check requires --ctx-size >= 32");
            return run_continuous_batching_check(*this, check_continuous_batching_requests);
        }
        std::vector<int64_t> prefill_sweep_sizes;
        if (!prefill_sweep_arg.empty()) {
            prefill_sweep_sizes = parse_ids(prefill_sweep_arg);
            TYLOINFER_RUNTIME_CHECK(std::all_of(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end(),
                [](int64_t value) { return value > 0; }), "--prefill-sweep values must be positive");
            if (context_size == 0) context_size = *std::max_element(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end());
        }
        if (!bench_qwen35_mtp.empty()) {
            TYLOINFER_RUNTIME_CHECK(bench_qwen35_mtp == "ordinary" || bench_qwen35_mtp == "mtp",
                "--bench-qwen35-mtp expects ordinary or mtp");
            TYLOINFER_RUNTIME_CHECK(!check_qwen35_mtp && ids_arg.empty() && ids_file.empty() && prefill_sweep_arg.empty(),
                "MTP benchmark cannot combine execution modes");
            if (context_size == 0) context_size = 512;
            TYLOINFER_RUNTIME_CHECK(gen >= 2 && context_size >= static_cast<int64_t>(gen) + 17 &&
                bench_qwen35_mtp_reps > 0 && bench_qwen35_mtp_reps <= 100,
                "MTP benchmark requires gen>=2, reps1-100 and context>=gen+17");
        }
        if (check_nemotron_h_vision || check_nemotron_h_mtp) {
            if (context_size == 0) context_size = 128;
            TYLOINFER_RUNTIME_CHECK(!check_nemotron_h_mtp || context_size >= 64,
                "Nemotron-H MTP correctness gate requires --ctx-size >= 64");
        }
        if (check_qwen35_mtp) {
            if (context_size == 0) context_size = 512;
            TYLOINFER_RUNTIME_CHECK(context_size >= 64, "MTP correctness gate requires --ctx-size >= 64");
        }
        const auto ids_vec = !ids_file.empty() ? load_ids_file(ids_file) :
            !ids_arg.empty() ? parse_ids(ids_arg) : std::vector<int64_t>{};
        if ((!block_trace_reference.empty() || !block_trace_output.empty()) && context_size == 0)
            context_size = static_cast<int64_t>(ids_vec.size());
        const auto t0 = std::chrono::steady_clock::now();
        auto program = model_path.empty() ? tyloi::cuda::Program::bare()
            : tyloi::cuda::Program::open(*this, !runtime_comparison_modes());
        auto& executor = program->executor();
        auto scope = executor.enter();
        if (check_backend_bf16_add) return run_backend_bf16_add_check(*program, 4096, 10000);
        if (check_backend_argmax) return run_backend_argmax_check(*program, 248320, 2000);
        if (!check_gdn_input.empty()) {
            TYLOINFER_RUNTIME_CHECK(!check_gdn_output.empty() && !check_gdn_state.empty(),
                "--check-gdn-input requires --check-gdn-output and --check-gdn-state");
            return run_gdn_operator_check(*program, check_gdn_input, check_gdn_output, check_gdn_state,
                check_gdn_tokens, check_gdn_q_heads, check_gdn_v_heads, check_gdn_head_dim);
        }
        if (!check_linear_conv_input.empty()) {
            TYLOINFER_RUNTIME_CHECK(!check_linear_conv_output.empty(),
                "--check-linear-conv-input requires --check-linear-conv-output");
            return run_linear_conv_operator_check(*program, check_linear_conv_input, check_linear_conv_output,
                check_linear_conv_tokens, check_linear_conv_q_heads, check_linear_conv_v_heads,
                check_linear_conv_key_dim, check_linear_conv_value_dim, check_linear_conv_kernel, 1.0e-6);
        }
        if (check_attention_decode > 0) return run_attention_decode_check(*program, check_attention_decode,
            check_attention_reps, check_attention_head_dim, check_attention_swa_decode, check_attention_window);
        if (check_gemma4_swa) return run_gemma4_swa_check(*program, check_attention_reps);
        if (check_glm_dsa) return run_glm_dsa_check(*program, check_attention_reps);
        if (check_dsv4_attention) return run_dsv4_attention_check(*program, check_attention_reps);
        if (check_dsv4_hc) return run_dsv4_hc_check(*program, check_attention_reps);
        if (check_deepseek_v41) return run_backend_dsv41_check(*program);
        if (check_text_session_state) return run_text_session_state_check(*program);
        if (model_path.empty()) throw std::runtime_error("missing model or execution mode (see --help)");
        auto gate_mode = InputGateMode::None;
        if (check_linear_gate == "sigmoid") gate_mode = InputGateMode::Sigmoid;
        else if (check_linear_gate == "silu") gate_mode = InputGateMode::Silu;
        else if (!check_linear_gate.empty()) throw std::runtime_error("--check-linear-gate must be sigmoid or silu");
        if (!check_linear_cpu.empty()) return run_cpu_linear_check(*program, check_linear_cpu, check_linear_m, gate_mode, check_linear_reps);
        if (!check_linear.empty()) return run_linear_check(*program, check_linear, check_linear_m, gate_mode, check_linear_reps);
        if (!check_tp_linear.empty()) {
            const auto axis = check_tp_axis_arg == "output" ? TensorParallelAxis::Output :
                check_tp_axis_arg == "input" ? TensorParallelAxis::Input :
                throw std::runtime_error("--check-tp-axis must be output or input");
            return run_tensor_parallel_linear_check(*program, check_tp_linear, axis, check_tp_m);
        }
        if (!check_ep_moe.empty()) return run_expert_parallel_moe_check(*program, check_ep_moe, check_ep_moe_tokens, check_ep_moe_routes);
        if (!check_linear_group.empty()) return run_linear_group_check(*program, check_linear_group, check_linear_m, check_linear_reps);
        if (!check_q8_embedding.empty()) return run_q8_embedding_check(*program, check_q8_embedding);
        if (check_gemma_geglu_layer >= 0) return run_gemma_geglu_check(*program, check_gemma_geglu_layer, check_gemma_geglu_reps);
        if (check_mfq_container) return run_container_check(*program);
        if (check_runtime_assets) return run_assets_check(*program, config_path, tokenizer_model, check_tokenizer_text);
        if (check_moe_layer >= 0) return run_moe_check(*program, config_path, check_moe_layer, parse_ids(check_moe_tokens), check_moe_reps);
        if (!check_mfe_tensor.empty()) {
            KlMmqScope mmq(parse_kl_mmq_mode(kl_mmq_arg), executor.mmq());
            const auto names = parse_tensor_names(check_mfe_tensor);
            if (!program->weights().empty() && names.size() != 1)
                throw std::runtime_error("cached --check-mfe-tensor accepts one tensor per invocation");
            for (const auto& name : names) {
                const int result = run_mfe_tensor_check(*program, name, check_mfe_tokens, check_mfe_routes,
                    check_mfe_reps, check_mfe_split_width, check_mfe_routed_input, check_mfe_benchmark_only);
                if (result) return result;
            }
            return 0;
        }
        if (!check_dsv4_output_a.empty()) return run_dsv4_output_a_check(*program, check_dsv4_output_a,
            check_dsv4_output_a_batch, check_dsv4_output_a_reps);
        if (minicpmo_eval_batch) {
            TYLOINFER_RUNTIME_CHECK(ids_arg.empty() && ids_file.empty(), "MiniCPM-o eval mode cannot combine token inputs");
            TYLOINFER_RUNTIME_CHECK(minicpmo_eval_vision_batch_size > 0, "--minicpmo-eval-vision-batch-size must be positive");
            return tyloi::cuda::minicpmo45::run_eval_batch(*program, context_size, minicpmo_eval_vision_batch_size);
        }
        program->load();
        const auto t1 = std::chrono::steady_clock::now();
        if (check_qwen4_decode_hidden) {
            const auto* variants = tyloi::cuda::env_lookup("TYLOINFER_MTP_GENERATION_VARIANTS");
            if (profile && variants && variants[0] == '1') return run_qwen4_generation_variants(*program, ids_vec);
            return profile ? run_qwen4_mtp_profile(*program, ids_vec) : run_qwen4_decode_hidden_check(*program);
        }
        if (check_flash_next) return run_flash_next_check(*program);
        if (check_nemotron_h) return run_nemotron_h_check(*program);
        if (check_nemotron_h_vision) return run_nemotron_h_vision_check(*program);
        if (check_nemotron_h_mtp) return run_nemotron_h_mtp_check(*program);
        if (check_flash_next_mtp) return run_flash_next_mtp_check(*program);
        if (check_qwen35_mtp) return run_qwen35_mtp_check(*program);
        if (!bench_qwen35_mtp.empty()) return run_qwen35_mtp_bench(*program, bench_qwen35_mtp == "mtp", gen, bench_qwen35_mtp_reps);
        if (!prefill_sweep_sizes.empty()) return run_prefill_sweep(*program, prefill_sweep_sizes, prefill_sweep_reps, profile);
        TYLOINFER_RUNTIME_CHECK(!ids_vec.empty(), "token ID lists require at least one integer");
        auto ids = tyloi::cuda::tensor(ids_vec, tyloi::TensorOptions().dtype(tyloi::kInt64).device(tyloi::kCUDA)).unsqueeze(0);
        if (compare_dsv4_hc_ops || compare_dsv4_hc_model) return run_hc_compare(*program, ids, compare_dsv4_hc_ops);
        if (!block_trace_reference.empty()) {
            auto reference = static_cast<tyloi::engine::LoadOptions>(*this);
            reference.model_path = block_trace_reference;
            return run_block_trace_compare(*program, std::move(reference), ids);
        }
        if (!block_trace_output.empty()) return run_block_trace_dump(*program, block_trace_output, ids, block_trace_start, block_trace_count);
        if (runtime_comparison_modes()) {
            using tyloi::cuda::diagnostics::Policy;
            Policy policy;
            if (compare_route_poll) policy = Policy::route_poll;
            else if (compare_window_flush) policy = Policy::window_flush;
            else if (compare_window_input_views) policy = Policy::window_input_views;
            else if (compare_route_spin) policy = Policy::route_spin;
            else if (compare_host_pin) policy = Policy::host_pin;
            else if (compare_shared_gate_fusion) policy = Policy::shared_gate_fusion;
            else if (compare_router_topk) policy = Policy::router_topk;
            else if (compare_warp_multi_sum) policy = Policy::warp_multi_sum;
            else if (compare_mhc_shared_norm) policy = Policy::mhc_shared_norm;
            else if (compare_mhc_dense_fusion) policy = Policy::mhc_dense_fusion;
            else if (compare_mhc_packed_groups) policy = Policy::mhc_packed_groups;
            else if (compare_mhc_fixed_formats) policy = Policy::mhc_fixed_formats;
            else if (compare_mhc_async_normalized) policy = Policy::mhc_async_normalized;
            else if (compare_gdn_fixed_geometry) policy = Policy::gdn_fixed_geometry;
            else if (compare_gdn_inplace_state) policy = Policy::gdn_inplace_state;
            else if (compare_qsa_fusion) policy = Policy::qsa_fusion;
            else if (compare_rotary_fusion) policy = Policy::rotary_fusion;
            else if (compare_attention_grouping) policy = Policy::attention_grouping;
            else if (compare_mhc_fusion) policy = Policy::mhc_fusion;
            else throw std::runtime_error("unsupported runtime comparison");
            return run_policy(*program, ids, {policy, comparison_rounds, warmup_prefill, gen,
                comparison_phase_timings, decode_inputs, t0, t1});
        }
        if (prefill_repeat > 0) {
            const auto* trace = tyloi::cuda::env_lookup("TYLOINFER_CHECK_PREFILL_REPEAT_TRACE");
            return repeat_prefill(*program, ids, prefill_repeat, trace && std::atoi(trace) != 0);
        }
        if (compare_decode_splitk) return tyloi::cuda::diagnostics::compare_decode_splitk(*program, ids);
        if (compare_mma_decode) return tyloi::cuda::diagnostics::compare_mma_decode(*program, ids,
            compare_mma_decode_steps, compare_mma_decode_planned_len);
        if (compare_mma_attention) return tyloi::cuda::diagnostics::compare_mma_attention(*program, ids);
        std::unordered_set<int64_t> stop_ids;
        if (!ignore_eos && tokenizer_model.empty() && decode_inputs.empty()) {
            auto source = tyloi::open_model_source(model_path);
            stop_ids = tyloi::models::stop_token_ids(tyloi::models::config_json(*source, config_path),
                program->model().info().vocab_size);
            if (stop_ids.empty()) std::cerr << "warning: no model EOS token IDs; generation is limited only by --gen\n";
        }
        return generate_diagnostic_tokens(*program, ids, gen, profile, t0, t1, decode_inputs, stop_ids,
            false, {warmup_prefill, false});
    }); }
 };

} // namespace

namespace tyloi::cuda {
int execute_diagnostics_command(DiagnosticsCommandOptions options) {
    return DiagnosticsCommand(std::move(options)).run();
}
} // namespace tyloi::cuda

namespace tyloi::cuda::commands {
namespace {

void print_diagnostics_help() {
    std::cout
        << "MFQ CUDA diagnostics\n\n"
        << "Usage:\n"
        << "  mfq-diagnostics [--model MODEL] CHECK [OPTIONS]\n\n"
        << "Checks and benchmarks:\n"
        << "  --check-linear NAME             check one GPU linear\n"
        << "  --check-linear-cpu NAME         check one CPU linear\n"
        << "  --check-linear-group NAMES      check a tensor group\n"
        << "  --check-q8-embedding NAME       check a Q8 embedding\n"
        << "  --check-tp-linear NAME          check tensor-parallel linear\n"
        << "  --check-ep-moe NAME             check expert-parallel MoE\n"
        << "  --check-gdn-input PATH          check GDN operator fixtures\n"
        << "  --check-linear-conv-input PATH  check linear-conv fixtures\n"
        << "  --check-moe-layer N             check a model MoE layer\n"
        << "  --check-mfe-tensor NAME         check an MFE tensor\n"
        << "  --check-dsv4-output-a NAME      check DeepSeek output-A\n"
        << "  --check-attention-decode N      check decode attention\n"
        << "  --check-gemma4-swa              check Gemma4 SWA\n"
        << "  --check-glm-dsa                 check GLM DSA\n"
        << "  --check-dsv4-attention          check DeepSeek-V4 attention\n"
        << "  --check-dsv4-hc                 check DeepSeek-V4 HC\n"
        << "  --check-deepseek-v41            run DeepSeek-V4.1 self-check\n"
        << "  --check-text-session-state      check session state\n"
        << "  --check-qwen35-mtp              check Qwen3.5 MTP\n"
        << "  --check-flash-next              check Flash-Next\n"
        << "  --check-qwen4-decode-hidden     check decode graph raw hidden and KV transitions\n"
        << "  --check-flash-next-mtp          check Flash-Next MTP\n"
        << "  --check-nemotron-h              check Nemotron-H text backbone\n"
        << "  --check-nemotron-h-mtp          check Nemotron-H multi-token prediction\n"
        << "  --check-nemotron-h-vision       check RADIO vision tower and projection\n"
        << "  --check-continuous-batching     check continuous batching\n"
        << "  --continuous-batching-check-requests N  check 2..32 concurrent requests\n"
        << "  --check-engine-isolation        check two concurrent CUDA engines\n"
        << "  --check-backend-bf16-add        check backend BF16 add\n"
        << "  --check-backend-argmax          check backend argmax\n"
        << "  --check-runtime-assets          validate embedded assets\n"
        << "  --check-mfq-container           validate the MFQ container\n"
        << "  --minicpmo-eval-batch           run MiniCPM-o batch evaluation\n"
        << "  --bench-qwen35-mtp MODE         benchmark ordinary or mtp mode\n"
        << "  --prefill-sweep LIST            benchmark prefill sizes\n"
        << "  --prefill-repeat N              repeat prefill N times\n"
        << "  --compare-block-trace PATH      compare a block trace\n"
        << "  --dump-block-trace PATH         write a block trace\n"
        << "  --compare-dsv4-hc-ops           compare DeepSeek HC operators\n"
        << "  --compare-dsv4-hc-model         compare DeepSeek HC model\n"
        << "  --compare-mma-attention         compare MMA attention\n"
        << "  --compare-decode-splitk         compare decode split-K\n"
        << "  --compare-mma-decode            compare MMA decode\n"
        << "  --compare-nvq-vec4              compare NVQ vec4\n"
        << "  --profile                       profile token generation\n\n"
        << "Common options:\n"
        << "  --model PATH                    model or split-model shard\n"
        << "  --config PATH                   external model config\n"
        << "  --ids LIST | --ids-file PATH    diagnostic token input\n"
        << "  --router-lookahead-audit PREFIX record actual FFN inputs and next-router logits; diagnostic only\n"
        << "  --compare-ffn-transfer-phases     one load, fixed primary tiers and CPU routes; baseline/phased/baseline\n"
        << "  --compare-router-lookahead       one load, fixed primary tiers and CPU routes; baseline/prefetch/baseline\n"
        << "  --compare-rotary-fusion          one load, fixed primary tiers and CPU routes; original/fused/original\n"
        << "  --compare-attention-grouping     one load, fixed primary tiers and CPU routes; separate/grouped/separate\n"
        << "  --compare-mhc-fusion             one load, fixed primary tiers and CPU routes; original/fused/original; honors TYLOINFER_GR_TWO_STAGE_DENSE_INJECTION\n"
        << "  --compare-mhc-shared-norm        one load, fixed primary tiers and CPU routes; fused/shared-norm/fused\n"
        << "  --compare-mhc-dense-fusion       one load, fixed primary tiers and CPU routes; original/fused/original BF16 injection\n"
        << "  --compare-mhc-packed-groups      one load, fixed routes; original/warp-cached metadata and 512-thread Up/original\n"
        << "  --compare-mfe-parallel-gu        one load, fixed primary tiers and CPU routes; serial/parallel/serial Gate-Up\n"
        << "  --compare-mfe-gpu-bundle         one load, fixed primary tiers and CPU routes; current/bundle/current FFN\n"
        << "  --compare-mfe-shared-first       one load, fixed primary tiers and CPU routes; off/on/off shared expert CTA order\n"
        << "  --compare-mfe-early-gate-up      one load, fixed primary tiers and CPU routes; off/on/off resident Gate-Up with cold Down\n"
        << "  --compare-cpu-background-calibration one load; synchronous/background/synchronous, fresh CPU costs\n"
        << "  --compare-cpu-transfer-budget    one load; original/conservative/original CPU policy; report natural route changes\n"
        << "  --compare-router-topk            one load, fixed primary tiers and CPU routes; original/cached/original\n"
        << "  --decode-inputs-file FILE        fixed decode input ids, raw int32, gen-1 entries\n"
        << "  --gen N                         generated tokens\n"
        << "  --ignore-eos                    fixed-length token diagnostics\n"
        << "  --compare-route-poll            compare route notification polling, 2000/0/2000 us\n"
        << "  --compare-window-flush          compare per-layer / one token / per-layer WDDM submission flush\n"
        << "  --compare-early-no-cpu-ready     compare late / early / late GPU-only CPU completion\n"
        << "  --compare-gpu-only-cpu-wait      compare kept / skipped / kept CPU wait with a full GPU quota\n"
        << "  --compare-fixed-gpu-timing      compare kept / skipped / kept rate events with a full GPU quota\n"
        << "  --compare-window-input-views    compare staged / producer / staged graph input addresses\n"
        << "  --compare-route-spin           compare yielding / Strata pause / yielding route waits\n"
        << "  --compare-moe-plan-reuse       one load, fixed primary tiers; allocating/reused/allocating host plans\n"
        << "  --compare-graph-optimizations  compare graph, shared-gate and residual optimizations off / on / off\n"
        << "  --compare-mhc-fixed-formats     compare generic / specialized / generic GR Down formats\n"
        << "  --compare-mhc-async-normalized  compare sync / async / sync GR Float activation copies\n"
        << "  --compare-gdn-fixed-geometry   compare generic / fixed / generic GDN geometry\n"
        << "  --compare-gdn-inplace-state    compare copied / inplace / copied GDN state\n"
        << "  --compare-shared-gate-fusion    compare cuBLAS / fused / cuBLAS shared scalar gates\n"
        << "  --compare-mfe-default-math      compare complete / default-only / complete MFE arithmetic kernels\n"
        << "  --compare-mfe-e8-narrow         compare wide / proven narrow / wide MFE E8 metadata kernels\n"
        << "  --comparison-rounds N           repeat the off/on/off runtime comparison (1..8)\n"
        << "  --comparison-phase-timings      append a separate off/on/off MFE timing round after performance passes\n"
        << "  --compare-host-pin              compare decode host placement, free/pinned/free\n"
        << "  --compare-resident-overlap      compare resident GPU work before host planning, off/on/off\n"
        << "  --compare-dma-graph-batch       compare ordinary/batched/ordinary weight DMA submissions\n"
        << "  --compare-ram-field-packing     compare direct/packed/direct small RAM fields, fixed decode inputs\n"
        << "  --compare-warp-multi-sum       compare original / Strata multi-row reduction / original\n"
        << "  --warmup-prefill N              prefill warmups, 0-100; default profile=1, otherwise=0\n"
        << "  --ctx-size N                    context size; 0 selects a default\n"
        << "  -t, --threads N                 positive CPU thread count\n"
        << "  -ngl, --n-gpu-layers N          non-negative GPU layer count\n"
        << "  --cpu-offload-layers RANGES     explicit CPU-offloaded layers\n"
        << "  --moe-gpu-cache-gb N            bounded MoE GPU cache\n"
        << "  --tensor-parallel DEVICES       tensor-parallel devices\n"
        << "  --expert-parallel DEVICES       expert-parallel devices\n"
        << "  --layer-parallel DEVICES        layer-placement devices\n"
        << "  -h, --help                      show this help\n";
}

bool has_diagnostic_action(const DiagnosticsCommandOptions& value) {
    return !value.check_linear.empty() || !value.check_linear_cpu.empty() ||
        !value.check_tp_linear.empty() || !value.check_ep_moe.empty() ||
        !value.check_linear_group.empty() || !value.check_q8_embedding.empty() ||
        !value.check_gdn_input.empty() || !value.check_linear_conv_input.empty() ||
        !value.check_mfe_tensor.empty() || !value.check_dsv4_output_a.empty() ||
        value.check_moe_layer >= 0 || value.check_gemma_geglu_layer >= 0 ||
        value.check_attention_decode > 0 || value.check_backend_bf16_add ||
        value.check_backend_argmax || value.check_gemma4_swa || value.check_glm_dsa ||
        value.check_dsv4_attention || value.check_dsv4_hc ||
        value.check_deepseek_v41 || value.check_text_session_state ||
        value.check_qwen35_mtp || value.check_continuous_batching ||
        value.check_engine_isolation ||
        value.check_flash_next || value.check_qwen4_decode_hidden || value.check_flash_next_mtp ||
        value.check_nemotron_h || value.check_nemotron_h_mtp ||
        value.check_nemotron_h_vision ||
        value.check_runtime_assets || value.check_mfq_container ||
        value.compare_dsv4_hc_ops || value.compare_dsv4_hc_model ||
        value.compare_mma_attention || value.compare_decode_splitk ||
        value.compare_mma_decode || value.compare_nvq_vec4 ||
        !value.block_trace_reference.empty() || !value.block_trace_output.empty() ||
        !value.bench_qwen35_mtp.empty() || !value.prefill_sweep_arg.empty() ||
        !value.ids_arg.empty() || !value.ids_file.empty() ||
        value.prefill_repeat > 0 || value.profile || value.minicpmo_eval_batch;
}

// Keep flag dispatch flat: MSVC limits nesting in long else-if chains.
bool parse_diagnostic_switch(std::string_view option, DiagnosticsCommandOptions& result) {
    static constexpr std::pair<std::string_view, bool DiagnosticsCommandOptions::*> switches[] = {
        {"--check-qwen4-decode-hidden", &DiagnosticsCommandOptions::check_qwen4_decode_hidden},
        {"--compare-ram-field-packing", &DiagnosticsCommandOptions::compare_ram_field_packing},
        {"--compare-qsa-fusion", &DiagnosticsCommandOptions::compare_qsa_fusion},
        {"--compare-moe-plan-reuse", &DiagnosticsCommandOptions::compare_moe_plan_reuse},
        {"--compare-mfe-default-math", &DiagnosticsCommandOptions::compare_mfe_default_math},
        {"--compare-mfe-e8-narrow", &DiagnosticsCommandOptions::compare_mfe_e8_narrow},
        {"--minicpmo-eval-batch", &DiagnosticsCommandOptions::minicpmo_eval_batch},
        {"--check-mfe-routed-input", &DiagnosticsCommandOptions::check_mfe_routed_input},
        {"--check-mfe-benchmark-only", &DiagnosticsCommandOptions::check_mfe_benchmark_only},
        {"--check-attention-swa-decode", &DiagnosticsCommandOptions::check_attention_swa_decode},
        {"--check-gemma4-swa", &DiagnosticsCommandOptions::check_gemma4_swa},
        {"--check-glm-dsa", &DiagnosticsCommandOptions::check_glm_dsa},
        {"--check-dsv4-attention", &DiagnosticsCommandOptions::check_dsv4_attention},
        {"--check-dsv4-hc", &DiagnosticsCommandOptions::check_dsv4_hc},
        {"--check-deepseek-v41", &DiagnosticsCommandOptions::check_deepseek_v41},
        {"--check-text-session-state", &DiagnosticsCommandOptions::check_text_session_state},
        {"--check-qwen35-mtp", &DiagnosticsCommandOptions::check_qwen35_mtp},
        {"--check-flash-next", &DiagnosticsCommandOptions::check_flash_next},
        {"--check-flash-next-mtp", &DiagnosticsCommandOptions::check_flash_next_mtp},
        {"--check-nemotron-h", &DiagnosticsCommandOptions::check_nemotron_h},
        {"--check-nemotron-h-mtp", &DiagnosticsCommandOptions::check_nemotron_h_mtp},
        {"--check-nemotron-h-vision", &DiagnosticsCommandOptions::check_nemotron_h_vision},
        {"--compare-dsv4-hc-ops", &DiagnosticsCommandOptions::compare_dsv4_hc_ops},
        {"--compare-dsv4-hc-model", &DiagnosticsCommandOptions::compare_dsv4_hc_model},
        {"--check-continuous-batching", &DiagnosticsCommandOptions::check_continuous_batching},
        {"--check-engine-isolation", &DiagnosticsCommandOptions::check_engine_isolation},
        {"--check-runtime-assets", &DiagnosticsCommandOptions::check_runtime_assets},
        {"--check-mfq-container", &DiagnosticsCommandOptions::check_mfq_container},
        {"--compare-ffn-transfer-phases", &DiagnosticsCommandOptions::compare_ffn_transfer_phases},
        {"--compare-route-poll", &DiagnosticsCommandOptions::compare_route_poll},
        {"--compare-window-flush", &DiagnosticsCommandOptions::compare_window_flush},
        {"--compare-early-no-cpu-ready", &DiagnosticsCommandOptions::compare_early_no_cpu_ready},
        {"--compare-gpu-only-cpu-wait", &DiagnosticsCommandOptions::compare_gpu_only_cpu_wait},
        {"--compare-fixed-gpu-timing", &DiagnosticsCommandOptions::compare_fixed_gpu_timing},
        {"--compare-window-input-views", &DiagnosticsCommandOptions::compare_window_input_views},
        {"--compare-route-spin", &DiagnosticsCommandOptions::compare_route_spin},
        {"--compare-shared-gate-fusion", &DiagnosticsCommandOptions::compare_shared_gate_fusion},
        {"--compare-graph-optimizations", &DiagnosticsCommandOptions::compare_graph_optimizations},
        {"--comparison-phase-timings", &DiagnosticsCommandOptions::comparison_phase_timings},
        {"--compare-host-pin", &DiagnosticsCommandOptions::compare_host_pin},
        {"--compare-resident-overlap", &DiagnosticsCommandOptions::compare_resident_overlap},
        {"--compare-dma-graph-batch", &DiagnosticsCommandOptions::compare_dma_graph_batch},
        {"--compare-warp-multi-sum", &DiagnosticsCommandOptions::compare_warp_multi_sum},
        {"--compare-router-lookahead", &DiagnosticsCommandOptions::compare_router_lookahead},
        {"--compare-rotary-fusion", &DiagnosticsCommandOptions::compare_rotary_fusion},
        {"--compare-attention-grouping", &DiagnosticsCommandOptions::compare_attention_grouping},
        {"--compare-mhc-fusion", &DiagnosticsCommandOptions::compare_mhc_fusion},
        {"--compare-mhc-shared-norm", &DiagnosticsCommandOptions::compare_mhc_shared_norm},
        {"--compare-mhc-dense-fusion", &DiagnosticsCommandOptions::compare_mhc_dense_fusion},
        {"--compare-mhc-packed-groups", &DiagnosticsCommandOptions::compare_mhc_packed_groups},
        {"--compare-mhc-fixed-formats", &DiagnosticsCommandOptions::compare_mhc_fixed_formats},
        {"--compare-mhc-async-normalized", &DiagnosticsCommandOptions::compare_mhc_async_normalized},
        {"--compare-gdn-fixed-geometry", &DiagnosticsCommandOptions::compare_gdn_fixed_geometry},
        {"--compare-gdn-inplace-state", &DiagnosticsCommandOptions::compare_gdn_inplace_state},
        {"--compare-mfe-parallel-gu", &DiagnosticsCommandOptions::compare_mfe_parallel_gu},
        {"--compare-mfe-gpu-bundle", &DiagnosticsCommandOptions::compare_mfe_gpu_bundle},
        {"--compare-mfe-shared-first", &DiagnosticsCommandOptions::compare_mfe_shared_first},
        {"--compare-mfe-early-gate-up", &DiagnosticsCommandOptions::compare_mfe_early_gu},
        {"--compare-cpu-background-calibration", &DiagnosticsCommandOptions::compare_cpu_background_calibration},
        {"--compare-cpu-transfer-budget", &DiagnosticsCommandOptions::compare_cpu_transfer_budget},
        {"--compare-router-topk", &DiagnosticsCommandOptions::compare_router_topk},
        {"--profile", &DiagnosticsCommandOptions::profile},
        {"--ignore-eos", &DiagnosticsCommandOptions::ignore_eos},
        {"--check-backend-bf16-add", &DiagnosticsCommandOptions::check_backend_bf16_add},
        {"--check-backend-argmax", &DiagnosticsCommandOptions::check_backend_argmax},
        {"--compare-mma-attention", &DiagnosticsCommandOptions::compare_mma_attention},
        {"--compare-decode-splitk", &DiagnosticsCommandOptions::compare_decode_splitk},
        {"--compare-mma-decode", &DiagnosticsCommandOptions::compare_mma_decode},
        {"--compare-nvq-vec4", &DiagnosticsCommandOptions::compare_nvq_vec4},
        {"--compare-niq-vec4", &DiagnosticsCommandOptions::compare_nvq_vec4},
    };
    for (const auto& [name, field] : switches) {
        if (option == name) {
            result.*field = true;
            return true;
        }
    }
    return false;
}

DiagnosticsCommandOptions parse_diagnostics(ArgCursor& args) {
    DiagnosticsCommandOptions result;
    result.device = tyloi::Device(tyloi::DeviceType::cuda);
    while (!args.empty()) {
        const std::string_view option = args.next();
        if (option == "--help" || option == "-h") {
            print_diagnostics_help();
            throw HelpRequested{};
        }
        if (parse_cuda_load_option(option, args, result) ||
                parse_token_input_option(option, args, result) ||
                parse_diagnostic_switch(option, result)) {
            continue;
        }
        if (option == "--continuous-batching-check-requests") {
            result.check_continuous_batching_requests = integer<int>(args.value(option),option);
            continue;
        }
        if (option == "--minicpmo-eval-vision-batch-size") {
            result.minicpmo_eval_vision_batch_size =
                integer<int64_t>(args.value(option), option);
            if (result.minicpmo_eval_vision_batch_size <= 0) {
                usage_error("--minicpmo-eval-vision-batch-size must be positive");
            }
        }
        else if (option == "--check-linear") result.check_linear = args.value(option);
        else if (option == "--check-linear-cpu") result.check_linear_cpu = args.value(option);
        else if (option == "--check-tp-linear") result.check_tp_linear = args.value(option);
        else if (option == "--check-tp-axis") result.check_tp_axis_arg = args.value(option);
        else if (option == "--check-tp-m") result.check_tp_m = integer<int>(args.value(option), option);
        else if (option == "--check-ep-moe" || option == "--check-tp-moe") {
            result.check_ep_moe = args.value(option);
        }
        else if (option == "--check-ep-moe-tokens" ||
                option == "--check-tp-moe-tokens") {
            result.check_ep_moe_tokens = integer<int>(args.value(option), option);
        }
        else if (option == "--check-ep-moe-routes" ||
                option == "--check-tp-moe-routes") {
            result.check_ep_moe_routes = integer<int>(args.value(option), option);
        }
        else if (option == "--check-linear-group") result.check_linear_group = args.value(option);
        else if (option == "--check-q8-embedding") result.check_q8_embedding = args.value(option);
        else if (option == "--check-gdn-input") result.check_gdn_input = args.value(option);
        else if (option == "--check-gdn-output") result.check_gdn_output = args.value(option);
        else if (option == "--check-gdn-state") result.check_gdn_state = args.value(option);
        else if (option == "--check-gdn-tokens") result.check_gdn_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-q-heads") result.check_gdn_q_heads = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-v-heads") result.check_gdn_v_heads = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-head-dim") result.check_gdn_head_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-input") result.check_linear_conv_input = args.value(option);
        else if (option == "--check-linear-conv-output") result.check_linear_conv_output = args.value(option);
        else if (option == "--check-linear-conv-tokens") result.check_linear_conv_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-q-heads") result.check_linear_conv_q_heads = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-v-heads") result.check_linear_conv_v_heads = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-key-dim") result.check_linear_conv_key_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-value-dim") result.check_linear_conv_value_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-kernel") result.check_linear_conv_kernel = integer<int>(args.value(option), option);
        else if (option == "--check-linear-m") result.check_linear_m = integer<int>(args.value(option), option);
        else if (option == "--check-linear-reps") result.check_linear_reps = integer<int>(args.value(option), option);
        else if (option == "--check-linear-gate") result.check_linear_gate = args.value(option);
        else if (option == "--check-gemma-geglu-layer") result.check_gemma_geglu_layer = integer<int>(args.value(option), option);
        else if (option == "--check-gemma-geglu-reps") result.check_gemma_geglu_reps = integer<int>(args.value(option), option);
        else if (option == "--check-moe-layer") result.check_moe_layer = integer<int>(args.value(option), option);
        else if (option == "--check-moe-tokens") {
            result.check_moe_tokens = args.value(option);
            validate_integer_list(result.check_moe_tokens, option);
        }
        else if (option == "--check-moe-reps") result.check_moe_reps = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-tensor") result.check_mfe_tensor = args.value(option);
        else if (option == "--check-mfe-tokens") result.check_mfe_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-routes") result.check_mfe_routes = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-reps") result.check_mfe_reps = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-split-width") result.check_mfe_split_width = integer<int>(args.value(option), option);
        else if (option == "--check-dsv4-output-a") result.check_dsv4_output_a = args.value(option);
        else if (option == "--check-dsv4-output-a-batch") result.check_dsv4_output_a_batch = integer<int>(args.value(option), option);
        else if (option == "--check-dsv4-output-a-reps") result.check_dsv4_output_a_reps = integer<int>(args.value(option), option);
        else if (option == "--check-attention-decode") result.check_attention_decode = integer<int>(args.value(option), option);
        else if (option == "--check-attention-reps") result.check_attention_reps = integer<int>(args.value(option), option);
        else if (option == "--check-attention-head-dim") result.check_attention_head_dim = integer<int>(args.value(option), option);
        else if (option == "--check-attention-window") result.check_attention_window = integer<int>(args.value(option), option);
        else if (option == "--bench-qwen35-mtp") result.bench_qwen35_mtp = args.value(option);
        else if (option == "--bench-qwen35-mtp-reps") result.bench_qwen35_mtp_reps = integer<int>(args.value(option), option);
        else if (option == "--kl-mmq") {
            result.kl_mmq_arg = args.value(option);
            try { (void)parse_kl_mmq_mode(result.kl_mmq_arg); }
            catch (const std::runtime_error& error) {
                usage_error(error.what());
            }
        }
        else if (option == "--compare-block-trace") result.block_trace_reference = args.value(option);
        else if (option == "--dump-block-trace") result.block_trace_output = args.value(option);
        else if (option == "--dump-block-trace-start") result.block_trace_start = integer<int64_t>(args.value(option), option);
        else if (option == "--dump-block-trace-count") result.block_trace_count = integer<int64_t>(args.value(option), option);
        else if (option == "--prefill-repeat") result.prefill_repeat = integer<int>(args.value(option), option);
        else if (option == "--prefill-sweep") {
            result.prefill_sweep_arg = args.value(option);
            validate_integer_list(result.prefill_sweep_arg, option);
        }
        else if (option == "--prefill-sweep-reps") result.prefill_sweep_reps = integer<int>(args.value(option), option);
        else if (option == "--check-tokenizer-text") result.check_tokenizer_text = args.value(option);
        else if (option == "--router-lookahead-audit") result.router_lookahead_audit = args.value(option);
        else if (option == "--comparison-rounds") {
            result.comparison_rounds=integer<int>(args.value(option),option);
            if(result.comparison_rounds<1 || result.comparison_rounds>8)
                throw std::invalid_argument("comparison rounds must be between one and eight");
        }
        else if (option == "--decode-inputs-file") result.decode_inputs_file = args.value(option);
        else if (option == "--warmup-prefill") {
            result.warmup_prefill = integer<int>(args.value(option), option);
            if (result.warmup_prefill < 0 || result.warmup_prefill > 100)
                usage_error("--warmup-prefill requires a count from 0 to 100");
        }
        else if (option == "--compare-mma-decode-steps") result.compare_mma_decode_steps = integer<int>(args.value(option), option);
        else if (option == "--compare-mma-decode-planned-len") result.compare_mma_decode_planned_len = integer<int>(args.value(option), option);
        else usage_error("unknown option: " + std::string(option));
    }
    if (!result.decode_inputs_file.empty()) {
        result.decode_inputs = load_ids_file(result.decode_inputs_file);
        if (result.gen <= 1 || result.decode_inputs.size() != size_t(result.gen - 1))
            usage_error("--decode-inputs-file requires exactly gen-1 entries and gen greater than 1");
        if (std::any_of(result.decode_inputs.begin(),result.decode_inputs.end(),[](int64_t id){return id<0;}))
            usage_error("--decode-inputs-file requires nonnegative token ids");
    }
    if (!result.ids_arg.empty() && !result.ids_file.empty()) {
        usage_error("--ids and --ids-file are mutually exclusive");
    }
    if (!has_diagnostic_action(result)) {
        usage_error("mfq-diagnostics requires a check or benchmark");
    }
    return result;
}

} // namespace

int run_diagnostics(int argc, char** argv) {
    try {
        ArgCursor args(argc, argv);
        auto options = parse_diagnostics(args);
        return tyloi::cuda::execute_diagnostics_command(std::move(options));
    } catch (const HelpRequested&) {
        return 0;
    } catch (const UsageError& error) {
        return print_usage_error(error);
    } catch (const std::exception& error) {
        std::cerr << "mfq-diagnostics: " << error.what() << '\n';
        return 1;
    }
}

} // namespace tyloi::cuda::commands
