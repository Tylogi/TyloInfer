"""Architecture/capability contract published by the C++ transport and CUDA plan layer.

Structure gate, not a behavior test: it reads the native source. See conftest.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRANSPORT_SRC = ROOT / "csrc" / "transport"
SERVER = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(TRANSPORT_SRC.rglob("*"))
    if path.suffix in {".cpp", ".h"}
)
CUDA_PLAN = (
    ROOT / "csrc" / "backends" / "cuda" / "include" / "cuda_model_plan.h"
).read_text(encoding="utf-8")
CUDA_ROOT = ROOT / "csrc" / "backends" / "cuda"
CUDA_DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
CUDA_COMPONENTS = "\n".join(
    (CUDA_ROOT / "storage" / name).read_text(encoding="utf-8")
    for name in ("model_loader.h", "model_loader.cpp")
)



ROOT = Path(__file__).resolve().parents[3]


TRANSPORT_SRC = ROOT / "csrc" / "transport"


SERVER = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(TRANSPORT_SRC.rglob("*"))
    if path.suffix in {".cpp", ".h"}
)


CUDA_PLAN = (
    ROOT / "csrc" / "backends" / "cuda" / "include" / "cuda_model_plan.h"
).read_text(encoding="utf-8")


CUDA_ROOT = ROOT / "csrc" / "backends" / "cuda"


CUDA_DECODE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)


CUDA_COMPONENTS = "\n".join(
    (CUDA_ROOT / "storage" / name).read_text(encoding="utf-8")
    for name in ("model_loader.h", "model_loader.cpp")
)




def test_cpp_server_publishes_the_same_architecture_capability_contract() -> None:
    assert "architecture_capability_profile(" in SERVER
    assert "kModelCapabilityRegistry" in SERVER
    for model_type in (
        "minicpmo",
        "deepseek_v41",
        "deepseek_v41_vision",
        "deepseek_v4",
        "deepseek_v4_vision",
        "glm_moe_dsa",
        "qwen3_5",
        "qwen4_exp",
        "glm5_next",
    ):
        assert f'"{model_type}"' in SERVER
    for feature in (
        "text",
        "image_input",
        "video_input",
        "audio_input",
        "audio_output",
        "full_duplex",
        "mtp",
    ):
        assert f'{{"{feature}", profile.{feature}}}' in SERVER
    assert '{"model_capabilities", model_capabilities}' in SERVER
    for state in (
        "vision_supported",
        "vision_available",
        "video_available",
        "mtp_supported",
        "mtp_available",
        "vision_enabled_default",
        "mtp_enabled_default",
    ):
        assert f'"{state}"' in SERVER




def test_cpp_transports_keep_health_metrics_out_of_response_performance() -> None:
    assert SERVER.count("add_request_runtime_metrics(performance, metrics)") == 3
    assert "add_runtime_metrics(performance)" not in SERVER
    assert "append_generation_metrics(values, metrics.mtp)" in SERVER
    mtp_metrics = (ROOT / "csrc/engine/src/mtp_metrics.cpp").read_text()
    for metric in (
        "mtp_available",
        "mtp_used",
        "mtp_cycles",
        "mtp_drafted_tokens",
        "mtp_accepted_tokens",
        "mtp_acceptance_rate",
    ):
        assert f'"{metric}"' in SERVER + mtp_metrics




def test_cuda_uses_one_architecture_and_optional_component_registry() -> None:
    assert "CudaModelPlan" in CUDA_PLAN
    assert "CudaComponentState" in CUDA_PLAN
    assert "cuda_model_plan(" in CUDA_PLAN
    assert "cuda_component_state(" in CUDA_PLAN
    for implementation in (
        "deepseek_v4",
        "qwen3_5",
        "gemma4",
        "glm_dsa",
        "minicpmo45",
    ):
        assert f'"{implementation}"' in CUDA_PLAN

    assert "load_runtime_components(" in CUDA_DECODE
    assert "auto runtime_components" in CUDA_DECODE
    assert "RuntimeComponents<mfq::cuda::Qwen35CausalLm>" in CUDA_COMPONENTS
    assert "RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>" in CUDA_COMPONENTS
    assert "server_minicpmo_runtime" not in CUDA_DECODE
