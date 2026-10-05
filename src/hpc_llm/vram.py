"""Small, explicitly approximate single-GPU context-memory planning estimates."""
from __future__ import annotations

from typing import Any

_GIB = 1024 ** 3
# Other architectures can use sliding/shared/recurrent caches: do not apply MHA math.
_DENSE = {"llama", "qwen2", "qwen3"}
_CACHE_BLOCKS = {"f16": (1, 2), "q8_0": (32, 34), "q4_0": (32, 18)}


def _positive(value: Any) -> int | None:
    return value if type(value) is int and 0 < value <= 2 ** 60 else None


def _nonnegative(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 2 ** 60 else None


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f} GiB"


def _kv_bytes(metadata: dict, settings: dict) -> int | None:
    """Full-attention KV with conservative 256-token padding and quant block scales.

    GGML q8_0 stores 34 bytes/32 values; q4_0 stores 18 bytes/32 values.
    See ggml/src/ggml-common.h in ggml-org/llama.cpp. This is not a runtime allocator.
    """
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or arch not in _DENSE:
        return None
    if any("sliding" in key or ".ssm." in key or "recurrent" in key
           for key, value in metadata.items() if value and isinstance(key, str)):
        return None
    layers = _positive(metadata.get(f"{arch}.block_count"))
    heads = _positive(metadata.get(f"{arch}.attention.head_count"))
    kv_heads = _positive(metadata.get(f"{arch}.attention.head_count_kv"))
    embed = _positive(metadata.get(f"{arch}.embedding_length"))
    context = _positive(settings.get("context"))
    if not all((layers, heads, kv_heads, embed, context)) or kv_heads > heads:
        return None
    fallback = embed // heads if embed % heads == 0 else None
    key_dim = _positive(metadata.get(f"{arch}.attention.key_length", fallback))
    value_dim = _positive(metadata.get(f"{arch}.attention.value_length", fallback))
    if not key_dim or not value_dim:
        return None
    padded = ((context + 255) // 256) * 256
    result = 0
    for kind, dim in (("k", key_dim), ("v", value_dim)):
        cache_type = settings.get(f"cache_type_{kind}", "f16")
        if not isinstance(cache_type, str) or cache_type not in _CACHE_BLOCKS:
            return None
        block, size = _CACHE_BLOCKS[cache_type]
        result += layers * padded * ((kv_heads * dim + block - 1) // block) * size
    return result


def estimate_vram(model: dict, settings: dict, capabilities: dict, resources: dict) -> str:
    """Return a plain-text advisory; never claim that a selected context will fit.

    Live measurements must refer to one visible allocated GPU and the same backend.
    GPU capacity comes only from these observations, never host RAM or GPU names.
    """
    context = _positive(settings.get("context"))
    total = _positive(capabilities.get("gpu_memory_total_bytes"))
    free = _nonnegative(capabilities.get("gpu_memory_free_bytes"))
    used = _positive(capabilities.get("gpu_memory_used_bytes"))
    observed_kv = _positive(capabilities.get("gpu_kv_bytes"))
    loaded_context = _positive(capabilities.get("loaded_context"))
    metadata = model.get("memory_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    gpu_count = resources.get("gpu_count", 1)
    unavailable = "VRAM estimate unavailable"
    if not context:
        return f"{unavailable}: enter a valid context size."
    if gpu_count != 1:
        return f"{unavailable}: multiple/zero GPUs; per-device placement is not estimated."

    # Never infer the distribution of layer weights, caches or companions across CPU/GPU.
    gpu_layers = settings.get("gpu_layers", -1)
    if type(gpu_layers) is not int or gpu_layers != -1:
        return f"{unavailable}: partial/explicit GPU offload needs measured allocation support."

    extras = bool(model.get("projector_path")) or settings.get("acceleration", "off") != "off"
    estimate: float | None = None
    details = ""
    # Changing runtime memory controls invalidates the loaded-process calibration.
    baseline = capabilities.get("memory_settings")
    memory_fields = {
        "gpu_layers": -1, "cache_type_k": "f16", "cache_type_v": "f16",
        "flash_attention": "auto", "acceleration": "off", "batch_size": 512,
        "ubatch_size": 128, "spec_draft_n_max": 3, "spec_draft_n_min": 0,
        "spec_draft_p_min": 0.0,
    }
    calibrated = isinstance(baseline, dict) and all(
        name in baseline and baseline[name] == settings.get(name, default)
        for name, default in memory_fields.items()
    )
    if used and observed_kv and loaded_context and observed_kv <= used and calibrated:
        padded = ((context + 255) // 256) * 256
        loaded_padded = ((loaded_context + 255) // 256) * 256
        # Shrinking unknown/hybrid caches need not release memory proportionally.
        growth = observed_kv * max(0, padded / loaded_padded - 1)
        reserve = max(_GIB, (used + growth) * 0.1)
        estimate = used + growth + reserve
        details = (f"Measured baseline {_gib(used)} + estimated KV growth {_gib(growth)}"
                   f" + {_gib(reserve)} allowance. Cache growth may be nonlinear.")
    else:
        kv = _kv_bytes(metadata, settings)
        weights = _positive(model.get("size_bytes"))
        if kv is not None and weights and not extras:
            reserve = max(2 * _GIB, (weights + kv) * 0.1)
            estimate = weights + kv + reserve
            details = f"Weights ~{_gib(weights)} + KV ~{_gib(kv)} + {_gib(reserve)} allowance."
    if estimate is None:
        reason = ("vision/MTP memory is unmeasured" if extras else "model cache layout or dimensions are unknown")
        observed = f" Current model uses {_gib(used)}." if used else ""
        return f"{unavailable}: {reason}.{observed} Lower context uses less cache; fit is not guaranteed."

    status = "GPU capacity unknown; fit is not checked."
    if total:
        # Free + this process is reclaimable on restart; other jobs are not.
        budget = min(total, free + (used or 0)) if free is not None and free <= total else total
        if estimate > budget:
            status = f"Likely exceeds available VRAM (~{_gib(budget)}); reduce context."
        elif estimate > budget * 0.85:
            status = f"Close to available VRAM (~{_gib(budget)}); reduce context for headroom."
        else:
            status = f"Below observed budget (~{_gib(budget)}); fit is not guaranteed."
    if extras:
        details += " Future image/MTP peaks are unmeasured."
    return f"VRAM ~{_gib(estimate)} at {context:,} context (approximate)\n{status}\n{details}"
