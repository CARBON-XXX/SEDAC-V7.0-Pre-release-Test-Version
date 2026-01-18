import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def _snapshot_download(model_id: str, cache_dir: str | None, local_files_only: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: huggingface_hub") from e
    p = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        ignore_patterns=["*.msgpack", "*.safetensors", "*.bin"],
    )
    return Path(p)


def _compute_num_layers(config: dict[str, Any]) -> int:
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        v = config.get(k)
        if isinstance(v, int) and v > 0:
            return v
    raise RuntimeError("Cannot find num_hidden_layers/n_layer/num_layers in config.json")


def _maybe_get_int(config: dict[str, Any], keys: list[str]) -> int | None:
    for k in keys:
        v = config.get(k)
        if isinstance(v, int):
            return v
    return None


def _guess_moe_layers(config: dict[str, Any], num_layers: int) -> list[int]:
    explicit = config.get("moe_layer_indices") or config.get("moe_layers") or config.get("expert_layer_indices")
    if isinstance(explicit, list) and all(isinstance(x, int) for x in explicit):
        return [int(x) for x in explicit if 0 <= int(x) < num_layers]

    freq = _maybe_get_int(config, ["moe_layer_freq", "moe_frequency", "expert_layer_freq"])
    if isinstance(freq, int) and freq > 0:
        return [i for i in range(num_layers) if (i + 1) % freq == 0]

    first_dense_replace = _maybe_get_int(config, ["first_k_dense_replace"])
    if isinstance(first_dense_replace, int) and 0 <= first_dense_replace < num_layers:
        return list(range(first_dense_replace, num_layers))

    return []


def _is_moe(config: dict[str, Any]) -> bool:
    for k in ("num_experts", "n_routed_experts", "moe_layer_freq", "expert_layer_freq", "moe_layer_indices", "moe_layers"):
        if k in config:
            return True
    return False


def _is_mla(config: dict[str, Any]) -> bool:
    mt = str(config.get("model_type") or "").lower()
    arch = ",".join([str(x) for x in (config.get("architectures") or [])]) if isinstance(config.get("architectures"), list) else ""
    s = (mt + " " + str(arch)).lower()
    return "deepseek" in s or "mla" in s or "latent" in s


def _recommend_taps(config: dict[str, Any]) -> dict[str, Any]:
    num_layers = _compute_num_layers(config)
    taps: list[int] = []

    if _is_moe(config):
        moe_layers = _guess_moe_layers(config, num_layers=num_layers)
        if moe_layers:
            taps.append(min(moe_layers))
            taps.append(min(num_layers - 1, taps[0] + 4))
        else:
            taps.append(min(num_layers - 1, max(0, int(round(num_layers * 0.5)))))
    else:
        taps.append(min(num_layers - 1, max(0, int(round(num_layers * 0.7)))))

    taps = sorted({int(x) for x in taps if 0 <= int(x) < num_layers})
    out: dict[str, Any] = {
        "model_type": config.get("model_type"),
        "num_layers": num_layers,
        "is_moe": _is_moe(config),
        "is_mla_like": _is_mla(config),
        "moe_layers_guess": _guess_moe_layers(config, num_layers=num_layers) if _is_moe(config) else [],
        "recommended_tap_layers": taps,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, default="")
    ap.add_argument("--config-path", type=str, default="")
    ap.add_argument("--cache-dir", type=str, default="")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    if str(args.config_path).strip():
        cfg_path = Path(str(args.config_path))
        cfg = _load_json(cfg_path)
    else:
        if not str(args.model_id).strip():
            raise RuntimeError("Either --config-path or --model-id must be provided")
        snap = _snapshot_download(
            model_id=str(args.model_id),
            cache_dir=(str(args.cache_dir) if str(args.cache_dir).strip() else None),
            local_files_only=bool(args.local_files_only),
        )
        cfg = _load_json(snap / "config.json")

    print(json.dumps(_recommend_taps(cfg), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

