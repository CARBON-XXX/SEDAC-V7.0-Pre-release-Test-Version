import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            return
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _snapshot_download(model_id: str, cache_dir: str | None, local_files_only: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: huggingface_hub") from e
    p = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        ignore_patterns=["*.msgpack"],
    )
    return Path(p)


def _compute_num_layers(config: dict[str, Any]) -> int:
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        v = config.get(k)
        if isinstance(v, int) and v > 0:
            return v
    raise RuntimeError("Cannot find num_hidden_layers/n_layer/num_layers in config.json")


def _set_num_layers(config: dict[str, Any], n: int) -> None:
    if "num_hidden_layers" in config:
        config["num_hidden_layers"] = int(n)
        return
    if "n_layer" in config:
        config["n_layer"] = int(n)
        return
    config["num_hidden_layers"] = int(n)


def _load_exit_head(exit_head_path: Path) -> Any:
    import torch

    if exit_head_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency: safetensors") from e
        sd = load_file(str(exit_head_path), device="cpu")
        if not isinstance(sd, dict):
            raise RuntimeError(f"Exit head safetensors must be a dict: {exit_head_path}")
        return sd
    sd = torch.load(exit_head_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise RuntimeError(f"Exit head state_dict must be a dict: {exit_head_path}")
    return sd


def _write_override_safetensors(dst_dir: Path, tensor_name: str, state_dict: Any) -> Path:
    import torch
    from safetensors.torch import save_file  # type: ignore

    w = state_dict.get("weight")
    if w is None:
        raise RuntimeError("Exit head state_dict must contain key 'weight'")
    if not isinstance(w, torch.Tensor):
        raise RuntimeError("Exit head weight must be a torch.Tensor")
    out_path = dst_dir / "sedac_override.safetensors"
    save_file({tensor_name: w.contiguous()}, str(out_path))
    return out_path


def _write_bundle_safetensors(dst_dir: Path, state_dict: Any, out_name: str) -> tuple[Path, list[str]]:
    import torch
    from safetensors.torch import save_file  # type: ignore

    if not isinstance(state_dict, dict):
        raise RuntimeError("Speculator state_dict must be a dict[str, Tensor]")
    tensors: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, torch.Tensor):
            tensors[k] = v.contiguous()
    if not tensors:
        raise RuntimeError("No tensors found in speculator state_dict")
    out_path = dst_dir / out_name
    save_file(tensors, str(out_path))
    return out_path, sorted(tensors.keys())


def _update_index(index: dict[str, Any], tensor_name: str, override_file: str) -> dict[str, Any]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("Invalid model.safetensors.index.json: missing weight_map")
    weight_map = dict(weight_map)
    weight_map[tensor_name] = override_file
    out = dict(index)
    out["weight_map"] = weight_map
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["draft_model", "speculator"], default="draft_model")
    ap.add_argument("--depth-ratio", type=float, default=0.7)
    ap.add_argument("--num-layers", type=int, default=0)
    ap.add_argument("--cache-dir", type=str, default="")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--exit-head-path", type=str, default="")
    ap.add_argument("--head-tensor-name", type=str, default="lm_head.weight")
    ap.add_argument("--speculator-bundle-name", type=str, default="sedac_speculator.safetensors")
    args = ap.parse_args()

    src = _snapshot_download(
        model_id=str(args.model_id),
        cache_dir=(str(args.cache_dir) if str(args.cache_dir) else None),
        local_files_only=bool(args.local_files_only),
    )
    dst = Path(str(args.out_dir))
    dst.mkdir(parents=True, exist_ok=True)

    src_config_path = src / "config.json"
    if not src_config_path.exists():
        raise RuntimeError(f"Missing config.json in snapshot: {src}")
    cfg = _load_json(src_config_path)
    mode = str(args.mode)
    if mode == "speculator":
        out_cfg: dict[str, Any] = {
            "sedac_speculator": True,
            "base_model_id": str(args.model_id),
            "model_type": cfg.get("model_type"),
            "hidden_size": cfg.get("hidden_size") or cfg.get("d_model"),
            "vocab_size": cfg.get("vocab_size"),
        }
        _write_json(dst / "config.json", out_cfg)
    else:
        total_layers = _compute_num_layers(cfg)
        if int(args.num_layers) > 0:
            draft_layers = int(args.num_layers)
        else:
            ratio = float(args.depth_ratio)
            if ratio <= 0.0 or ratio >= 1.0:
                raise RuntimeError("--depth-ratio must be in (0,1) unless --num-layers is set")
            draft_layers = max(1, min(total_layers, int(round(total_layers * ratio))))
        _set_num_layers(cfg, draft_layers)
        _write_json(dst / "config.json", cfg)

    for name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        p = src / name
        if p.exists():
            _hardlink_or_copy(p, dst / name)

    if mode == "speculator":
        if not str(args.exit_head_path).strip():
            raise RuntimeError("--exit-head-path is required for --mode speculator")
        spec_sd = _load_exit_head(Path(str(args.exit_head_path)))
        bundle_path, keys = _write_bundle_safetensors(dst, spec_sd, str(args.speculator_bundle_name))
        _write_json(
            dst / "model.safetensors.index.json",
            {
                "metadata": {"total_size": os.path.getsize(bundle_path)},
                "weight_map": {k: bundle_path.name for k in keys},
            },
        )
        return 0

    index_path = src / "model.safetensors.index.json"
    override_written = False
    if str(args.exit_head_path).strip():
        exit_sd = _load_exit_head(Path(str(args.exit_head_path)))
        _write_override_safetensors(dst, str(args.head_tensor_name), exit_sd)
        override_written = True

    if index_path.exists():
        index = _load_json(index_path)
        if override_written:
            index = _update_index(index, str(args.head_tensor_name), "sedac_override.safetensors")
        _write_json(dst / "model.safetensors.index.json", index)

        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            needed_files = sorted({str(v) for v in weight_map.values() if isinstance(v, str)})
            for fname in needed_files:
                if fname == "sedac_override.safetensors":
                    continue
                sp = src / fname
                if sp.exists():
                    _hardlink_or_copy(sp, dst / fname)
    else:
        for f in src.iterdir():
            if f.is_file() and f.name.endswith(".safetensors"):
                _hardlink_or_copy(f, dst / f.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
