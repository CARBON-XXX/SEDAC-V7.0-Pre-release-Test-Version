
import argparse
import os
import sys


def _resolve_target_path(arg_target_path: str | None) -> str:
    if arg_target_path:
        return arg_target_path
    env_target = os.environ.get("SEDAC_VLLM_QWEN2_PATH")
    if env_target:
        return env_target
    try:
        import vllm.model_executor.models.qwen2 as qwen2  # type: ignore[import-not-found]

        return str(qwen2.__file__)
    except Exception:
        return "/home/ason/miniconda3/envs/sedac_dev/lib/python3.10/site-packages/vllm/model_executor/models/qwen2.py"

# Define the code snippets to insert
probe_def = """
SEDAC_PATCH_VERSION = 5

class LREProbe(nn.Module):
    def __init__(self, input_dim, rank=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, rank, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.head = nn.Linear(rank, 1)
        self.act = nn.Softplus()

    def forward(self, x):
        h = self.proj(x)
        h = self.norm(h)
        return self.act(self.head(h))
"""

init_code = """
        import os
        from vllm.logger import init_logger

        self._sedac_patch_begin = 5
        self._sedac_logger = init_logger("vllm.sedac")
        _sedac_enabled_env = os.environ.get("SEDAC_ENABLED", "0")
        self.sedac_enabled = _sedac_enabled_env.lower() in ("1", "true", "yes")
        self.sedac_mode = os.environ.get("SEDAC_MODE", "early_exit")
        self.sedac_layer = int(os.environ.get("SEDAC_LAYER", "21"))
        self.sedac_probe = None
        self.sedac_threshold = float(os.environ.get("SEDAC_THRESHOLD", "0.3"))
        self.sedac_adaptive = os.environ.get("SEDAC_ADAPTIVE", "0").lower() in ("1", "true", "yes")
        self.sedac_adaptive_alpha = float(os.environ.get("SEDAC_ADAPTIVE_ALPHA", "0.1"))
        self.sedac_adaptive_sensitivity = float(os.environ.get("SEDAC_ADAPTIVE_SENSITIVITY", "0.5"))
        self.sedac_avg_entropy = 0.0
        self.sedac_calibration_steps = int(os.environ.get("SEDAC_CALIBRATION_STEPS", "20"))
        try:
            _q = float(os.environ.get("SEDAC_CALIBRATION_QUANTILE", "0.9"))
        except Exception:
            _q = 0.9
        if _q < 0.0:
            _q = 0.0
        if _q > 1.0:
            _q = 1.0
        self.sedac_calibration_quantile = _q
        self.sedac_calibrated = False
        self.sedac_entropy_history = []
        # Latch disabled by default for safety
        self.sedac_latch = False 
        self._sedac_latched_exit = None
        self.sedac_log_every = int(os.environ.get("SEDAC_LOG_EVERY", "50"))
        self.sedac_calls = 0
        self.sedac_exit_calls = 0
        _probe_path_default = "/mnt/g/SEDACV5.0 FAST/sedac_data/sedac_probe_layer21.pth"
        _probe_path = os.environ.get("SEDAC_PROBE_PATH", _probe_path_default)
        self._sedac_logger.warning(
            "SEDAC patch v%d active enabled=%s env=%s mode=%s layer=%d threshold=%.6f probe_path=%s",
            5,
            self.sedac_enabled,
            _sedac_enabled_env,
            self.sedac_mode,
            self.sedac_layer,
            self.sedac_threshold,
            _probe_path,
        )

        if self.sedac_enabled:
            try:
                probe_path = _probe_path
                probe_rank = int(os.environ.get("SEDAC_PROBE_RANK", "64"))
                pp_group = get_pp_group()
                _pp_ws = getattr(pp_group, "world_size", 1)
                pp_world_size = int(_pp_ws() if callable(_pp_ws) else _pp_ws)
                if pp_world_size != 1:
                    self.sedac_enabled = False
                    self._sedac_logger.warning("SEDAC disabled: pipeline_parallel_world_size=%d", pp_world_size)
                elif not os.path.exists(probe_path):
                    self.sedac_enabled = False
                    self._sedac_logger.warning("SEDAC probe missing path=%s; disabled", probe_path)
                else:
                    probe = LREProbe(config.hidden_size, rank=probe_rank)
                    state_dict = torch.load(probe_path, map_location="cpu")
                    probe.load_state_dict(state_dict, strict=True)
                    probe.eval()
                    if torch.cuda.is_available():
                        probe = probe.to("cuda")
                    self.sedac_probe = probe
                    self._sedac_logger.warning(
                        "SEDAC probe loaded path=%s layer=%d threshold=%.6f mode=%s",
                        probe_path,
                        self.sedac_layer,
                        self.sedac_threshold,
                        self.sedac_mode,
                    )
            except Exception:
                self.sedac_enabled = False
                self._sedac_logger.exception("SEDAC probe load failed path=%s", os.environ.get("SEDAC_PROBE_PATH", ""))

        self._sedac_patch_end = 5
"""

forward_code = """
            _sedac_patch_forward_begin = 5
            if getattr(self, "sedac_enabled", False) and self.sedac_probe is not None and self.sedac_mode == "early_exit":
                abs_layer = idx + self.start_layer
                if abs_layer == self.sedac_layer:
                    with torch.no_grad():
                        # Latch logic REMOVED for PPL safety. 
                        # We always compute probe to ensure dynamic safety per batch.
                        
                        curr_res = residual if residual is not None else 0
                        current_h = hidden_states + curr_res
                        target_dtype = self.sedac_probe.proj.weight.dtype
                        if current_h.dtype != target_dtype:
                            current_h = current_h.to(target_dtype)
                        
                        predicted_entropy = self.sedac_probe(current_h)
                        # Use MAX entropy for safety (if any token is unsure, don't exit)
                        batch_entropy_metric = float(predicted_entropy.max().item())
                        
                        self.sedac_calls += 1
                        current_threshold = self.sedac_threshold

                        if getattr(self, "sedac_adaptive", False):
                            if not self.sedac_calibrated:
                                self.sedac_entropy_history.append(batch_entropy_metric)
                                if len(self.sedac_entropy_history) >= self.sedac_calibration_steps:
                                    sorted_ent = sorted(self.sedac_entropy_history)
                                    q_idx = int(len(sorted_ent) * float(getattr(self, "sedac_calibration_quantile", 0.9)))
                                    if q_idx < 0:
                                        q_idx = 0
                                    if q_idx >= len(sorted_ent):
                                        q_idx = len(sorted_ent) - 1
                                    self.sedac_threshold = sorted_ent[q_idx]
                                    self.sedac_avg_entropy = sum(sorted_ent) / len(sorted_ent)
                                    self.sedac_calibrated = True
                                    self._sedac_logger.warning(
                                        "SEDAC calibration done threshold=%.6f samples=%d",
                                        self.sedac_threshold,
                                        len(self.sedac_entropy_history),
                                    )
                                exit_now = False
                            else:
                                # Update moving average
                                self.sedac_avg_entropy = (1.0 - self.sedac_adaptive_alpha) * self.sedac_avg_entropy + self.sedac_adaptive_alpha * batch_entropy_metric
                                delta = (self.sedac_avg_entropy - batch_entropy_metric) * self.sedac_adaptive_sensitivity
                                current_threshold = self.sedac_threshold + delta
                                if current_threshold < 0.0:
                                    current_threshold = 0.0
                                exit_now = bool(batch_entropy_metric < current_threshold)
                        else:
                            exit_now = bool(batch_entropy_metric < current_threshold)

                        if exit_now:
                            self.sedac_exit_calls += 1
                            # SKIP MLP for remaining layers
                            for next_layer in self.layers[idx+1:]:
                                next_layer.sedac_skip_mlp = True

                        if exit_now or (self.sedac_log_every > 0 and (self.sedac_calls % self.sedac_log_every) == 0):
                            self._sedac_logger.warning(
                                "SEDAC decision abs_layer=%d max_entropy=%.6f threshold=%.6f base=%.6f exit=%s exits=%d calls=%d",
                                abs_layer,
                                batch_entropy_metric,
                                current_threshold,
                                self.sedac_threshold,
                                exit_now,
                                self.sedac_exit_calls,
                                self.sedac_calls,
                            )
            _sedac_patch_forward_end = 5
"""

decoder_patch = """
# --- SEDAC DECODER PATCH ---
def _sedac_decoder_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    
    # SEDAC: Skip MLP if flag is set
    if getattr(self, "sedac_skip_mlp", False):
        hidden_states.zero_()
        self.sedac_skip_mlp = False # Auto-reset
    else:
        hidden_states = self.mlp(hidden_states)
        
    return hidden_states, residual

Qwen2DecoderLayer.forward = _sedac_decoder_forward
# ---------------------------
"""

def _strip_sedac_blocks(lines: list[str]) -> list[str]:
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if stripped.startswith("SEDAC_PATCH_VERSION ="):
            i += 1
            continue

        if stripped.startswith("self._sedac_patch_begin ="):
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("self._sedac_patch_end ="):
                    i += 1
                    break
                i += 1
            continue

        if stripped.startswith("_sedac_patch_forward_begin ="):
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("_sedac_patch_forward_end ="):
                    i += 1
                    break
                i += 1
            continue

        if stripped == "# --- SEDAC DECODER PATCH ---":
             i += 1
             while i < len(lines):
                 if lines[i].strip() == "# ---------------------------":
                     i += 1
                     break
                 i += 1
             continue

        if stripped.startswith("Qwen2DecoderLayer.forward = _sedac_decoder_forward"):
            i += 1
            continue
            
        out.append(line)
        i += 1
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    target_path = _resolve_target_path(args.target)
    print(f"Patching target: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = _strip_sedac_blocks(lines)

    new_lines: list[str] = []
    inserted_probe = False
    inserted_init = False
    inserted_forward = False

    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Insert Probe Definition BEFORE Qwen2MLP class definition
        if not inserted_probe and "class Qwen2MLP(nn.Module):" in line:
             new_lines.pop() # Remove the line we just appended
             new_lines.append(probe_def)
             new_lines.append(line) # Add it back
             inserted_probe = True
             
        # Insert Init Code
        if not inserted_init and "self.aux_hidden_state_layers = tuple[int, ...]()" in line:
            new_lines.append(init_code)
            inserted_init = True
            
        # Insert Forward Code (AFTER the layer execution line)
        if not inserted_forward and "hidden_states, residual = layer(positions, hidden_states, residual)" in line:
            new_lines.append(forward_code)
            inserted_forward = True

    # Append Decoder Patch at the end
    new_lines.append(decoder_patch)

    # Safety check
    if not (inserted_probe and inserted_init and inserted_forward):
        print("Error: Could not find all insertion points.")
        print(f"Probe: {inserted_probe}, Init: {inserted_init}, Forward: {inserted_forward}")
        sys.exit(1)

    # Write back
    with open(target_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("Successfully patched qwen2.py")
