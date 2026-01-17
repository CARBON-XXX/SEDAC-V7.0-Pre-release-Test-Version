
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
SEDAC_PATCH_VERSION = 3

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

        self._sedac_patch_begin = 3
        self._sedac_logger = init_logger("vllm.sedac")
        _sedac_enabled_env = os.environ.get("SEDAC_ENABLED", "0")
        self.sedac_enabled = _sedac_enabled_env.lower() in ("1", "true", "yes")
        self.sedac_mode = os.environ.get("SEDAC_MODE", "early_exit")
        self.sedac_layer = int(os.environ.get("SEDAC_LAYER", "21"))
        self.sedac_probe = None
        self.sedac_threshold = float(os.environ.get("SEDAC_THRESHOLD", "0.3"))
        self.sedac_latch = os.environ.get("SEDAC_LATCH", "0").lower() in ("1", "true", "yes")
        self.sedac_log_every = int(os.environ.get("SEDAC_LOG_EVERY", "50"))
        self.sedac_calls = 0
        self.sedac_exit_calls = 0
        _probe_path_default = "/mnt/g/SEDACV5.0 FAST/sedac_data/sedac_probe_layer21.pth"
        _probe_path = os.environ.get("SEDAC_PROBE_PATH", _probe_path_default)
        self._sedac_logger.warning(
            "SEDAC patch v%d active enabled=%s env=%s mode=%s layer=%d threshold=%.6f probe_path=%s",
            3,
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

        self._sedac_patch_end = 3
"""

forward_code = """
            _sedac_patch_forward_begin = 3
            if getattr(self, "sedac_enabled", False) and self.sedac_probe is not None and self.sedac_mode == "early_exit":
                abs_layer = idx + self.start_layer
                if abs_layer == self.sedac_layer:
                    with torch.no_grad():
                        curr_res = residual if residual is not None else 0
                        current_h = hidden_states + curr_res
                        current_h = torch.nan_to_num(current_h, nan=0.0)
                        target_dtype = self.sedac_probe.proj.weight.dtype
                        if current_h.dtype != target_dtype:
                            current_h = current_h.to(target_dtype)
                        predicted_entropy = self.sedac_probe(current_h)
                        mean_entropy = float(predicted_entropy.mean().item())
                        self.sedac_calls += 1
                        latched_exit = False
                        if getattr(self, "sedac_latch", False):
                            latched_exit = bool(getattr(self, "_sedac_latched_exit", False))
                        exit_now = latched_exit or (mean_entropy < self.sedac_threshold)
                        if exit_now and getattr(self, "sedac_latch", False) and not latched_exit:
                            self._sedac_latched_exit = True
                            latched_exit = True
                        if exit_now:
                            self.sedac_exit_calls += 1
                        if exit_now or (self.sedac_log_every > 0 and (self.sedac_calls % self.sedac_log_every) == 0):
                            self._sedac_logger.warning(
                                "SEDAC decision abs_layer=%d mean_entropy=%.6f threshold=%.6f exit=%s latched=%s exits=%d calls=%d",
                                abs_layer,
                                mean_entropy,
                                self.sedac_threshold,
                                exit_now,
                                latched_exit,
                                self.sedac_exit_calls,
                                self.sedac_calls,
                            )
                        if exit_now:
                            break
            _sedac_patch_forward_end = 3
"""

ap = argparse.ArgumentParser()
ap.add_argument("--target-path", default=None)
args = ap.parse_args()

target_path = _resolve_target_path(args.target_path)
print(f"Target qwen2.py: {target_path}")

# Read the file
with open(target_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def _strip_sedac_blocks(in_lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(in_lines):
        line = in_lines[i]
        stripped = line.strip()

        if stripped.startswith("SEDAC_PATCH_VERSION"):
            i += 1
            while i < len(in_lines):
                if "class Qwen2MLP(nn.Module):" in in_lines[i]:
                    break
                i += 1
            continue

        if stripped == "self._sedac_patch_begin = 3":
            i += 1
            while i < len(in_lines):
                if in_lines[i].strip() == "self._sedac_patch_end = 3":
                    i += 1
                    break
                i += 1
            continue

        if stripped == "_sedac_patch_forward_begin = 3":
            i += 1
            while i < len(in_lines):
                if in_lines[i].strip() == "_sedac_patch_forward_end = 3":
                    i += 1
                    break
                i += 1
            continue

        if stripped in (
            "# --- SEDAC Probe Definition ---",
            "# --- SEDAC INJECTION ---",
            "# --- SEDAC LOGIC ---",
        ):
            i += 1
            while i < len(in_lines):
                if in_lines[i].strip() in (
                    "# ------------------------------",
                    "# -----------------------",
                    "# -------------------",
                ):
                    i += 1
                    break
                i += 1
            continue

        out.append(line)
        i += 1
    return out


lines = _strip_sedac_blocks(lines)

new_lines: list[str] = []
inserted_probe = False
inserted_init = False
inserted_forward = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Insert Probe Definition BEFORE Qwen2MLP class definition
    if not inserted_probe and "class Qwen2MLP(nn.Module):" in line:
         # We just appended the line "class Qwen2MLP...", so we need to insert BEFORE it?
         # No, new_lines.append(line) is done. So we are inserting AFTER it.
         # Wait, class def should be at module level.
         # If I append after "class ...", it's inside the class? No.
         # I should insert BEFORE.
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

# Safety check
if not (inserted_probe and inserted_init and inserted_forward):
    print("Error: Could not find all insertion points.")
    print(f"Probe: {inserted_probe}, Init: {inserted_init}, Forward: {inserted_forward}")
    sys.exit(1)

# Write back
with open(target_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Successfully patched qwen2.py")
