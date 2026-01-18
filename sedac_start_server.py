import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def _wait_ready(proc: subprocess.Popen[bytes], timeout_s: float) -> None:
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            out = (bytes(buf) + (proc.stdout.read() if proc.stdout else b"")).decode("utf-8", errors="replace")
            raise RuntimeError(f"Server exited early:\n{out}")
        if proc.stdout is None:
            time.sleep(0.1)
            continue
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        buf += line
        s = line.decode("utf-8", errors="replace")
        print(s, end="", flush=True)
        if "Application startup complete" in s or "Uvicorn running on" in s:
            return
    raise RuntimeError("Server did not become ready in time")


def _wait_http_ready(host: str, port: int, timeout_s: float) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=2.0) as resp:
                if int(getattr(resp, "status", 200)) in (200, 401):
                    return
        except URLError:
            time.sleep(0.5)
            continue
        except Exception:
            time.sleep(0.5)
            continue
    raise RuntimeError(f"HTTP server not responding: {url}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conda-env", default="sedac_dev")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    ap.add_argument("--speculative-config-file", type=str, default="")
    ap.add_argument("--speculative-config", type=str, default="")
    ap.add_argument("--timeout-ready-s", type=float, default=600.0)
    ap.add_argument("--disable-sedac", action="store_true")
    ap.add_argument("--sedac-adaptive", action="store_true")
    ap.add_argument("--sedac-adaptive-alpha", type=float, default=0.1)
    ap.add_argument("--sedac-adaptive-sensitivity", type=float, default=0.5)
    ap.add_argument("--sedac-layer", type=int, default=21)
    ap.add_argument("--sedac-threshold", type=float, default=0.3)
    ap.add_argument("--sedac-calibration-steps", type=int, default=20)
    ap.add_argument("--sedac-calibration-quantile", type=float, default=0.9)
    ap.add_argument("--sedac-log-every", type=int, default=0)
    ap.add_argument(
        "--sedac-latch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force latching (safe for KV cache)",
    )
    args = ap.parse_args()

    spec_arg = ""
    spec_src = ""
    if str(args.speculative_config_file).strip():
        spec_src = Path(str(args.speculative_config_file)).read_text(encoding="utf-8")
    elif str(args.speculative_config).strip():
        spec_src = str(args.speculative_config)
    if spec_src.strip():
        spec_obj = json.loads(spec_src)
        if not isinstance(spec_obj, dict):
            raise RuntimeError("--speculative-config must be a JSON object")
        spec_json = json.dumps(spec_obj, ensure_ascii=False).replace("'", "\\u0027")
        spec_arg = f" --speculative-config '{spec_json}'"

    sedac_val = "0" if args.disable_sedac else "1"
    cmd = [
        "bash",
        "-lc",
        f"source /home/ason/miniconda3/etc/profile.d/conda.sh && conda activate {str(args.conda_env)}"
        " && export HF_HUB_OFFLINE=1"
        " && export TRANSFORMERS_OFFLINE=1"
        " && export NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1"
        " && export no_proxy=localhost,127.0.0.1,0.0.0.0,::1"
        f" && export PROMETHEUS_MULTIPROC_DIR=/tmp/sedac_prom_{int(args.port)}"
        " && rm -rf \"$PROMETHEUS_MULTIPROC_DIR\" && mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\""
        f" && export SEDAC_ENABLED={sedac_val}"
        f" && export SEDAC_ADAPTIVE={'1' if args.sedac_adaptive else '0'}"
        f" && export SEDAC_ADAPTIVE_ALPHA={float(args.sedac_adaptive_alpha)}"
        f" && export SEDAC_ADAPTIVE_SENSITIVITY={float(args.sedac_adaptive_sensitivity)}"
        f" && export SEDAC_LAYER={int(args.sedac_layer)}"
        f" && export SEDAC_THRESHOLD={float(args.sedac_threshold)}"
        f" && export SEDAC_CALIBRATION_STEPS={int(args.sedac_calibration_steps)}"
        f" && export SEDAC_CALIBRATION_QUANTILE={float(args.sedac_calibration_quantile)}"
        f" && export SEDAC_LOG_EVERY={int(args.sedac_log_every)}"
        f" && export SEDAC_LATCH={'1' if args.sedac_latch else '0'}"
        " && export SEDAC_PROBE_PATH=\"/mnt/g/SEDACV5.0 FAST/sedac_data/sedac_probe_layer21.pth\""
        " && python -m vllm.entrypoints.openai.api_server"
        f" --host {str(args.host)}"
        f" --port {int(args.port)}"
        f" --model '{str(args.model)}'"
        f" --max-model-len {int(args.max_model_len)}"
        f" --gpu-memory-utilization {float(args.gpu_memory_utilization)}"
        " --enforce-eager"
        + spec_arg,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _wait_ready(proc, timeout_s=float(args.timeout_ready_s))
    _wait_http_ready(str(args.host), int(args.port), timeout_s=float(args.timeout_ready_s))
    print(f"[sedac_start_server] ready: http://127.0.0.1:{int(args.port)}/v1", flush=True)
    try:
        while True:
            time.sleep(3600.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

