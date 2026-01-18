import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Trial:
    method: str
    depth_ratio: float
    num_speculative_tokens: int
    draft_dir: str
    prompt_lookup: int | None


@dataclass(frozen=True)
class TrialResult:
    trial: Trial
    ppl: float
    tps: float
    acceptance_rate: float | None
    token_recovery_rate: float | None


def _parse_block_json(text: str, marker: str) -> dict[str, Any]:
    i = text.find(marker)
    if i < 0:
        raise RuntimeError(f"Missing marker: {marker}")
    j = text.find("{", i)
    if j < 0:
        raise RuntimeError(f"Missing JSON after marker: {marker}")
    depth = 0
    for k in range(j, len(text)):
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                raw = text[j : k + 1]
                out = json.loads(raw)
                if not isinstance(out, dict):
                    raise RuntimeError(f"Expected object JSON at {marker}")
                return out
    raise RuntimeError(f"Unterminated JSON after marker: {marker}")


def _start_server(
    *,
    conda_env: str,
    host: str,
    port: int,
    base_model: str,
    speculative_config_json: str | None,
    max_model_len: int,
    gpu_mem_util: float,
    extra_args: list[str],
) -> subprocess.Popen[bytes]:
    spec_arg = f" --speculative-config '{speculative_config_json}'" if speculative_config_json else ""
    cmd = [
        "bash",
        "-lc",
        "source /home/ason/miniconda3/etc/profile.d/conda.sh"
        f" && conda activate {conda_env}"
        " && export SEDAC_ENABLED=0"
        f" && export PROMETHEUS_MULTIPROC_DIR=/tmp/sedac_prom_{int(port)}"
        " && rm -rf \"$PROMETHEUS_MULTIPROC_DIR\" && mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\""
        " && python -m vllm.entrypoints.openai.api_server"
        f" --host {host}"
        f" --port {port}"
        f" --model '{base_model}'"
        f" --max-model-len {max_model_len}"
        f" --gpu-memory-utilization {gpu_mem_util}"
        + spec_arg
        + (" " + " ".join(extra_args) if extra_args else ""),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


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


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected object JSON: {path}")
    return obj


def _run_bench(*, base_url: str, model: str, max_tokens: int, warmup: int, repeat: int) -> TrialResult:
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-json", default="sedac_data/sedac_calibration.json")
    ap.add_argument("--method", type=str, default="ngram")
    ap.add_argument("--spec-model", type=str, default="")
    ap.add_argument("--speculative-token-tree", type=str, default="")
    ap.add_argument("--depth-ratios", default="0.6,0.7,0.8")
    ap.add_argument("--draft-tokens", default="2,3,4")
    ap.add_argument("--prompt-lookup", default="5,6,7")
    ap.add_argument("--num-layers", type=int, default=0)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--bench-max-tokens", type=int, default=128)
    ap.add_argument("--bench-warmup", type=int, default=1)
    ap.add_argument("--bench-repeat", type=int, default=3)
    ap.add_argument("--min-ar", type=float, default=0.4)
    ap.add_argument("--min-trr", type=float, default=0.4)
    ap.add_argument("--max-ppl-ratio", type=float, default=1.01)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--conda-env", default="sedac_dev")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--timeout-ready-s", type=float, default=240.0)
    ap.add_argument("--keep-server", action="store_true")
    args = ap.parse_args()

    method = str(args.method).strip().lower()
    spec_model = str(args.spec_model).strip()
    token_tree = str(args.speculative_token_tree).strip()
    depth_ratios = [float(x) for x in str(args.depth_ratios).split(",") if str(x).strip()]
    draft_tokens = [int(x) for x in str(args.draft_tokens).split(",") if str(x).strip()]
    prompt_lookup = [int(x) for x in str(args.prompt_lookup).split(",") if str(x).strip()]

    trials: list[Trial] = []
    if method == "ngram":
        for n in prompt_lookup:
            for t in draft_tokens:
                trials.append(Trial(method=method, depth_ratio=0.0, num_speculative_tokens=t, draft_dir="", prompt_lookup=n))
    elif method == "suffix":
        for t in draft_tokens:
            trials.append(Trial(method=method, depth_ratio=0.0, num_speculative_tokens=t, draft_dir="", prompt_lookup=None))
    elif method == "mtp":
        for t in draft_tokens:
            trials.append(Trial(method=method, depth_ratio=0.0, num_speculative_tokens=t, draft_dir="", prompt_lookup=None))
    elif method in ("eagle", "medusa"):
        if not spec_model:
            raise RuntimeError("--spec-model is required when --method is eagle/medusa")
        for t in draft_tokens:
            trials.append(Trial(method=method, depth_ratio=0.0, num_speculative_tokens=t, draft_dir=spec_model, prompt_lookup=None))
    elif method in ("draft_model", "draft-model"):
        raise RuntimeError("vLLM v1 path in 0.13.0 does not support draft_model; use ngram/suffix/medusa/eagle/mtp.")
    else:
        raise RuntimeError(f"Unsupported method: {method}")

    results: list[TrialResult] = []

    baseline_ppl: float | None = None
    baseline_tps: float | None = None
    if not bool(args.no_baseline):
        proc = _start_server(
            conda_env=str(args.conda_env),
            host=str(args.host),
            port=int(args.port),
            base_model=str(args.base_model),
            speculative_config_json=None,
            max_model_len=int(args.max_model_len),
            gpu_mem_util=float(args.gpu_memory_utilization),
            extra_args=[],
        )
        try:
            _wait_ready(proc, timeout_s=float(args.timeout_ready_s))
            _wait_http_ready(str(args.host), int(args.port), timeout_s=300.0)
            bench_json = f"/tmp/sedac_bench_baseline_{int(args.port)}.json"
            bench_cmd = [
                "bash",
                "-lc",
                "python3 '/mnt/g/SEDACV5.0 FAST/qwen7b_vllm_bench.py'"
                f" --base-url 'http://127.0.0.1:{int(args.port)}/v1'"
                f" --model '{str(args.base_model)}'"
                f" --max-tokens {int(args.bench_max_tokens)}"
                f" --warmup {int(args.bench_warmup)}"
                f" --repeat {int(args.bench_repeat)}"
                " --concurrency 1"
                " --ignore-eos 1"
                " --seed 1"
                f" --json-out '{bench_json}'"
                " --eval-ppl",
            ]
            try:
                subprocess.check_output(bench_cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
            except subprocess.CalledProcessError as e:
                msg = (e.output or b"").decode("utf-8", errors="replace")
                raise RuntimeError(f"Baseline benchmark failed (exit={e.returncode}). Output:\n{msg}") from e
            bench_obj = _read_json(bench_json)
            ppl_obj = bench_obj.get("ppl_a") if isinstance(bench_obj.get("ppl_a"), dict) else {}
            sum_obj = bench_obj.get("summary_a") if isinstance(bench_obj.get("summary_a"), dict) else {}
            baseline_ppl = float(ppl_obj.get("ppl", float("inf")))
            baseline_tps = (
                float(sum_obj.get("throughput", {}).get("avg_completion_tps", 0.0))
                if isinstance(sum_obj.get("throughput"), dict)
                else 0.0
            )
        finally:
            if not bool(args.keep_server):
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except Exception:
                    proc.kill()

    for tr in trials:
        if tr.method == "ngram":
            spec = {
                "method": "ngram",
                "num_speculative_tokens": int(tr.num_speculative_tokens),
                "prompt_lookup_min": int(tr.prompt_lookup or 0),
                "prompt_lookup_max": int(tr.prompt_lookup or 0),
            }
        elif tr.method == "suffix":
            spec = {"method": "suffix", "num_speculative_tokens": int(tr.num_speculative_tokens)}
        elif tr.method == "mtp":
            spec = {"method": "mtp", "num_speculative_tokens": int(tr.num_speculative_tokens)}
        elif tr.method in ("eagle", "medusa"):
            spec = {
                "method": tr.method,
                "model": str(tr.draft_dir),
                "num_speculative_tokens": int(tr.num_speculative_tokens),
            }
        else:
            raise RuntimeError("unreachable")
        if token_tree:
            spec["speculative_token_tree"] = token_tree
        spec_json = json.dumps(spec, ensure_ascii=False)
        proc = _start_server(
            conda_env=str(args.conda_env),
            host=str(args.host),
            port=int(args.port),
            base_model=str(args.base_model),
            speculative_config_json=spec_json.replace("'", "\\u0027"),
            max_model_len=int(args.max_model_len),
            gpu_mem_util=float(args.gpu_memory_utilization),
            extra_args=[],
        )
        try:
            _wait_ready(proc, timeout_s=float(args.timeout_ready_s))
            _wait_http_ready(str(args.host), int(args.port), timeout_s=300.0)
            bench_json = f"/tmp/sedac_bench_{int(args.port)}.json"
            bench_cmd = [
                "bash",
                "-lc",
                "python3 '/mnt/g/SEDACV5.0 FAST/qwen7b_vllm_bench.py'"
                f" --base-url 'http://127.0.0.1:{int(args.port)}/v1'"
                f" --model '{str(args.base_model)}'"
                f" --max-tokens {int(args.bench_max_tokens)}"
                f" --warmup {int(args.bench_warmup)}"
                f" --repeat {int(args.bench_repeat)}"
                " --concurrency 1"
                " --ignore-eos 1"
                " --seed 1"
                f" --json-out '{bench_json}'"
                " --eval-ppl",
            ]
            try:
                subprocess.check_output(bench_cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
            except subprocess.CalledProcessError as e:
                msg = (e.output or b"").decode("utf-8", errors="replace")
                raise RuntimeError(f"Benchmark failed (exit={e.returncode}). Output:\n{msg}") from e
            bench_obj = _read_json(bench_json)
            ppl_obj = bench_obj.get("ppl_a") if isinstance(bench_obj.get("ppl_a"), dict) else {}
            sum_obj = bench_obj.get("summary_a") if isinstance(bench_obj.get("summary_a"), dict) else {}
            spec_obj = bench_obj.get("spec_decode_metrics") if isinstance(bench_obj.get("spec_decode_metrics"), dict) else {}
            ppl = float(ppl_obj.get("ppl", float("inf")))
            tps = float(sum_obj.get("throughput", {}).get("avg_completion_tps", 0.0)) if isinstance(sum_obj.get("throughput"), dict) else 0.0
            ar = spec_obj.get("acceptance_rate")
            trr = spec_obj.get("token_recovery_rate")
            results.append(
                TrialResult(
                    trial=tr,
                    ppl=ppl,
                    tps=tps,
                    acceptance_rate=(float(ar) if isinstance(ar, (int, float)) else None),
                    token_recovery_rate=(float(trr) if isinstance(trr, (int, float)) else None),
                )
            )
        finally:
            if not bool(args.keep_server):
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except Exception:
                    proc.kill()

    min_ar = float(args.min_ar)
    min_trr = float(args.min_trr)
    max_ppl_ratio = float(args.max_ppl_ratio)
    eligible = [
        r
        for r in results
        if (r.acceptance_rate is None or r.acceptance_rate >= min_ar)
        and (r.token_recovery_rate is None or r.token_recovery_rate >= min_trr)
    ]
    best = max(eligible, key=lambda r: r.tps) if eligible else None
    out_json: dict[str, Any] = {
        "base_model": str(args.base_model),
        "method": method,
        "spec_model": spec_model or None,
        "speculative_token_tree": token_tree or None,
        "thresholds": {"min_ar": min_ar, "min_trr": min_trr, "max_ppl_ratio": max_ppl_ratio},
        "baseline": (
            {"ppl": baseline_ppl, "tps": baseline_tps}
            if (baseline_ppl is not None and baseline_tps is not None)
            else None
        ),
        "trials": [
            {
                "method": r.trial.method,
                "depth_ratio": r.trial.depth_ratio,
                "num_speculative_tokens": r.trial.num_speculative_tokens,
                "draft_dir": r.trial.draft_dir,
                "prompt_lookup": r.trial.prompt_lookup,
                "ppl": r.ppl,
                "tps": r.tps,
                "speedup_over_baseline": (r.tps / baseline_tps) if baseline_tps and baseline_tps > 0 else None,
                "ppl_ratio_over_baseline": (r.ppl / baseline_ppl) if baseline_ppl and baseline_ppl > 0 else None,
                "acceptance_rate": r.acceptance_rate,
                "token_recovery_rate": r.token_recovery_rate,
            }
            for r in results
        ],
        "best": (
            {
                "method": best.trial.method,
                "depth_ratio": best.trial.depth_ratio,
                "num_speculative_tokens": best.trial.num_speculative_tokens,
                "draft_dir": best.trial.draft_dir,
                "prompt_lookup": best.trial.prompt_lookup,
                "ppl": best.ppl,
                "tps": best.tps,
                "speedup_over_baseline": (best.tps / baseline_tps) if baseline_tps and baseline_tps > 0 else None,
                "ppl_ratio_over_baseline": (best.ppl / baseline_ppl) if baseline_ppl and baseline_ppl > 0 else None,
                "acceptance_rate": best.acceptance_rate,
                "token_recovery_rate": best.token_recovery_rate,
            }
            if best
            else None
        ),
        "enabled": bool(best is not None),
        "fuse_reason": ("no_config_meets_ar_trr_threshold" if best is None else None),
    }
    Path(str(args.out_json)).parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out_json)).write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_json.get("best") or {}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
