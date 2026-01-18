from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


JsonValue = (
    str
    | int
    | float
    | bool
    | None
    | dict[str, "JsonValue"]
    | list["JsonValue"]
)
JsonDict = dict[str, JsonValue]


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected object JSON: {path}")
    return obj


def _write_json(path: Path, obj: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _start_server(
    *,
    port: int,
    host: str,
    conda_env: str,
    model: str,
    max_model_len: int,
    gpu_memory_util: float,
    speculative_config: dict[str, Any] | None,
    sedac_enabled: bool,
    sedac_adaptive: bool = False,
    sedac_adaptive_alpha: float = 0.1,
    sedac_adaptive_sensitivity: float = 0.5,
    sedac_latch: bool = True,
    sedac_threshold: float = 0.3,
    sedac_calibration_steps: int = 20,
    sedac_calibration_quantile: float = 0.9,
    sedac_log_every: int = 0,
    sedac_layer: int = 21,
) -> subprocess.Popen[bytes]:
    spec_arg = ""
    if speculative_config is not None:
        spec_json = json.dumps(speculative_config, ensure_ascii=False).replace("'", "\\u0027")
        spec_arg = f" --speculative-config '{spec_json}'"
    
    sedac_val = "1" if sedac_enabled else "0"
    sedac_adaptive_val = "1" if sedac_adaptive else "0"
    sedac_latch_val = "1" if sedac_latch else "0"
    cmd = [
        "bash",
        "-lc",
        "source /home/ason/miniconda3/etc/profile.d/conda.sh"
        f" && conda activate {conda_env}"
        " && export HF_HUB_OFFLINE=1"
        " && export TRANSFORMERS_OFFLINE=1"
        " && export NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1"
        " && export no_proxy=localhost,127.0.0.1,0.0.0.0,::1"
        f" && export SEDAC_ENABLED={sedac_val}"
        f" && export SEDAC_LAYER={int(sedac_layer)}"
        f" && export SEDAC_THRESHOLD={float(sedac_threshold)}"
        f" && export SEDAC_ADAPTIVE={sedac_adaptive_val}"
        f" && export SEDAC_ADAPTIVE_ALPHA={sedac_adaptive_alpha}"
        f" && export SEDAC_ADAPTIVE_SENSITIVITY={sedac_adaptive_sensitivity}"
        f" && export SEDAC_LATCH={sedac_latch_val}"
        f" && export SEDAC_CALIBRATION_STEPS={int(sedac_calibration_steps)}"
        f" && export SEDAC_CALIBRATION_QUANTILE={float(sedac_calibration_quantile)}"
        f" && export SEDAC_LOG_EVERY={int(sedac_log_every)}"
        " && export SEDAC_PROBE_PATH=\"/mnt/g/SEDACV5.0 FAST/sedac_data/sedac_probe_layer21.pth\""
        f" && export PROMETHEUS_MULTIPROC_DIR=/tmp/sedac_prom_{int(port)}"
        " && rm -rf \"$PROMETHEUS_MULTIPROC_DIR\" && mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\""
        " && python -m vllm.entrypoints.openai.api_server"
        f" --host {host}"
        f" --port {port}"
        f" --model '{model}'"
        f" --max-model-len {int(max_model_len)}"
        f" --gpu-memory-utilization {float(gpu_memory_util)}"
        " --enforce-eager"
        + spec_arg,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _wait_server_ready(proc: subprocess.Popen[bytes], timeout_s: float, verbose: bool) -> None:
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            out = (bytes(buf) + (proc.stdout.read() if proc.stdout else b"")).decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM server exited early:\n{out}")
        if proc.stdout is None:
            time.sleep(0.1)
            continue
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        buf += line
        s = line.decode("utf-8", errors="replace")
        if verbose:
            print(s.rstrip("\n"), flush=True)
        if "Application startup complete" in s or "Uvicorn running on" in s:
            return
    raise RuntimeError("vLLM server did not become ready in time")


def _wait_http_ready(port: int, timeout_s: float) -> None:
    url = f"http://127.0.0.1:{int(port)}/v1/models"
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

def _run_bench_online(
    *,
    port: int,
    model: str,
    max_tokens: int,
    warmup: int,
    repeat: int,
    concurrency: int,
    eval_ppl: bool,
    out_json: Path,
) -> dict[str, Any]:
    out_json = Path(out_json)
    timeout_s = 60.0 + float(max_tokens) * float(max(1, repeat)) * 2.0
    args = [
        "bash",
        "-lc",
        "python3 '/mnt/g/SEDACV5.0 FAST/qwen7b_vllm_bench.py'"
        f" --base-url 'http://127.0.0.1:{int(port)}/v1'"
        f" --model '{model}'"
        f" --max-tokens {int(max_tokens)}"
        f" --warmup {int(warmup)}"
        f" --repeat {int(repeat)}"
        f" --concurrency {int(concurrency)}"
        " --ignore-eos 1"
        " --seed 1"
        f" --json-out '{str(out_json)}'"
        + (" --eval-ppl" if eval_ppl else ""),
    ]
    subprocess.check_call(args, timeout=timeout_s)
    return _read_json(out_json)


def _run_bench_offline(
    *,
    conda_env: str,
    model: str,
    max_tokens: int,
    warmup: int,
    repeat: int,
    batch_size: int,
    gpu_memory_util: float,
    max_model_len: int,
    speculative_config: dict[str, Any] | None,
    out_json: Path,
    sedac_enabled: bool,
    sedac_adaptive: bool = False,
    sedac_adaptive_alpha: float = 0.1,
    sedac_adaptive_sensitivity: float = 0.5,
    sedac_latch: bool = True,
) -> dict[str, Any]:
    out_json = Path(out_json)
    spec_arg = ""
    if speculative_config is not None:
        spec_json = json.dumps(speculative_config, ensure_ascii=False).replace("'", "\\u0027")
        spec_arg = f" --speculative-config '{spec_json}'"
    sedac_val = "1" if sedac_enabled else "0"
    sedac_adaptive_val = "1" if sedac_adaptive else "0"
    sedac_latch_val = "1" if sedac_latch else "0"
    cmd = [
        "bash",
        "-lc",
        "source /home/ason/miniconda3/etc/profile.d/conda.sh"
        f" && conda activate {conda_env}"
        f" && export SEDAC_ENABLED={sedac_val}"
        f" && export SEDAC_ADAPTIVE={sedac_adaptive_val}"
        f" && export SEDAC_ADAPTIVE_ALPHA={sedac_adaptive_alpha}"
        f" && export SEDAC_ADAPTIVE_SENSITIVITY={sedac_adaptive_sensitivity}"
        f" && export SEDAC_LATCH={sedac_latch_val}"
        " && export SEDAC_PROBE_PATH=\"/mnt/g/SEDACV5.0 FAST/sedac_data/sedac_probe_layer21.pth\""
        f" && python3 '/mnt/g/SEDACV5.0 FAST/sedac_offline_bench.py'"
        f" --model '{model}'"
        f" --max-tokens {int(max_tokens)}"
        f" --warmup {int(warmup)}"
        f" --repeat {int(repeat)}"
        f" --batch-size {int(batch_size)}"
        f" --gpu-memory-utilization {float(gpu_memory_util)}"
        f" --max-model-len {int(max_model_len)}"
        f" --json-out '{str(out_json)}'"
        " --enforce-eager"
        + spec_arg,
    ]
    timeout_s = 120.0 + float(max_tokens) * float(max(1, repeat)) * 2.0
    subprocess.check_call(cmd, timeout=timeout_s)
    return _read_json(out_json)


def _maybe_build_cpp(cfg_cpp: dict[str, Any]) -> None:
    cmd = str(cfg_cpp.get("build_cmd") or "").strip()
    if cmd:
        subprocess.check_call(["bash", "-lc", cmd])
        return
    bin_path = str(cfg_cpp.get("bin_path") or "").strip()
    if not bin_path:
        return
    d = os.path.dirname(bin_path)
    if d:
        subprocess.check_call(["bash", "-lc", f"cd '{d}' && bash build.sh"])


def _run_bench_cpp(
    *,
    cfg_cpp: dict[str, Any],
    port: int,
    model: str,
    max_tokens: int,
    warmup: int,
    repeat: int,
    out_json: Path,
) -> dict[str, Any]:
    out_json = Path(out_json)
    bin_path = str(cfg_cpp.get("bin_path") or "").strip()
    if not bin_path:
        raise RuntimeError("cpp.bin_path is required")
    cmd = (
        f"'{bin_path}'"
        f" --base-url 'http://127.0.0.1:{int(port)}/v1'"
        f" --model '{model}'"
        f" --max-tokens {int(max_tokens)}"
        f" --warmup {int(warmup)}"
        f" --repeat {int(repeat)}"
        f" --json-out '{str(out_json)}'"
    )
    timeout_s = 60.0 + float(max_tokens) * float(max(1, repeat)) * 2.0
    subprocess.check_call(["bash", "-lc", cmd], timeout=timeout_s)
    return _read_json(out_json)


def _extract_tps(obj: dict[str, Any]) -> float:
    summary = obj.get("summary_a")
    if isinstance(summary, dict):
        thr = summary.get("throughput")
        if isinstance(thr, dict):
            v = thr.get("avg_completion_tps")
            if isinstance(v, (int, float)):
                return float(v)
    return 0.0


def _extract_ppl(obj: dict[str, Any]) -> float | None:
    ppl_a = obj.get("ppl_a")
    if isinstance(ppl_a, dict):
        v = ppl_a.get("ppl")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_ar_trr(obj: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    spec = obj.get("spec_decode_metrics")
    if spec is None:
        return None, None, None
    if not isinstance(spec, dict):
        return None, None, "invalid_spec_decode_metrics"
    if isinstance(spec.get("error"), str):
        return None, None, str(spec.get("error"))
    ar = spec.get("acceptance_rate")
    trr = spec.get("token_recovery_rate")
    ar_v = float(ar) if isinstance(ar, (int, float)) else None
    trr_v = float(trr) if isinstance(trr, (int, float)) else None
    return ar_v, trr_v, None


@dataclass(frozen=True)
class Exp:
    name: str
    mode: str
    speculative_config: dict[str, Any] | None
    sedac_enabled: bool = True
    sedac_adaptive: bool = False
    sedac_adaptive_alpha: float = 0.1
    sedac_adaptive_sensitivity: float = 0.5
    sedac_latch: bool = True
    sedac_threshold: float = 0.3
    sedac_calibration_steps: int = 20
    sedac_calibration_quantile: float = 0.9
    sedac_log_every: int = 0
    sedac_layer: int = 21


def _spec_key(
    spec: dict[str, Any] | None,
    sedac_enabled: bool,
    sedac_adaptive: bool,
    alpha: float,
    sensitivity: float,
    latch: bool,
    threshold: float,
    calib_steps: int,
    calib_quantile: float,
    log_every: int,
    sedac_layer: int,
) -> str:
    s = "null"
    if spec is not None:
        s = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    return (
        f"{s}|sedac={sedac_enabled}|adaptive={sedac_adaptive}|alpha={alpha}|sens={sensitivity}"
        f"|latch={latch}|thr={threshold}|calib_steps={calib_steps}|calib_q={calib_quantile}"
        f"|log_every={log_every}|layer={sedac_layer}"
    )


def main() -> int:
    no_proxy = "localhost,127.0.0.1,0.0.0.0,::1"
    prev_no_proxy = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    if prev_no_proxy.strip():
        no_proxy = no_proxy + "," + prev_no_proxy
    os.environ["no_proxy"] = no_proxy
    os.environ["NO_PROXY"] = no_proxy
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/test_matrix.json")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--timeout-ready-s", type=float, default=300.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = _read_json(Path(str(args.config)))
    base_model = str(cfg.get("base_model") or "")
    conda_env = str(cfg.get("conda_env") or "sedac_dev")
    online = cfg.get("online") if isinstance(cfg.get("online"), dict) else {}
    bench = cfg.get("bench") if isinstance(cfg.get("bench"), dict) else {}
    offline = cfg.get("offline") if isinstance(cfg.get("offline"), dict) else {}
    cfg_cpp = cfg.get("cpp") if isinstance(cfg.get("cpp"), dict) else {}

    out_dir = Path(str(args.out_dir) if str(args.out_dir).strip() else (Path("sedac_reports") / _now_tag()))
    out_dir.mkdir(parents=True, exist_ok=True)

    exps_raw = cfg.get("experiments")
    if not isinstance(exps_raw, list) or not exps_raw:
        raise RuntimeError("experiments must be a non-empty list")
    exps: list[Exp] = []
    for e in exps_raw:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        mode = str(e.get("mode") or "").strip()
        spec = e.get("speculative_config")
        spec_dict = spec if isinstance(spec, dict) else None
        sedac_enabled = bool(e.get("sedac_enabled", True))
        sedac_adaptive = bool(e.get("sedac_adaptive", False))
        sedac_adaptive_alpha = float(e.get("sedac_adaptive_alpha", 0.1))
        sedac_adaptive_sensitivity = float(e.get("sedac_adaptive_sensitivity", 0.5))
        sedac_latch = bool(e.get("sedac_latch", True))
        sedac_threshold = float(e.get("sedac_threshold", 0.3))
        sedac_calibration_steps = int(e.get("sedac_calibration_steps", 20))
        sedac_calibration_quantile = float(e.get("sedac_calibration_quantile", 0.9))
        sedac_log_every = int(e.get("sedac_log_every", 0))
        sedac_layer = int(e.get("sedac_layer", 21))
        if name and mode:
            exps.append(Exp(
                name=name,
                mode=mode,
                speculative_config=spec_dict,
                sedac_enabled=sedac_enabled,
                sedac_adaptive=sedac_adaptive,
                sedac_adaptive_alpha=sedac_adaptive_alpha,
                sedac_adaptive_sensitivity=sedac_adaptive_sensitivity,
                sedac_latch=sedac_latch,
                sedac_threshold=sedac_threshold,
                sedac_calibration_steps=sedac_calibration_steps,
                sedac_calibration_quantile=sedac_calibration_quantile,
                sedac_log_every=sedac_log_every,
                sedac_layer=sedac_layer,
            ))

    results: dict[str, dict[str, Any]] = {}
    baseline_tps: dict[str, float] = {}
    baseline_ppl: dict[str, float] = {}

    if bool(cfg_cpp.get("enabled", False)):
        _maybe_build_cpp(cfg_cpp)

    port_base = int(online.get("port_base", 8000))
    host = str(online.get("host") or "0.0.0.0")
    max_model_len = int(online.get("max_model_len", 1024))
    gpu_mem_util = float(online.get("gpu_memory_utilization", 0.6))

    bench_max_tokens = int(bench.get("max_tokens", 128))
    bench_warmup = int(bench.get("warmup", 1))
    bench_repeat = int(bench.get("repeat", 3))
    bench_conc = int(bench.get("concurrency", 1))
    bench_eval_ppl = bool(bench.get("eval_ppl", True))

    off_max_tokens = int(offline.get("max_tokens", 128))
    off_warmup = int(offline.get("warmup", 1))
    off_repeat = int(offline.get("repeat", 3))
    off_batch_size = int(offline.get("batch_size", 1))
    off_gpu_mem_util = float(offline.get("gpu_memory_utilization", 0.6))
    off_max_model_len = int(offline.get("max_model_len", 1024))

    offline_exps = [e for e in exps if e.mode == "offline"]
    online_exps = [e for e in exps if e.mode in ("online", "cpp_client")]

    run_idx = 0
    for exp in offline_exps:
        exp_out = out_dir / f"{run_idx:02d}_{exp.name}.json"
        run_idx += 1
        if bool(args.verbose):
            print(f"[suite] offline start name={exp.name}", flush=True)
        obj = _run_bench_offline(
            conda_env=conda_env,
            model=base_model,
            max_tokens=off_max_tokens,
            warmup=off_warmup,
            repeat=off_repeat,
            batch_size=off_batch_size,
            gpu_memory_util=off_gpu_mem_util,
            max_model_len=off_max_model_len,
            speculative_config=exp.speculative_config,
            out_json=exp_out,
            sedac_enabled=exp.sedac_enabled,
            sedac_adaptive=exp.sedac_adaptive,
            sedac_adaptive_alpha=exp.sedac_adaptive_alpha,
            sedac_adaptive_sensitivity=exp.sedac_adaptive_sensitivity,
            sedac_latch=exp.sedac_latch,
        )
        if bool(args.verbose):
            print(f"[suite] offline done  name={exp.name}", flush=True)
        tps = _extract_tps(obj)
        ppl = _extract_ppl(obj)
        ar, trr, spec_err = _extract_ar_trr(obj)
        results[exp.name] = {
            "mode": exp.mode,
            "speculative_config": exp.speculative_config,
            "tps": tps,
            "ppl": ppl,
            "acceptance_rate": ar,
            "token_recovery_rate": trr,
            "spec_decode_error": spec_err,
            "json_path": str(exp_out),
        }
        if exp.speculative_config is None:
            baseline_tps[exp.mode] = tps
            if isinstance(ppl, (int, float)) and float(ppl) > 0:
                baseline_ppl[exp.mode] = float(ppl)

    groups: dict[str, list[Exp]] = {}
    for exp in online_exps:
        groups.setdefault(
            _spec_key(
                exp.speculative_config,
                exp.sedac_enabled,
                exp.sedac_adaptive,
                exp.sedac_adaptive_alpha,
                exp.sedac_adaptive_sensitivity,
                exp.sedac_latch,
                exp.sedac_threshold,
                exp.sedac_calibration_steps,
                exp.sedac_calibration_quantile,
                exp.sedac_log_every,
                exp.sedac_layer,
            ),
            [],
        ).append(exp)

    for g_idx, (k, group_exps) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        port = port_base + g_idx
        if bool(args.verbose):
            names = ",".join([e.name for e in group_exps])
            print(f"[suite] server start port={port} exps={names}", flush=True)
        exp0 = group_exps[0] if group_exps else None
        proc = _start_server(
            port=port,
            host=host,
            conda_env=conda_env,
            model=base_model,
            max_model_len=max_model_len,
            gpu_memory_util=gpu_mem_util,
            speculative_config=exp0.speculative_config if exp0 else None,
            sedac_enabled=exp0.sedac_enabled if exp0 else True,
            sedac_adaptive=exp0.sedac_adaptive if exp0 else False,
            sedac_adaptive_alpha=exp0.sedac_adaptive_alpha if exp0 else 0.1,
            sedac_adaptive_sensitivity=exp0.sedac_adaptive_sensitivity if exp0 else 0.5,
            sedac_latch=exp0.sedac_latch if exp0 else True,
            sedac_threshold=exp0.sedac_threshold if exp0 else 0.3,
            sedac_calibration_steps=exp0.sedac_calibration_steps if exp0 else 20,
            sedac_calibration_quantile=exp0.sedac_calibration_quantile if exp0 else 0.9,
            sedac_log_every=exp0.sedac_log_every if exp0 else 0,
            sedac_layer=exp0.sedac_layer if exp0 else 21,
        )
        try:
            _wait_server_ready(proc, timeout_s=float(args.timeout_ready_s), verbose=bool(args.verbose))
            _wait_http_ready(port=port, timeout_s=float(args.timeout_ready_s))
            if proc.poll() is not None:
                raise RuntimeError("vLLM server exited after reporting ready")
            for exp in group_exps:
                exp_out = out_dir / f"{run_idx:02d}_{exp.name}.json"
                run_idx += 1
                if exp.mode == "cpp_client":
                    obj = _run_bench_cpp(
                        cfg_cpp=cfg_cpp,
                        port=port,
                        model=base_model,
                        max_tokens=bench_max_tokens,
                        warmup=bench_warmup,
                        repeat=bench_repeat,
                        out_json=exp_out,
                    )
                else:
                    obj = _run_bench_online(
                        port=port,
                        model=base_model,
                        max_tokens=bench_max_tokens,
                        warmup=bench_warmup,
                        repeat=bench_repeat,
                        concurrency=bench_conc,
                        eval_ppl=bench_eval_ppl,
                        out_json=exp_out,
                    )
                tps = _extract_tps(obj)
                ppl = _extract_ppl(obj)
                ar, trr, spec_err = _extract_ar_trr(obj)
                results[exp.name] = {
                    "mode": exp.mode,
                    "speculative_config": exp.speculative_config,
                    "sedac_enabled": exp.sedac_enabled,
                    "sedac_adaptive": exp.sedac_adaptive,
                    "sedac_adaptive_alpha": exp.sedac_adaptive_alpha,
                    "sedac_adaptive_sensitivity": exp.sedac_adaptive_sensitivity,
                    "sedac_latch": exp.sedac_latch,
                    "sedac_threshold": exp.sedac_threshold,
                    "sedac_calibration_steps": exp.sedac_calibration_steps,
                    "sedac_calibration_quantile": exp.sedac_calibration_quantile,
                    "sedac_log_every": exp.sedac_log_every,
                    "sedac_layer": exp.sedac_layer,
                    "tps": tps,
                    "ppl": ppl,
                    "acceptance_rate": ar,
                    "token_recovery_rate": trr,
                    "spec_decode_error": spec_err,
                    "json_path": str(exp_out),
                }
                if exp.speculative_config is None and not exp.sedac_enabled:
                    baseline_tps[exp.mode] = tps
                    if isinstance(ppl, (int, float)) and float(ppl) > 0:
                        baseline_ppl[exp.mode] = float(ppl)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        if bool(args.verbose):
            print(f"[suite] server done  port={port}", flush=True)

    for name, r in results.items():
        mode = str(r.get("mode") or "")
        base = float(baseline_tps.get(mode, 0.0))
        tps = float(r.get("tps") or 0.0)
        r["speedup_over_baseline"] = (tps / base) if base > 0 else None
        ppl = r.get("ppl")
        base_ppl = baseline_ppl.get(mode)
        if isinstance(ppl, (int, float)) and isinstance(base_ppl, (int, float)) and float(base_ppl) > 0:
            r["ppl_ratio_over_baseline"] = float(ppl) / float(base_ppl)
        else:
            r["ppl_ratio_over_baseline"] = None

    report: JsonDict = {
        "config": {
            "base_model": base_model,
            "conda_env": conda_env,
            "online": online,
            "bench": bench,
            "offline": offline,
            "cpp": cfg_cpp,
            "thresholds": cfg.get("thresholds"),
        },
        "results": results,
    }
    _write_json(out_dir / "report.json", report)

    lines = [
        "# SEDAC Test Suite Report",
        "",
        f"Output: {out_dir}",
        "",
        "| name | mode | tps | speedup | ppl | ppl_ratio | AR | TRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in sorted(results.items(), key=lambda kv: str(kv[0])):
        tps = float(r.get("tps") or 0.0)
        sp = r.get("speedup_over_baseline")
        sp_s = f"{float(sp):.3f}" if isinstance(sp, (int, float)) else "-"
        ppl = r.get("ppl")
        ppl_s = f"{float(ppl):.3f}" if isinstance(ppl, (int, float)) else "-"
        pplr = r.get("ppl_ratio_over_baseline")
        pplr_s = f"{float(pplr):.4f}" if isinstance(pplr, (int, float)) else "-"
        ar = r.get("acceptance_rate")
        trr = r.get("token_recovery_rate")
        ar_s = f"{float(ar):.3f}" if isinstance(ar, (int, float)) else "-"
        trr_s = f"{float(trr):.3f}" if isinstance(trr, (int, float)) else "-"
        lines.append(f"| {name} | {r.get('mode')} | {tps:.3f} | {sp_s} | {ppl_s} | {pplr_s} | {ar_s} | {trr_s} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

