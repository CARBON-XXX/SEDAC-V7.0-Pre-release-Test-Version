from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any


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


@dataclass(frozen=True)
class RunResult:
    tag: str
    latency_s: float
    completion_tokens: int

    @property
    def completion_tps(self) -> float:
        if self.latency_s <= 0:
            return 0.0
        return float(self.completion_tokens) / self.latency_s


def default_prompts() -> list[str]:
    return [
        "The capital of France is",
        "123 * 456 =",
        "Return raw JSON with keys a=1,b=2:",
        "Write a short Python function that parses a JSON string and returns a dict.",
    ]


def _count_completion_tokens(outputs: Any) -> int:
    total = 0
    for out in outputs:
        for o in getattr(out, "outputs", []):
            tok_ids = getattr(o, "token_ids", None)
            if isinstance(tok_ids, list):
                total += len(tok_ids)
            else:
                tok_count = getattr(o, "token_count", None)
                if isinstance(tok_count, int):
                    total += tok_count
    return int(total)


def _summarize(results: list[RunResult]) -> JsonDict:
    latencies = [r.latency_s for r in results if r.latency_s > 0]
    tps = [r.completion_tps for r in results if r.completion_tps > 0]
    out_tok = [r.completion_tokens for r in results if r.completion_tokens >= 0]
    return {
        "runs": len(results),
        "latency_s": {
            "p50": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "throughput": {
            "avg_completion_tps": (statistics.mean(tps) if tps else 0.0),
        },
        "completion_tokens": {
            "sum": int(sum(out_tok)),
            "avg": (statistics.mean(out_tok) if out_tok else 0.0),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--speculative-config", type=str, default="")
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    spec_cfg_for_out: dict[str, Any] | None = None
    spec_cfg_for_llm: dict[str, Any] | None = None
    if str(args.speculative_config).strip():
        spec_cfg_raw = json.loads(str(args.speculative_config))
        if not isinstance(spec_cfg_raw, dict):
            raise RuntimeError("--speculative-config must be a JSON object")
        spec_cfg_for_out = json.loads(json.dumps(spec_cfg_raw, ensure_ascii=False))
        spec_cfg_for_llm = json.loads(json.dumps(spec_cfg_raw, ensure_ascii=False))

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "dtype": str(args.dtype),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "seed": int(args.seed),
        "trust_remote_code": bool(args.trust_remote_code),
        "max_model_len": int(args.max_model_len),
        "enforce_eager": bool(args.enforce_eager),
    }
    if spec_cfg_for_llm is not None:
        llm_kwargs["speculative_config"] = spec_cfg_for_llm

    llm = LLM(model=str(args.model), **llm_kwargs)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(args.max_tokens))

    prompts = default_prompts()
    batch_size = max(1, int(args.batch_size))

    def run_once(tag: str) -> RunResult:
        batch = (prompts * ((batch_size + len(prompts) - 1) // len(prompts)))[:batch_size]
        t0 = time.perf_counter()
        outs = llm.generate(batch, sampling)
        t1 = time.perf_counter()
        comp_toks = _count_completion_tokens(outs)
        return RunResult(tag=tag, latency_s=max(1e-9, t1 - t0), completion_tokens=int(comp_toks))

    for i in range(max(0, int(args.warmup))):
        _ = run_once(tag=f"warmup{i}")

    results: list[RunResult] = []
    for i in range(max(1, int(args.repeat))):
        results.append(run_once(tag=f"run{i}"))

    summary = _summarize(results)
    out_obj: JsonDict = {
        "params": {
            "mode": "offline",
            "model": str(args.model),
            "max_tokens": int(args.max_tokens),
            "repeat": int(args.repeat),
            "warmup": int(args.warmup),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "dtype": str(args.dtype),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "max_model_len": int(args.max_model_len),
            "speculative_config": spec_cfg_for_out,
        },
        "summary_a": summary,
        "per_run": [
            {
                "tag": r.tag,
                "latency_s": r.latency_s,
                "completion_tokens": r.completion_tokens,
                "completion_tps": r.completion_tps,
            }
            for r in results
        ],
    }
    if str(args.json_out).strip():
        with open(str(args.json_out), "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(json.dumps(out_obj, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
