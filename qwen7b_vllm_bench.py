from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import exp
from typing import Iterable, Optional, Sequence, Tuple, cast


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
class RequestSpec:
    name: str
    messages: list[dict[str, str]]
    max_tokens: int


@dataclass(frozen=True)
class RunResult:
    tag: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    text: str

    @property
    def completion_tps(self) -> float:
        if self.latency_s <= 0:
            return 0.0
        return float(self.completion_tokens) / self.latency_s


def _http_json(
    url: str,
    payload: JsonDict,
    timeout_s: float,
    api_key: Optional[str],
) -> JsonDict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e}") from e
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Bad response: expected JSON object, got: {type(parsed)}")
    return cast(JsonDict, parsed)


def _http_get_json(url: str, timeout_s: float, api_key: Optional[str]) -> JsonDict:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url=url, headers=headers, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e}") from e
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Bad response: expected JSON object, got: {type(parsed)}")
    return cast(JsonDict, parsed)


def list_models(base_url: str, timeout_s: float, api_key: Optional[str]) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    resp = _http_get_json(url=url, timeout_s=timeout_s, api_key=api_key)
    data = resp.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"/models bad response: {resp}")
    out: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            out.append(str(item["id"]))
    return out


@dataclass(frozen=True)
class PplSample:
    name: str
    prompt: str
    target: str


def default_ppl_samples() -> list[PplSample]:
    return [
        PplSample(name="capital", prompt="The capital of France is", target=" Paris."),
        PplSample(name="math", prompt="123 * 456 =", target=" 56088."),
        PplSample(name="json", prompt="Return raw JSON with keys a=1,b=2:", target=' {"a": 1, "b": 2}'),
        PplSample(name="memory", prompt="Remember token K9Z7Q. Reply OK.", target=" OK."),
    ]


def completion_logprobs(
    base_url: str,
    model: str,
    text: str,
    timeout_s: float,
    api_key: Optional[str],
    seed: int | None,
) -> tuple[list[float | None], list[int]]:
    url = base_url.rstrip("/") + "/completions"
    payload: JsonDict = {
        "model": model,
        "prompt": text,
        "max_tokens": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "echo": True,
        "logprobs": 1,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    resp = _http_json(url=url, payload=payload, timeout_s=timeout_s, api_key=api_key)
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"Bad response: missing choices: {resp}")
    lp = choices[0].get("logprobs")
    if not isinstance(lp, dict):
        raise RuntimeError(f"Bad response: missing logprobs: {resp}")
    token_logprobs = lp.get("token_logprobs")
    text_offset = lp.get("text_offset")
    if not isinstance(token_logprobs, list) or not isinstance(text_offset, list):
        raise RuntimeError(f"Bad response: logprobs shape invalid: {resp}")
    out_lp: list[float | None] = []
    out_off: list[int] = []
    for x, off in zip(token_logprobs, text_offset):
        out_lp.append(float(x) if isinstance(x, (int, float)) else None)
        out_off.append(int(off) if isinstance(off, int) else 0)
    return out_lp, out_off


def eval_ppl(
    base_url: str,
    model: str,
    samples: Sequence[PplSample],
    timeout_s: float,
    api_key: Optional[str],
    seed: int | None,
) -> JsonDict:
    total_nll = 0.0
    total_tokens = 0
    per_sample: list[JsonDict] = []
    for s in samples:
        text = s.prompt + s.target
        lps, offs = completion_logprobs(
            base_url=base_url,
            model=model,
            text=text,
            timeout_s=timeout_s,
            api_key=api_key,
            seed=seed,
        )
        boundary = len(s.prompt)
        nll = 0.0
        n = 0
        for lp, off in zip(lps, offs):
            if off < boundary:
                continue
            if lp is None:
                continue
            nll += -float(lp)
            n += 1
        ppl = exp(nll / n) if n > 0 else float("inf")
        per_sample.append({"name": s.name, "tokens": n, "nll": nll, "ppl": ppl})
        total_nll += nll
        total_tokens += n
    ppl_total = exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    return {"ppl": ppl_total, "tokens": total_tokens, "samples": per_sample}


def chat_once(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    api_key: Optional[str],
    ignore_eos: bool,
    seed: int | None,
) -> RunResult:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: JsonDict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "ignore_eos": bool(ignore_eos),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    t0 = time.perf_counter()
    resp = _http_json(url=url, payload=payload, timeout_s=timeout_s, api_key=api_key)
    t1 = time.perf_counter()

    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Bad response: missing choices: {resp}")

    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise RuntimeError(f"Bad response: choices[0].message not dict: {resp}")

    text = msg.get("content")
    if not isinstance(text, str):
        raise RuntimeError(f"Bad response: message.content not str: {resp}")

    usage = resp.get("usage")
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    else:
        total_tokens = prompt_tokens + completion_tokens

    return RunResult(
        tag="",
        latency_s=(t1 - t0),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        text=text,
    )


def default_requests(max_tokens: int) -> list[RequestSpec]:
    sys_msg = {"role": "system", "content": "You are a concise assistant."}
    return [
        RequestSpec(
            name="capital",
            messages=[sys_msg, {"role": "user", "content": "The capital of France is? Answer with one word."}],
            max_tokens=max_tokens,
        ),
        RequestSpec(
            name="math",
            messages=[
                sys_msg,
                {
                    "role": "user",
                    "content": "Calculate 123 * 456. Only output the final result number, no explanation.",
                },
            ],
            max_tokens=max_tokens,
        ),
        RequestSpec(
            name="json",
            messages=[
                sys_msg,
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object with keys a,b where a=1 and b=2. "
                        "Output raw JSON only (no Markdown fences, no extra text)."
                    ),
                },
            ],
            max_tokens=max_tokens,
        ),
        RequestSpec(
            name="longrange",
            messages=[
                sys_msg,
                {"role": "user", "content": "Remember this token: K9Z7Q. Reply OK."},
                {"role": "user", "content": "What token did I ask you to remember? Reply exactly."},
            ],
            max_tokens=max_tokens,
        ),
    ]


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def summarize(results: Sequence[RunResult]) -> JsonDict:
    latencies = sorted([r.latency_s for r in results])
    completion_tokens = [r.completion_tokens for r in results]
    total_completion_tokens = int(sum(completion_tokens))
    total_latency = float(sum(latencies))
    avg_tps = (float(total_completion_tokens) / total_latency) if total_latency > 0 else 0.0

    out: JsonDict = {
        "runs": len(results),
        "latency_s": {
            "min": float(min(latencies)) if latencies else 0.0,
            "max": float(max(latencies)) if latencies else 0.0,
            "mean": float(statistics.mean(latencies)) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p90": _percentile(latencies, 90),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
        },
        "tokens": {
            "completion_total": total_completion_tokens,
            "completion_mean": float(statistics.mean(completion_tokens)) if completion_tokens else 0.0,
        },
        "throughput": {"avg_completion_tps": avg_tps},
    }
    return out


def _iter_jobs(
    requests: Sequence[RequestSpec],
    repeat: int,
) -> Iterable[Tuple[str, RequestSpec]]:
    for req in requests:
        for i in range(repeat):
            yield (f"{req.name}#{i}", req)


def run_bench(
    base_url: str,
    model: str,
    requests: Sequence[RequestSpec],
    warmup: int,
    repeat: int,
    concurrency: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    api_key: Optional[str],
    jsonl_path: Optional[str],
    ignore_eos: bool,
    seed: int | None,
) -> list[RunResult]:
    if warmup > 0:
        for _ in range(warmup):
            _ = chat_once(
                base_url=base_url,
                model=model,
                messages=requests[0].messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=requests[0].max_tokens,
                timeout_s=timeout_s,
                api_key=api_key,
                ignore_eos=ignore_eos,
                seed=seed,
            )

    def _one(tag: str, req: RequestSpec) -> RunResult:
        r = chat_once(
            base_url=base_url,
            model=model,
            messages=req.messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=req.max_tokens,
            timeout_s=timeout_s,
            api_key=api_key,
            ignore_eos=ignore_eos,
            seed=seed,
        )
        return RunResult(
            tag=tag,
            latency_s=r.latency_s,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            total_tokens=r.total_tokens,
            text=r.text,
        )

    results: list[RunResult] = []
    writer = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures: list[Future[RunResult]] = []
            for tag, req in _iter_jobs(requests=requests, repeat=repeat):
                futures.append(ex.submit(_one, tag, req))
            for fut in futures:
                r = fut.result()
                results.append(r)
                if writer:
                    writer.write(
                        json.dumps(
                            {
                                "tag": r.tag,
                                "latency_s": r.latency_s,
                                "prompt_tokens": r.prompt_tokens,
                                "completion_tokens": r.completion_tokens,
                                "total_tokens": r.total_tokens,
                                "text": r.text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if writer:
            writer.flush()
            writer.close()

    return results


def _compare_by_tag(results_a: Sequence[RunResult], results_b: Sequence[RunResult]) -> JsonDict:
    map_a = {r.tag: r for r in results_a}
    map_b = {r.tag: r for r in results_b}
    tags = sorted(set(map_a.keys()) | set(map_b.keys()))
    mismatches: list[JsonDict] = []
    missing_a: list[str] = []
    missing_b: list[str] = []
    for tag in tags:
        ra = map_a.get(tag)
        rb = map_b.get(tag)
        if ra is None:
            missing_a.append(tag)
            continue
        if rb is None:
            missing_b.append(tag)
            continue
        if ra.text != rb.text:
            mismatches.append(
                {
                    "tag": tag,
                    "a_text": ra.text,
                    "b_text": rb.text,
                    "a_latency_s": ra.latency_s,
                    "b_latency_s": rb.latency_s,
                }
            )
    return {
        "runs_a": len(results_a),
        "runs_b": len(results_b),
        "missing_in_a": missing_a,
        "missing_in_b": missing_b,
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", type=str, default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--base-url-b", type=str, default=os.environ.get("VLLM_BASE_URL_B", ""))
    ap.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"),
    )
    ap.add_argument("--api-key", type=str, default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--ignore-eos", type=int, default=int(os.environ.get("VLLM_IGNORE_EOS", "1")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("VLLM_SEED", "1")))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--jsonl", type=str, default="")
    ap.add_argument("--print-text", action="store_true")
    ap.add_argument("--eval-ppl", action="store_true")
    args = ap.parse_args()

    api_key: Optional[str] = args.api_key if args.api_key else None
    available_a = list_models(base_url=args.base_url, timeout_s=float(args.timeout_s), api_key=api_key)
    if available_a and args.model not in available_a:
        print("server_models_a=", available_a)
    base_url_b = str(args.base_url_b).strip()
    if base_url_b:
        available_b = list_models(base_url=base_url_b, timeout_s=float(args.timeout_s), api_key=api_key)
        if available_b and args.model not in available_b:
            print("server_models_b=", available_b)

    if bool(args.eval_ppl):
        samples = default_ppl_samples()
        seed: int | None = int(args.seed) if int(args.seed) >= 0 else None
        ppl_a = eval_ppl(
            base_url=args.base_url,
            model=args.model,
            samples=samples,
            timeout_s=float(args.timeout_s),
            api_key=api_key,
            seed=seed,
        )
        print("=== ppl_a ===")
        print(json.dumps(ppl_a, ensure_ascii=False, indent=2))
        if base_url_b:
            ppl_b = eval_ppl(
                base_url=base_url_b,
                model=args.model,
                samples=samples,
                timeout_s=float(args.timeout_s),
                api_key=api_key,
                seed=seed,
            )
            print("=== ppl_b ===")
            print(json.dumps(ppl_b, ensure_ascii=False, indent=2))
            a = float(ppl_a.get("ppl", float("inf")) or float("inf"))
            b = float(ppl_b.get("ppl", float("inf")) or float("inf"))
            if a != 0.0 and a != float("inf") and b != float("inf"):
                print("=== ppl_ratio_b_over_a ===")
                print(json.dumps({"ratio": b / a}, ensure_ascii=False, indent=2))

    reqs = default_requests(max_tokens=int(args.max_tokens))
    ignore_eos = bool(int(args.ignore_eos))
    seed = int(args.seed) if int(args.seed) >= 0 else None
    results_a = run_bench(
        base_url=args.base_url,
        model=args.model,
        requests=reqs,
        warmup=int(args.warmup),
        repeat=int(args.repeat),
        concurrency=int(args.concurrency),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        timeout_s=float(args.timeout_s),
        api_key=api_key,
        jsonl_path=(args.jsonl if args.jsonl else None),
        ignore_eos=ignore_eos,
        seed=seed,
    )

    print("=== per-run ===")
    for r in results_a:
        msg = f"{r.tag:12s} latency_s={r.latency_s:.3f} out_tok={r.completion_tokens:4d} tps={r.completion_tps:8.2f}"
        if args.print_text:
            msg += f" text={r.text!r}"
        print(msg)

    print("=== summary ===")
    print(json.dumps(summarize(results_a), ensure_ascii=False, indent=2))

    if base_url_b:
        results_b = run_bench(
            base_url=base_url_b,
            model=args.model,
            requests=reqs,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            concurrency=int(args.concurrency),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            timeout_s=float(args.timeout_s),
            api_key=api_key,
            jsonl_path=None,
            ignore_eos=ignore_eos,
            seed=seed,
        )
        print("=== summary_b ===")
        print(json.dumps(summarize(results_b), ensure_ascii=False, indent=2))

        diff = _compare_by_tag(results_a=results_a, results_b=results_b)
        print("=== compare ===")
        print(json.dumps(diff, ensure_ascii=False, indent=2))
        if int(diff.get("mismatch_count", 0) or 0) != 0:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

