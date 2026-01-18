import argparse
import os
import sys


def _resolve_target_path(arg_target_path: str | None) -> str:
    if arg_target_path:
        return arg_target_path
    env_target = os.environ.get("SEDAC_VLLM_API_SERVER_PATH")
    if env_target:
        return env_target
    try:
        import vllm.entrypoints.openai.api_server as api_server  # type: ignore[import-not-found]

        return str(api_server.__file__)
    except Exception:
        return "/home/ason/miniconda3/envs/sedac_dev/lib/python3.10/site-packages/vllm/entrypoints/openai/api_server.py"


PATCH_MARKER_V1 = "SEDAC_OPENAI_METRICS_PATCH = 1"
PATCH_MARKER_V2 = "SEDAC_OPENAI_METRICS_PATCH_V2 = 1"


INJECT_HELPER_V2 = """
SEDAC_OPENAI_METRICS_PATCH_V2 = 1

def _register_sedac_metrics_endpoint(app: FastAPI) -> None:
    @app.get("/metrics")
    async def _metrics() -> Response:
        import os
        from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

        mp_dir = str(os.environ.get("PROMETHEUS_MULTIPROC_DIR") or "").strip()
        if mp_dir:
            from prometheus_client import multiprocess

            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            data = generate_latest(registry)
        else:
            import prometheus_client

            data = generate_latest(prometheus_client.REGISTRY)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-path", default=None)
    args = ap.parse_args()

    target_path = _resolve_target_path(args.target_path)
    print(f"Target api_server.py: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        src = f.read()

    if PATCH_MARKER_V2 in src:
        print("Already patched (v2).")
        return 0

    anchor = "def build_app(args: Namespace) -> FastAPI:"
    if anchor not in src:
        raise RuntimeError("Cannot find build_app definition anchor")

    if PATCH_MARKER_V1 in src:
        i = src.find(PATCH_MARKER_V1)
        j = src.find(anchor, i)
        if i < 0 or j < 0:
            raise RuntimeError("Cannot locate v1 patch region to upgrade")
        src = src[:i] + INJECT_HELPER_V2 + "\n\n" + src[j:]
    else:
        if INJECT_HELPER_V2.strip() not in src:
            src = src.replace(anchor, INJECT_HELPER_V2 + "\n\n" + anchor, 1)

    call_anchor = "app.state.args = args"
    if call_anchor not in src:
        raise RuntimeError("Cannot find app.state.args assignment anchor")
    if "_register_sedac_metrics_endpoint(app)" not in src:
        src = src.replace(call_anchor, call_anchor + "\n    _register_sedac_metrics_endpoint(app)", 1)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(src)
    print("Patched /metrics endpoint into OpenAI server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
