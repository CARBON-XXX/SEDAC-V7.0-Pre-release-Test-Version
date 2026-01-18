import sys
print("1. Start", flush=True)

try:
    import torch
    print(f"2. Torch imported: {torch.__version__}", flush=True)
    print(f"   CUDA available: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"2. Torch failed: {e}", flush=True)

try:
    import transformers
    print(f"3. Transformers imported: {transformers.__version__}", flush=True)
except Exception as e:
    print(f"3. Transformers failed: {e}", flush=True)

print("4. Done", flush=True)
