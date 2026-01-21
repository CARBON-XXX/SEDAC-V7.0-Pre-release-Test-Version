"""
SEDAC Universal Full-Layer Data Collector (Architecture-Agnostic)
==================================================================

Collects hidden states from all layers of any Transformer model for full-layer monitoring validation.
Auto-detects model layer structure, supports Qwen, Llama, Mistral, GPT, and any architecture.

Usage:
    python sedac_collect_data.py --model Qwen/Qwen2.5-3B-Instruct
    python sedac_collect_data.py --model meta-llama/Llama-2-7b-hf
    python sedac_collect_data.py --model mistralai/Mistral-7B-v0.1
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sedac.core.universal_monitor import auto_detect_layers


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits (architecture-agnostic)."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def collect_hidden_states_universal(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 512,
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]], int]:
    """
    Collect hidden states from all layers (architecture-agnostic).
    
    Auto-detects model layer structure, supports any Transformer.
    """
    model.eval()
    
    layers = auto_detect_layers(model)
    num_layers = len(layers)
    print(f"  Auto-detected {num_layers} transformer layers")
    
    hidden_states = {i: [] for i in range(num_layers)}
    entropies = {i: [] for i in range(num_layers)}
    layer_outputs = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            layer_outputs[layer_idx] = h.detach()
        return hook
    
    hooks = []
    for i, layer in enumerate(layers):
        hook = layer.register_forward_hook(make_hook(i))
        hooks.append(hook)
    
    try:
        for text in tqdm(texts, desc="Collecting"):
            layer_outputs.clear()
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False,
            ).to(device)
            
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                token_entropy = compute_entropy(logits)
            
            for layer_idx in range(num_layers):
                if layer_idx in layer_outputs:
                    h = layer_outputs[layer_idx]
                    h_flat = h.squeeze(0) if h.dim() == 3 else h
                    hidden_states[layer_idx].append(h_flat.cpu())
                    
                    e_flat = token_entropy.squeeze(0)
                    entropies[layer_idx].append(e_flat.cpu())
    
    finally:
        for hook in hooks:
            hook.remove()
    
    return hidden_states, entropies, num_layers


def main():
    parser = argparse.ArgumentParser(description="Universal all-layer data collector")
    parser.add_argument("--model", type=str, required=True, help="Any HuggingFace model")
    parser.add_argument("--output", type=str, default="sedac_data_full")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In machine learning, neural networks learn patterns from data.",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "What is 2 + 2? The answer is 4.",
        "Explain gravity: Objects attract each other with a force.",
    ] * (args.num_samples // 5 + 1)
    sample_texts = sample_texts[:args.num_samples]
    
    print(f"Collecting from {len(sample_texts)} samples...")
    hidden_states, entropies, num_layers = collect_hidden_states_universal(
        model, tokenizer, sample_texts, args.device
    )
    
    print("\nSaving data...")
    for layer_idx in range(num_layers):
        if hidden_states[layer_idx]:
            h = torch.cat(hidden_states[layer_idx], dim=0)
            e = torch.cat(entropies[layer_idx], dim=0)
            
            torch.save(h, os.path.join(args.output, f"hidden_states_layer{layer_idx}.pt"))
            torch.save(e, os.path.join(args.output, f"entropies_layer{layer_idx}.pt"))
    
    print(f"\n✅ Saved to {args.output}/")
    print(f"   Layers: {num_layers}")
    print(f"   Model: {args.model} (architecture auto-detected)")


if __name__ == "__main__":
    main()
