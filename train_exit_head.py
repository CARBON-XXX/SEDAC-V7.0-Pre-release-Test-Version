
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4")
ap.add_argument("--exit-layer", type=int, default=20)
ap.add_argument("--seq-len", type=int, default=512)
ap.add_argument("--max-steps", type=int, default=200)
ap.add_argument("--batch-size", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--grad-clip", type=float, default=1.0)
ap.add_argument("--save-path", type=str, default="sedac_data/exit_head_layer21.pth")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--head-type", choices=["linear", "mlp"], default="mlp")
ap.add_argument("--d-inner", type=int, default=1024)
ap.add_argument("--temperature", type=float, default=2.0)
ap.add_argument("--kd-alpha", type=float, default=1.0)
ap.add_argument("--hard-alpha", type=float, default=0.0)
args = ap.parse_args()

MODEL_ID = str(args.model_id)
EXIT_LAYER = int(args.exit_layer)
BATCH_SIZE = int(args.batch_size)
GRAD_ACCUM = int(args.grad_accum)
LEARNING_RATE = float(args.lr)
MAX_STEPS = int(args.max_steps)
SEQ_LEN = int(args.seq_len)
SAVE_PATH = str(args.save_path)
GRAD_CLIP = float(args.grad_clip)
HEAD_TYPE = str(args.head_type)
D_INNER = int(args.d_inner)
TEMPERATURE = float(args.temperature)
KD_ALPHA = float(args.kd_alpha)
HARD_ALPHA = float(args.hard_alpha)

# --- Setup ---
torch.manual_seed(int(args.seed))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.eval()
# Freeze model
for param in model.parameters():
    param.requires_grad = False

print("Model loaded.")

# --- Prepare Exit Head ---
# We want the Exit Head to work on top of the Final Norm.
# Draft Model Structure: Layers 0..20 -> Final Norm -> Exit Head
# So Exit Head should be initialized from the original LM Head.
print("Initializing Exit Head...")
hidden_size = model.config.hidden_size
vocab_size = model.config.vocab_size

if D_INNER <= 0:
    D_INNER = hidden_size


class SedacConnector(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, d_inner: int, head_type: str):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.use_mlp = head_type == "mlp"
        if self.use_mlp:
            self.fc1 = nn.Linear(hidden_size, d_inner, bias=True)
            self.fc2 = nn.Linear(d_inner, hidden_size, bias=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        if self.use_mlp:
            x = self.fc2(F.silu(self.fc1(x)))
        return self.lm_head(x)


exit_head = SedacConnector(hidden_size=hidden_size, vocab_size=vocab_size, d_inner=D_INNER, head_type=HEAD_TYPE)
with torch.no_grad():
    exit_head.lm_head.weight.copy_(model.lm_head.weight)
exit_head.half().to(device).train()

optimizer = torch.optim.AdamW(exit_head.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=int(tokenizer.pad_token_id), reduction="mean")

# --- Data ---
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.filter(lambda x: len(x['text']) > 100)

# Hook for Layer 20 Output
layer_outputs = []
def get_activation(name):
    def hook(model, input, output):
        # output is tuple (hidden_states, ...)
        layer_outputs.append(output[0])
    return hook

# Hook Layer 20 (index 20)
model.model.layers[EXIT_LAYER].register_forward_hook(get_activation("exit_layer"))

# --- Training Loop ---
print("Starting training...")
step = 0
optimizer.zero_grad()

# Infinite iterator
def data_gen():
    while True:
        for x in dataset:
            yield x['text']

data_iter = data_gen()

progress_bar = tqdm(range(MAX_STEPS))
for _ in progress_bar:
    layer_outputs.clear() # Clear previous hooks
    batch_texts = []
    for _ in range(BATCH_SIZE):
        batch_texts.append(next(data_iter))
    
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        max_length=SEQ_LEN,
        truncation=True,
        padding=True,
    ).to(device)
    
    # Forward Pass (Main Model)
    # We only need it to run up to EXIT_LAYER, but transformers doesn't support partial run easily without modification.
    # So we run full, but we detach gradients anyway.
    with torch.no_grad():
        out = model(**inputs)
        
    # Get Layer 20 output
    if len(layer_outputs) == 0:
        raise RuntimeError("Hook not called!")
    elif len(layer_outputs) == 1:
        hidden_states = layer_outputs[0]
    else:
        # Hook called multiple times (e.g. if model splits batch)
        hidden_states = torch.cat(layer_outputs, dim=0)
        
    with torch.no_grad():
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        normed_hidden_states = model.model.norm(hidden_states)
        normed_hidden_states = torch.nan_to_num(normed_hidden_states, nan=0.0, posinf=65504.0, neginf=-65504.0)
        
    # Forward Pass (Exit Head)
    logits = exit_head(normed_hidden_states)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=65504.0, neginf=-65504.0)

    with torch.no_grad():
        teacher_logits = torch.nan_to_num(out.logits, nan=0.0, posinf=65504.0, neginf=-65504.0)
        if teacher_logits.ndim == 2:
            teacher_logits = teacher_logits.unsqueeze(0)
        teacher_ids = torch.argmax(teacher_logits, dim=-1)

    shift_student = logits[:, :-1, :].contiguous().float().clamp(-30.0, 30.0)
    shift_teacher_ids = teacher_ids[:, 1:].contiguous()

    loss_hard = criterion(shift_student.view(-1, vocab_size), shift_teacher_ids.view(-1))
    loss = HARD_ALPHA * loss_hard

    if KD_ALPHA > 0.0:
        t = max(1e-3, float(TEMPERATURE))
        with torch.no_grad():
            shift_teacher_logits = teacher_logits[:, 1:, :].contiguous().float().clamp(-30.0, 30.0)
            teacher_probs = torch.softmax(shift_teacher_logits / t, dim=-1)
        student_log_probs = torch.log_softmax(shift_student / t, dim=-1)
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t * t)
        loss = loss + KD_ALPHA * loss_kd
    loss = loss / GRAD_ACCUM
    
    # Check for NaN loss
    if torch.isnan(loss):
        print(f"NaN loss detected at step {step}!")
        optimizer.zero_grad()
        continue

    loss.backward()
    
    if (step + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(exit_head.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.set_description(f"Loss: {loss.item() * GRAD_ACCUM:.4f}")
        
    step += 1
    if step >= MAX_STEPS:
        break

# --- Save ---
print(f"Saving Exit Head to {SAVE_PATH}...")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
sd = exit_head.state_dict()
prefixed = {f"sedac_connector.{k}": v.detach().cpu() for k, v in sd.items()}
torch.save(prefixed, SAVE_PATH)
print("Done.")
