
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from tqdm import tqdm

# --- 配置 ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4", help="Model ID or path")
parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
parser.add_argument("--samples", type=int, default=500, help="Number of samples")
parser.add_argument("--save-dir", type=str, default="sedac_data", help="Output directory")
args = parser.parse_args()

MODEL_ID = args.model
DATASET_NAME = args.dataset
DATASET_CONFIG = args.dataset_config
TARGET_LAYER = 20  # 注意：Python索引从0开始，Layer 21 对应 index 20
NUM_SAMPLES = args.samples  # 采集多少条样本（根据显存可调整，500条足够训练探针）
MAX_LENGTH = 512   # 每条文本长度，设短点防爆显存
SAVE_DIR = args.save_dir

# --- 准备环境 ---
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}...")

# 1. 加载模型 (使用 HuggingFace Transformers 而不是 vLLM，方便 Hook)
print("Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 2. 加载数据
print("Loading WikiText...")
data = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
# 过滤掉太短的文本
data = data.filter(lambda x: len(x['text']) > 100)

# 3. 定义 Hook (钩子)
# 用于临时截获流经第 21 层的数据
activations = []
def get_activation(name):
    # hook(model, input, output)
    # output[0] shape: [batch=1, seq_len, hidden_dim]
    def hook(model, input, output):
        # 我们把它转回 CPU 并detach，防止显存爆炸
        # shape: [1, seq_len, hidden_dim]
        activations.append(output[0].detach().cpu())
    return hook

# 注册钩子到目标层
# Qwen2 的层命名通常是 model.layers.X
model.model.layers[TARGET_LAYER].register_forward_hook(get_activation("layer_21"))

# 4. 开始采集
print(f"Collecting activations from Layer {TARGET_LAYER+1}...")
collected_count = 0
all_entropies = [] # 存储真实熵

with torch.no_grad():
    for i, item in tqdm(enumerate(data), total=NUM_SAMPLES):
        if collected_count >= NUM_SAMPLES:
            break
            
        text = item['text']
        # batch_size=1
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(device)
        
        # 前向传播 (Forward Pass)
        # 这时 Hook 会自动触发，把数据存进 activations 列表
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits # [batch=1, seq_len, vocab_size]
        
        # 计算真实熵 (Entropy)
        # Entropy = -sum(p * log(p))
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1) # [batch=1, seq_len]
        all_entropies.append(entropy.detach().cpu())

        collected_count += 1

# 5. 整理并保存
print("Saving data to disk...")
# activations 是 list of [1, seq_len, hidden_dim]
# 我们需要把它们沿 seq_len 维度拼接，但每个样本 seq_len 不同，所以不能直接 stack
# 先 squeeze 掉 batch 维度，变成 list of [seq_len, hidden_dim]
flat_activations = [x.squeeze(0) for x in activations] 
flat_entropies = [x.squeeze(0) for x in all_entropies]

# 然后沿 dim=0 (token 数量) 拼接
all_hidden_states = torch.cat(flat_activations, dim=0) # [Total_Tokens, Hidden_Dim]
all_entropies_tensor = torch.cat(flat_entropies, dim=0) # [Total_Tokens]

# 确保形状对齐
if all_hidden_states.shape[0] != all_entropies_tensor.shape[0]:
    # 有时候 hook 和 logits 的 token 数可能会有一点差异（如 padding），这里做个截断对齐
    min_len = min(all_hidden_states.shape[0], all_entropies_tensor.shape[0])
    all_hidden_states = all_hidden_states[:min_len]
    all_entropies_tensor = all_entropies_tensor[:min_len]

torch.save(all_hidden_states, os.path.join(SAVE_DIR, "hidden_states_layer21.pt"))
torch.save(all_entropies_tensor, os.path.join(SAVE_DIR, "entropies_layer21.pt"))

print(f"✅ Done! Saved {all_hidden_states.shape[0]} tokens of data.")
print(f"Hidden Shape: {all_hidden_states.shape}")
print(f"Entropy Shape: {all_entropies_tensor.shape}")
print(f"File size approx: {all_hidden_states.element_size() * all_hidden_states.numel() / 1024 / 1024:.2f} MB")
