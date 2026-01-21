# SEDAC V7.0 设计方案：全层熵监控架构

## V6.0 缺陷分析

| 缺陷 | 根因 | 影响 |
|------|------|------|
| **批处理木桶效应** | GPU 同步执行，整个 batch 必须等最慢 token | 延迟收益有限 |
| **浅层退出语义风险** | Layer 7 固定检查点，浅层语义不完整 | 质量下降 |
| **工程实现脆弱** | Monkey patching + 硬编码字符串匹配 | 维护困难 |
| **阈值调优复杂** | 3 层阈值耦合，搜索空间爆炸 | 难以调参 |
| **硬件架构耦合** | 硬编码 Qwen2 类名 | 泛化性差 |

---

## V7.0 核心设计：全层熵监控 + 单阈值决策

### 设计理念

```
V6.0: 固定检查点 (L7, L14, L21) → 3 个独立阈值 → 耦合复杂
V7.0: 每层监控熵变化率 → 单一退出条件 → 解耦简化
```

### 核心创新

#### 1. 轻量级全层熵估计器 (Lightweight Per-Layer Entropy Estimator)

**问题**：每层都跑完整 LREProbe 开销太大

**方案**：使用 **hidden state 变化率** 作为熵的代理指标

```python
# 计算方式：相邻层 hidden state 的余弦相似度
stability[i] = cosine_similarity(h[i], h[i-1])

# 稳定性高 (接近1) → 语义收敛 → 可以退出
# 稳定性低 (接近0) → 语义变化中 → 继续计算
```

**优势**：
- 无需额外参数，无需训练
- 计算开销极低（一次向量内积）
- 物理意义清晰：语义稳定 = 可退出

#### 2. 单阈值动态退出条件

```python
# V6.0: 每层独立阈值，耦合严重
if risk[7] < thr_7: exit()
if risk[14] < thr_14: exit()
if risk[21] < thr_21: exit()

# V7.0: 统一退出条件
# 连续 K 层稳定性 > τ 时退出
consecutive_stable = 0
for i in range(num_layers):
    if stability[i] > tau:
        consecutive_stable += 1
        if consecutive_stable >= K:
            exit_at_layer(i)
    else:
        consecutive_stable = 0  # 重置
```

**参数**：
- `τ` (tau): 稳定性阈值，单一参数
- `K`: 连续稳定层数要求（默认 3）

#### 3. 解决批处理木桶效应

**方案 A：Speculative Execution (投机执行)**

```
不等待退出决策，继续执行后续层
如果最终确认可退出，丢弃多余计算
```

**方案 B：Token-Group 分离（推荐）**

```
1. 在 Layer M 进行一次同步检查
2. 将 batch 分为 "exit_group" 和 "continue_group"
3. exit_group 直接输出，continue_group 继续计算
4. 最后合并结果
```

```python
# 伪代码
for layer_idx in range(num_layers):
    hidden = layer(hidden)
    
    if layer_idx == M:  # 单一检查点，M 动态选择
        stability = compute_stability(hidden, prev_hidden)
        exit_mask = stability > tau
        
        if exit_mask.any():
            # 分离退出 tokens
            exit_tokens = hidden[exit_mask]
            continue_tokens = hidden[~exit_mask]
            
            # 只让 continue_tokens 继续
            hidden = continue_tokens
```

#### 4. 架构无关的 Hook 机制

替代 Monkey Patching，使用 PyTorch 原生 Hook：

```python
class SEDACHook:
    def __init__(self, model, tau=0.95, K=3):
        self.tau = tau
        self.K = K
        self.prev_hidden = None
        self.consecutive_stable = 0
        
        # 注册 forward hook 到每一层
        for layer in model.layers:
            layer.register_forward_hook(self.check_exit)
    
    def check_exit(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        if self.prev_hidden is not None:
            stability = F.cosine_similarity(
                hidden.mean(dim=1), 
                self.prev_hidden.mean(dim=1), 
                dim=-1
            )
            
            if (stability > self.tau).all():
                self.consecutive_stable += 1
                if self.consecutive_stable >= self.K:
                    raise EarlyExitSignal(hidden)
            else:
                self.consecutive_stable = 0
        
        self.prev_hidden = hidden.detach()
```

**优势**：
- 不修改模型源码
- 模型架构无关（Qwen2, Llama3, Mistral 通用）
- vLLM 版本升级不影响

---

## 参数对比

| 参数 | V6.0 | V7.0 |
|------|------|------|
| 探针数量 | 3 | 0 (无需训练) |
| 阈值数量 | 3 (耦合) | 1 (独立) |
| 检查点 | 固定 (7,14,21) | 动态 (每层) |
| 架构依赖 | Qwen2 only | 通用 |

---

## 预期收益

| 指标 | V6.0 | V7.0 预期 |
|------|------|----------|
| 阈值调优复杂度 | O(n³) | O(1) |
| 浅层误退出风险 | 高 | 低 (连续稳定检测) |
| 工程维护成本 | 高 | 低 (Hook 机制) |
| 模型泛化性 | 仅 Qwen2 | 通用 |

---

## 实现计划

1. **Phase 1**: 实现全层稳定性监控 + 单阈值退出
2. **Phase 2**: 实现 Token-Group 分离解决木桶效应
3. **Phase 3**: 架构无关 Hook 机制
4. **Phase 4**: 与 V6.0 对比测试

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 稳定性指标与熵不完全等价 | 可选：保留轻量探针作为辅助 |
| Hook 机制可能有性能开销 | 使用 torch.compile 优化 |
| 连续稳定检测可能过于保守 | 调整 K 值，或使用滑动窗口 |
