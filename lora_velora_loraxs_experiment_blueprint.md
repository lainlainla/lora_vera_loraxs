# LoRA / VeRA / VeLoRA / LoRA-XS 对比实验设计蓝图

> 版本：v1.0  
> 目标：把前面交流内容整理成可执行的实验蓝图，用于评估 LoRA、LoRA-XS、VeRA / VeLoRA 相关方法的复现难度、资源消耗、评估指标、实现代码差异、完成时间与代码重合度。

---

## 0. 术语校准：先区分 VeRA 和 VeLoRA

前面讨论中最重要的澄清是：

- **LoRA-XS 论文中直接对比和讨论 parameter sharing 的方法是 VeRA，不是 VeLoRA。**
- **VeRA**：parameter sharing LoRA 变体。它使用跨层共享的 frozen low-rank matrices，并训练小的 scaling vectors。
- **VeLoRA**：activation memory saving 方法。它主要压缩 forward 中需要保存给 backward 的 activations，并不是典型 parameter sharing 方法。

因此，本实验建议分成两条线：

1. **Parameter-sharing 主线**  
   比较：
   
   $$
   \text{LoRA} \rightarrow \text{VeRA} \rightarrow \text{LoRA-XS} \rightarrow \text{Shared-}R\text{ LoRA-XS}
   $$

2. **Activation-memory 主线，可选扩展**  
   比较：
   
   $$
   \text{LoRA} \rightarrow \text{LoRA + VeLoRA} \rightarrow \text{LoRA-XS} \rightarrow \text{LoRA-XS + VeLoRA}
   $$

如果最终报告必须写“LoRA / VeLoRA / LoRA-XS 三者对比”，建议在报告中明确说明：

> VeLoRA 与 LoRA-XS 优化目标不同：VeLoRA 主要优化训练显存中的 activation memory；LoRA-XS 主要优化 trainable parameters 和 adapter storage。因此直接比较 accuracy 并不公平，必须同时比较参数量、显存、速度、实现复杂度和复现成本。

---

## 1. 实验总目标

本项目不是单纯比较 accuracy，而是比较三类方法的 **复现难度与工程代价**。

核心研究问题：

$$
Q_1:\ \text{LoRA-XS 是否能在远少于 LoRA 的 trainable parameters 下保持或超过 LoRA 性能？}
$$

$$
Q_2:\ \text{VeRA 的 parameter sharing 与 LoRA-XS 的 SVD basis 相比，参数效率和性能如何？}
$$

$$
Q_3:\ \text{如果把 parameter sharing 引入 LoRA-XS，是否能进一步降低参数量？}
$$

$$
Q_4:\ \text{VeLoRA 是否能在不显著降低性能的情况下减少 activation memory？}
$$

$$
Q_5:\ \text{三类方法在实现复杂度、调试成本、代码重合度和完成时间上差异多大？}
$$

---

## 2. 理论主线

### 2.1 LoRA

标准 LoRA 将 frozen weight $W$ 修改为：

$$
W' = W + \Delta W
$$

$$
\Delta W = BA
$$

其中：

$$
A \in \mathbb{R}^{r \times d_{in}},\quad B \in \mathbb{R}^{d_{out} \times r}
$$

对于方阵 $W\in\mathbb{R}^{n\times n}$，每个模块的 LoRA 可训练参数量为：

$$
P_{\text{LoRA,module}} = 2nr
$$

若共有 $L$ 层，每层插入 $q$ 个 LoRA 模块：

$$
P_{\text{LoRA}} = Lq(2nr)
$$

LoRA 的优点是实现成熟、PEFT 支持完善、复现难度低。缺点是当有大量用户或任务需要保存 adapter 时，checkpoint storage 仍然较大。

---

### 2.2 VeRA：parameter sharing baseline

VeRA 的核心思想：

$$
\text{shared frozen random basis} + \text{layer-specific trainable scaling vectors}
$$

它不是每层都训练完整的 LoRA $A,B$，而是跨层共享 frozen low-rank matrices，只训练少量 scaling vectors。

可训练参数近似写作：

$$
P_{\text{VeRA}} = Lq(n+r)
$$

其中：

- $L$：层数；
- $q$：每层插入模块数；
- $n$：hidden dimension；
- $r$：rank。

VeRA 是 parameter sharing 主线中的重要 baseline，因为它代表：

$$
\text{随机共享基底} + \text{小规模可训练调制}
$$

---

### 2.3 LoRA-XS：梯度子空间 / SVD 视角

LoRA-XS 的核心不是随机低秩更新，而是基于预训练权重 $W$ 的 SVD 选择更新子空间。

对预训练权重做 SVD：

$$
W = U\Sigma V^\top
$$

取 top-$r$：

$$
U_r,\quad \Sigma_r,\quad V_r
$$

LoRA-XS 设置：

$$
A = U_r\Sigma_r
$$

$$
B = V_r^\top
$$

然后只训练一个小矩阵：

$$
R \in \mathbb{R}^{r\times r}
$$

最终形式：

$$
W' = W + U_r\Sigma_r R V_r^\top
$$

或：

$$
W' = W + ARB
$$

其中 $A,B$ frozen，只有 $R$ trainable。

每个模块的可训练参数量为：

$$
P_{\text{LoRA-XS,module}} = r^2
$$

总参数量：

$$
P_{\text{LoRA-XS}} = Lqr^2
$$

与 LoRA 对比：

$$
\frac{P_{\text{LoRA}}}{P_{\text{LoRA-XS}}}
=
\frac{2n}{r}
$$

与 VeRA 对比：

$$
\frac{P_{\text{VeRA}}}{P_{\text{LoRA-XS}}}
=
\frac{n+r}{r^2}
$$

当 $n\gg r$ 时，LoRA-XS 的参数优势非常明显。

---

### 2.4 LoRA-XS 的梯度投影视角

LoRA-XS 可以解释为：把 full gradient 限制在由 $W$ 的主奇异方向定义的低维子空间中。

定义子空间：

$$
S^r_{A,B}
=
\{AXB^\top: X\in \mathbb{R}^{r\times r}\}
$$

如果 full gradient 是：

$$
G=\nabla_W \mathcal{L}
$$

则其在该子空间中的投影为：

$$
p_{A,B}(G)
=
A[A^\top G B]B^\top
$$

LoRA-XS 实际优化的是低维 latent update：

$$
U_r^\top G V_r
$$

因此，LoRA-XS 可以理解为：

> 使用预训练权重的 top singular directions 近似未来 fine-tuning gradient 的主要更新方向。

---

### 2.5 Shared-$R$ LoRA-XS：建议做的小改进

不建议直接把 LoRA-XS 的 $A,B$ 跨层共享，因为 LoRA-XS 的理论基础是每层权重 $W_{\ell,m}$ 自己的 SVD：

$$
W_{\ell,m} = U_{\ell,m}\Sigma_{\ell,m}V_{\ell,m}^\top
$$

更合理的 parameter sharing 是：

> 每层保留自己的 SVD basis，但共享中间的 $R$。

形式：

$$
W_{\ell,m}'
=
W_{\ell,m}
+
U_{\ell,m,r}
\Sigma_{\ell,m,r}
R_{g(\ell,m)}
V_{\ell,m,r}^\top
$$

其中 $g(\ell,m)$ 是共享分组函数。

#### 方案 A：原版 LoRA-XS，不共享

$$
R_{\ell,m}
$$

参数量：

$$
P = Lqr^2
$$

#### 方案 B：按模块类型共享

所有 `q_proj` 共享 $R_q$，所有 `v_proj` 共享 $R_v$：

$$
R_{g(\ell,m)} = R_m
$$

参数量：

$$
P = qr^2
$$

#### 方案 C：按层组共享

每 $s$ 层共享一个 $R$：

$$
g(\ell,m)=\left(\left\lfloor \frac{\ell}{s}\right\rfloor,m\right)
$$

参数量：

$$
P = Gqr^2
$$

其中 $G=\lceil L/s\rceil$。

#### 方案 D：共享 $R$ + layer-specific scalar

$$
W_{\ell,m}'
=
W_{\ell,m}
+
\alpha_{\ell,m}
U_{\ell,m,r}
\Sigma_{\ell,m,r}
R_m
V_{\ell,m,r}^\top
$$

参数量：

$$
P = qr^2 + Lq
$$

该版本最推荐，因为它兼顾：

$$
\text{SVD-informed basis}
$$

和：

$$
\text{parameter sharing}
$$

---

## 3. 梯度冲突分析：解释 parameter sharing 是否有效

对于第 $\ell$ 层模块的 full gradient：

$$
G_{\ell,m} = \nabla_{W_{\ell,m}}\mathcal{L}
$$

LoRA-XS 中 $R_{\ell,m}$ 的梯度可以近似写作：

$$
\nabla_{R_{\ell,m}}\mathcal{L}
=
\Sigma_{\ell,m,r}
U_{\ell,m,r}^{\top}
G_{\ell,m}
V_{\ell,m,r}
$$

如果共享同一个 $R_m$，则 shared $R_m$ 的梯度为：

$$
\nabla_{R_m}\mathcal{L}
=
\sum_{\ell}
\Sigma_{\ell,m,r}
U_{\ell,m,r}^{\top}
G_{\ell,m}
V_{\ell,m,r}
$$

因此，parameter sharing 是否合理，取决于不同层投影后的 gradient 是否方向一致。

定义：

$$
g_i=
\operatorname{vec}
\left(
\Sigma_{i,r}U_{i,r}^{\top}G_iV_{i,r}
\right)
$$

$$
g_j=
\operatorname{vec}
\left(
\Sigma_{j,r}U_{j,r}^{\top}G_jV_{j,r}
\right)
$$

cosine similarity：

$$
\cos(g_i,g_j)
=
\frac{\langle g_i,g_j\rangle}{\|g_i\|\|g_j\|}
$$

梯度冲突率：

$$
C
=
\frac{2}{N(N-1)}
\sum_{i<j}
\mathbf{1}[\cos(g_i,g_j)<0]
$$

解释：

- $C$ 越小，说明共享 $R$ 越合理；
- $C$ 越大，说明不同层更新方向冲突，共享会损害性能。

---

## 4. 实验方法矩阵

### 4.1 主实验方法

| 方法 | 是否必须 | 作用 | 主要比较维度 |
|---|---:|---|---|
| LoRA | 必须 | baseline | 性能、参数量、速度、代码复杂度 |
| VeRA | 建议必须 | parameter sharing baseline | 参数共享效果、参数量、性能 |
| LoRA-XS | 必须 | paper 核心方法 | SVD basis、参数效率、性能 |
| Shared-$R$ LoRA-XS | 必须 | 你的改进点 | 共享 $R$、梯度冲突、极低参数量 |
| LoRA + VeLoRA | 可选 | activation memory baseline | 显存、速度、实现复杂度 |
| LoRA-XS + VeLoRA | 可选 | 组合型扩展 | 参数效率 + activation memory |

---

### 4.2 最小可行实验矩阵

| 方法 | Rank / 设置 | Target Modules | 目标 |
|---|---:|---|---|
| LoRA | $r=4,8,16$ | q, v | 标准 baseline |
| VeRA | $r=128,256$ | q, v | parameter sharing baseline |
| LoRA-XS | $r=4,8,16,20$ | q, v | 同 target fair comparison |
| LoRA-XS-paper-style | $r=8,16,20,25$ | q, v, attn output, FC1 | 接近 paper setting |
| Shared-$R$ LoRA-XS | $r=8,16,20$ | q, v | 你的改进方法 |
| Shared-$R$+scale LoRA-XS | $r=8,16,20$ | q, v | 稳定共享版本 |
| LoRA + VeLoRA | LoRA $r=8$, sub-token $M=16,32,64$ | q, v | activation memory stress test |

---

## 5. 数据集与模型设计

### 5.1 阶段 A：最小可复现实验

| 项目 | 设置 |
|---|---|
| 模型 | RoBERTa-base |
| 数据集 | SST-2, MRPC |
| seeds | 1 |
| 方法 | LoRA, LoRA-XS, Shared-$R$ LoRA-XS |
| 目标 | 跑通训练、记录参数量、显存、时间、代码复杂度 |

预计时间：3–5 天。

---

### 5.2 阶段 B：正式主实验

| 项目 | 设置 |
|---|---|
| 模型 | RoBERTa-base；资源足够则 RoBERTa-large |
| 数据集 | SST-2, MRPC, CoLA, QNLI |
| seeds | 3 |
| 方法 | LoRA, VeRA, LoRA-XS, Shared-$R$ LoRA-XS |
| 指标 | accuracy / F1 / MCC、参数量、显存、训练时间、代码复杂度 |

预计时间：7–14 天。

---

### 5.3 阶段 C：可选 LLM 扩展实验

| 项目 | 设置 |
|---|---|
| 模型 | 0.5B–1.5B decoder-only model |
| 数据集 | GSM8K subset / Alpaca subset |
| 方法 | LoRA, LoRA-XS, Shared-$R$ LoRA-XS |
| 指标 | exact match / loss / ppl、显存、训练时间 |
| 目标 | 验证方法是否能迁移到 decoder-only LLM |

预计额外时间：5–10 天。

---

### 5.4 阶段 D：VeLoRA 显存压力实验，可选

| 项目 | 设置 |
|---|---|
| 模型 | RoBERTa-base / RoBERTa-large |
| 数据集 | SST-2 / MRPC |
| seq length | 128, 256, 512 |
| batch size | 8, 16, 32 |
| 方法 | LoRA vs LoRA + VeLoRA |
| 指标 | peak VRAM, saved activation memory, step time, score |

目标：

$$
\text{验证 VeLoRA 是否真实降低 activation memory}
$$

---

## 6. 评估指标

### 6.1 任务性能指标

#### SST-2 / QNLI / RTE

$$
\text{Accuracy}
=
\frac{\#\text{correct predictions}}{\#\text{all examples}}
$$

#### MRPC

$$
\text{Accuracy},\quad F1
$$

#### CoLA

$$
\text{Matthews Correlation Coefficient}
$$

#### STS-B，可选

$$
\text{Pearson Correlation}
$$

#### GSM8K，可选

$$
\text{Exact Match}
$$

---

### 6.2 参数效率指标

#### 可训练参数量

$$
P_{\text{train}}
=
\sum_{p\in\Theta_{\text{train}}}\operatorname{numel}(p)
$$

#### 可训练参数比例

$$
R_{\text{train}}
=
\frac{P_{\text{train}}}{P_{\text{total}}}
$$

#### Adapter checkpoint size

$$
S_{\text{adapter}}
=
P_{\text{saved}}\times \text{bytes per parameter}
$$

注意：

- LoRA 保存 $A,B$；
- LoRA-XS 理论上最好只保存 $R$；
- 如果保存 frozen $A,B$，会削弱 LoRA-XS 的 storage advantage；
- Shared-$R$ LoRA-XS 只保存共享 $R$ 和少量 scale 参数。

#### 参数效率得分

$$
E_{\text{param}}
=
\frac{\text{Task Score}}{\log_{10}(P_{\text{train}}+1)}
$$

或：

$$
\Delta M_{\text{per 1M params}}
=
\frac{M_{\text{method}}-M_{\text{baseline}}}{P_{\text{train}}/10^6}
$$

---

### 6.3 显存指标

#### Peak allocated memory

$$
VRAM_{\text{alloc-peak}}
=
\max_t \operatorname{torch.cuda.memory\_allocated}(t)
$$

#### Peak reserved memory

$$
VRAM_{\text{reserved-peak}}
=
\max_t \operatorname{torch.cuda.memory\_reserved}(t)
$$

#### nvidia-smi 实际显存

$$
VRAM_{\text{nvidia-smi}}
$$

#### Saved tensor memory

使用 `torch.autograd.graph.saved_tensors_hooks` 统计：

$$
M_{\text{saved-tensors}}
=
\sum_i \operatorname{numel}(T_i)\times \text{bytes}(T_i)
$$

该指标对 VeLoRA 特别重要，因为 VeLoRA 优化的是 backward 需要保存的 activations。

#### 显存下降率

$$
\Delta VRAM
=
\frac{
VRAM_{\text{method}}-VRAM_{\text{LoRA}}
}{
VRAM_{\text{LoRA}}
}
$$

若为负数，表示省显存。

#### Activation compression ratio

$$
CR_{\text{act}}
=
\frac{
M_{\text{saved-tensors, LoRA}}
}{
M_{\text{saved-tensors, method}}
}
$$

---

### 6.4 时间指标

#### 初始化时间

$$
T_{\text{init}}
$$

LoRA-XS 额外记录：

$$
T_{\text{SVD}}
$$

#### 每 step 时间

$$
T_{\text{step}}
=
\frac{T_{\text{train}}}{N_{\text{steps}}}
$$

#### 总训练时间

$$
T_{\text{total}}
=
T_{\text{init}} + T_{\text{train}} + T_{\text{eval}}
$$

#### Throughput

$$
\text{samples/sec}
=
\frac{N_{\text{samples}}}{T_{\text{train}}}
$$

$$
\text{tokens/sec}
=
\frac{N_{\text{tokens}}}{T_{\text{train}}}
$$

#### Time-to-quality

设目标分数为：

$$
M_{\text{target}}=0.95\times M_{\text{LoRA-final}}
$$

记录：

$$
T_{\text{to-target}}
=
\min_t\{t:M_t\ge M_{\text{target}}\}
$$

---

### 6.5 代码复杂度与代码重合度指标

#### 方法专属代码行数

$$
LOC_{\text{method}}
$$

只统计：

```text
src/methods/lora.py
src/methods/vera.py
src/methods/loraxs.py
src/methods/shared_loraxs.py
src/methods/velora.py
```

不要把 dataset、trainer、eval 公共代码算进去。

#### 公共代码复用率

$$
ReuseRate_m
=
\frac{
LOC_{\text{common}}
}{
LOC_{\text{common}}+LOC_{\text{method},m}
}
$$

#### 两两代码重合度：token Jaccard

$$
Overlap(i,j)
=
\frac{
|Tokens_i \cap Tokens_j|
}{
|Tokens_i \cup Tokens_j|
}
$$

#### difflib similarity

也可以使用：

```python
from difflib import SequenceMatcher
sim = SequenceMatcher(None, code_i, code_j).ratio()
```

#### 修改文件数

$$
N_{\text{files-changed}}
$$

#### 自定义 autograd 数量

$$
N_{\text{custom-autograd}}
$$

预期：

| 方法 | Custom Autograd |
|---|---:|
| LoRA | 0 |
| VeRA | 0 |
| LoRA-XS | 0 |
| Shared-$R$ LoRA-XS | 0 |
| VeLoRA | 1+ |

---

## 7. 实验记录 schema

建议每次 run 保存一行 JSONL 或 CSV：

```text
run_id
method
base_model
dataset
seed
rank
target_modules
sharing_type
learning_rate
batch_size
max_seq_len
num_epochs
trainable_params
trainable_ratio
adapter_size_mb
peak_alloc_gb
peak_reserved_gb
saved_tensors_gb
svd_init_sec
train_sec
eval_sec
step_time_sec
samples_per_sec
tokens_per_sec
score
best_epoch
method_loc
files_changed
reuse_rate
overlap_with_lora
custom_autograd_count
notes
```

---

## 8. 结果表模板

### 8.1 任务性能表

| Method | Dataset | Rank | Sharing | Seed 1 | Seed 2 | Seed 3 | Mean | Std |
|---|---|---:|---|---:|---:|---:|---:|---:|
| LoRA | SST-2 | 8 | none |  |  |  |  |  |
| VeRA | SST-2 | 256 | shared $A,B$ |  |  |  |  |  |
| LoRA-XS | SST-2 | 16 | none |  |  |  |  |  |
| Shared-$R$ LoRA-XS | SST-2 | 16 | module-wise |  |  |  |  |  |

---

### 8.2 参数与存储表

| Method | Rank | Target Modules | Trainable Params | Trainable % | Adapter Size MB | Saved Buffers MB |
|---|---:|---|---:|---:|---:|---:|
| LoRA | 8 | q,v |  |  |  |  |
| VeRA | 256 | q,v |  |  |  |  |
| LoRA-XS | 16 | q,v |  |  |  |  |
| Shared-$R$ LoRA-XS | 16 | q,v |  |  |  |  |

---

### 8.3 显存与速度表

| Method | Seq Len | Batch | Peak Alloc GB | Peak Reserved GB | Saved Tensor GB | Step Time s | Samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 128 | 32 |  |  |  |  |  |
| VeRA | 128 | 32 |  |  |  |  |  |
| LoRA-XS | 128 | 32 |  |  |  |  |  |
| Shared-$R$ LoRA-XS | 128 | 32 |  |  |  |  |  |
| LoRA + VeLoRA | 512 | 16 |  |  |  |  |  |

---

### 8.4 初始化与 human time 成本表

| Method | Env Setup | Implementation Time | Debug Time | SVD Time | Total Human Time | Main Risk |
|---|---:|---:|---:|---:|---:|---|
| LoRA |  |  |  | 0 |  | PEFT config |
| VeRA |  |  |  | 0 |  | sharing matrix registration |
| LoRA-XS |  |  |  |  |  | SVD orientation / checkpoint |
| Shared-$R$ LoRA-XS |  |  |  |  |  | gradient conflict / sharing design |
| VeLoRA |  |  |  | 0 |  | custom backward / memory validation |

---

### 8.5 代码复杂度表

| Method | Method LOC | Files Changed | Custom Autograd | Reuse Rate | Overlap with LoRA |
|---|---:|---:|---:|---:|---:|
| LoRA |  |  | 0 |  | 1.00 |
| VeRA |  |  | 0 |  |  |
| LoRA-XS |  |  | 0 |  |  |
| Shared-$R$ LoRA-XS |  |  | 0 |  |  |
| VeLoRA |  |  | 1+ |  |  |

---

## 9. 必须画的图

1. **Score vs Trainable Parameters**
2. **Score vs Adapter Size**
3. **Score vs Peak GPU Memory**
4. **Peak Memory vs Sequence Length**
5. **Score vs Training Time**
6. **Step Time vs Rank**
7. **Gradient Conflict Rate vs Sharing Granularity**
8. **Code LOC / Reuse Rate Bar Chart**
9. **Reproduction Difficulty Radar Chart**
10. **Pareto Frontier：Score / Params / Memory / Time**

---

## 10. 代码结构建议

```text
project/
  README.md
  configs/
    roberta_base_sst2.yaml
    roberta_base_mrpc.yaml
    roberta_base_cola.yaml
    roberta_base_qnli.yaml

  src/
    train.py
    evaluate.py

    methods/
      __init__.py
      lora.py
      vera.py
      loraxs.py
      shared_loraxs.py
      velora.py

    metrics/
      resource_meter.py
      code_overlap.py
      gradient_conflict.py
      param_counter.py

    utils/
      seed.py
      logging.py
      checkpoint.py
      svd.py

  scripts/
    run_stage_a.sh
    run_stage_b.sh
    run_memory_stress.sh
    analyze_code_overlap.py
    plot_results.py

  results/
    raw_runs.jsonl
    summary_tables/
    figures/
```

---

## 11. 实现要点

### 11.1 LoRA

优先使用 PEFT 实现，保证 baseline 稳定。

需要记录：

- target modules；
- rank；
- alpha；
- dropout；
- trainable params；
- adapter checkpoint size。

---

### 11.2 VeRA

实现重点：

- 构造 shared frozen random matrices；
- 每层注册 trainable scaling vectors；
- 确保 shared matrices 不是每层重复保存；
- 参数统计时只统计 scaling vectors。

主要风险：

- 共享矩阵如果被错误注册为每层独立 buffer，会导致存储统计不准确；
- scaling vector shape 容易和 Linear weight orientation 对不上。

---

### 11.3 LoRA-XS

实现重点：

1. 对每个 target linear layer 读取 frozen weight：

   $$
   W\in\mathbb{R}^{d_{out}\times d_{in}}
   $$

2. 对 $W$ 做 truncated SVD：

   $$
   W = U\Sigma V^\top
   $$

3. 取 top-$r$：

   $$
   A=U_r\Sigma_r,\quad B=V_r^\top
   $$

4. 注册 $A,B$ 为 frozen buffer。
5. 注册 $R\in\mathbb{R}^{r\times r}$ 为 trainable parameter。
6. forward：

   $$
   y=xW^\top + x(ARB)^\top
   $$

7. checkpoint 最好只保存 $R$，加载时重新从 base model weight 计算 $A,B$。

主要风险：

- PyTorch `nn.Linear.weight` 的 shape 是 `[d_out, d_in]`；
- 公式和代码中的转置方向容易出错；
- SVD 初始化时间要单独记录；
- 如果把 frozen $A,B$ 也保存，会扭曲 storage comparison。

---

### 11.4 Shared-$R$ LoRA-XS

推荐实现三个版本：

#### Version 1：module-wise shared $R$

$$
R_q,\quad R_v
$$

所有层的 q_proj 共享 $R_q$，所有层的 v_proj 共享 $R_v$。

#### Version 2：group-wise shared $R$

每 $s$ 层共享：

$$
R_{group,m}
$$

推荐：

$$
s\in\{2,4,6\}
$$

#### Version 3：shared $R$ + scalar

$$
\alpha_{\ell,m}R_m
$$

每层每模块一个 scalar：

$$
\alpha_{\ell,m}
$$

这是最稳的 parameter sharing 版本。

---

### 11.5 VeLoRA

若做 VeLoRA，重点是验证它是否真的省 activation memory。

不能只写普通 wrapper，否则 PyTorch autograd 仍可能保存原始 input activation。

需要：

- 自定义 `torch.autograd.Function`；
- forward 保存压缩后的 activation；
- backward 用重构 activation 近似计算 gradient；
- 用 `saved_tensors_hooks` 检查保存 tensor 是否减少。

主要风险：

- backward 公式写错；
- score 降低；
- step time 增加；
- 显存下降没有达到预期；
- 与 mixed precision / gradient checkpointing 交互复杂。

---

## 12. 复现难度评分体系

定义总难度：

$$
D
=
0.25D_{\text{impl}}
+
0.20D_{\text{compute}}
+
0.20D_{\text{metric}}
+
0.20D_{\text{stability}}
+
0.15D_{\text{paper-gap}}
$$

每项 1–5 分：

| 分数 | 含义 |
|---:|---|
| 1 | 很容易，基本有现成库 |
| 2 | 有少量工程改动 |
| 3 | 需要自定义模块，但逻辑清楚 |
| 4 | 需要 custom autograd / 显存验证 / 多轮调试 |
| 5 | 需要大算力、多 seed、多任务、复杂复现 |

预估评分：

| Method | Impl | Compute | Metric | Stability | Paper-gap | Overall |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 1 | 2 | 1 | 2 | 1 | 1.4 |
| VeRA | 2 | 2 | 2 | 3 | 3 | 2.4 |
| LoRA-XS | 3 | 2 | 2 | 3 | 3 | 2.7–3.1 |
| Shared-$R$ LoRA-XS | 4 | 2 | 3 | 4 | 4 | 3.4–3.8 |
| VeLoRA | 5 | 2 | 3 | 4 | 4 | 3.8–4.3 |
| 完整对比实验 | 4 | 3 | 3 | 4 | 4 | 3.8–4.2 |

---

## 13. 资源需求评估

### 13.1 最小资源

| 项目 | 配置 |
|---|---|
| GPU | 1 × 24GB，例如 RTX 3090 / 4090 |
| 模型 | RoBERTa-base |
| 数据集 | SST-2, MRPC |
| seeds | 1 |
| 预计完成 | 3–5 天 |

适合快速验证。

---

### 13.2 推荐资源

| 项目 | 配置 |
|---|---|
| GPU | 1 × A100 40GB 或 1 × RTX 4090 |
| 模型 | RoBERTa-base + 部分 RoBERTa-large |
| 数据集 | SST-2, MRPC, CoLA, QNLI |
| seeds | 3 |
| 预计完成 | 7–14 天 |

适合课程项目 / 小论文复现。

---

### 13.3 论文级资源

| 项目 | 配置 |
|---|---|
| GPU | A100 / H100 |
| 模型 | RoBERTa-large + 7B/8B decoder-only models |
| 数据集 | GLUE + GSM8K + MATH + commonsense |
| seeds | 5 |
| 预计完成 | 3–6 周 |

适合接近论文级复现。

---

## 14. 时间计划

### 14.1 7 天版本

| Day | 任务 |
|---:|---|
| Day 1 | 搭建统一训练框架，跑通 LoRA baseline |
| Day 2 | 实现 LoRA-XS Linear wrapper，跑通 SST-2 |
| Day 3 | 实现 Shared-$R$ LoRA-XS，验证参数共享 |
| Day 4 | 实现 VeRA 或简化 VeRA baseline |
| Day 5 | 跑 SST-2 / MRPC 单 seed 对比 |
| Day 6 | 统计参数量、显存、时间、代码复杂度、代码重合度 |
| Day 7 | 画图，写复现难度报告 |

---

### 14.2 14 天版本

| 阶段 | 时间 |
|---|---:|
| 统一代码框架 | 2 天 |
| LoRA + LoRA-XS 实现 | 2–3 天 |
| VeRA + Shared-$R$ 实现 | 3 天 |
| GLUE 4 个任务 × 3 seeds | 4–5 天 |
| 显存/时间/代码复杂度统计 | 1–2 天 |
| 图表与报告 | 2 天 |

---

### 14.3 3–6 周版本

| 阶段 | 时间 |
|---|---:|
| 完整 GLUE 复现 | 1–2 周 |
| Shared-$R$ ablation | 1 周 |
| VeLoRA 显存压力实验 | 1 周 |
| LLM instruction tuning 扩展 | 1–2 周 |
| 报告和可视化 | 3–5 天 |

---

## 15. 最终验收标准

### 合格版本

- LoRA、LoRA-XS、Shared-$R$ LoRA-XS 至少 3 个方法跑通；
- 至少 2 个数据集；
- 有 accuracy / F1 / MCC；
- 有 trainable params、peak VRAM、training time；
- 有代码复杂度统计。

---

### 较好版本

- LoRA、VeRA、LoRA-XS、Shared-$R$ LoRA-XS 全部跑通；
- 4 个 GLUE 任务；
- 3 seeds；
- 有 parameter efficiency curve；
- 有 code overlap；
- 有 reproduction difficulty score。

---

### 很好版本

- 加入 VeLoRA activation memory stress test；
- 做 max sequence length 128 / 256 / 512 对比；
- 用 saved tensor hooks 验证 activation memory；
- 做 gradient conflict analysis；
- 做 sharing granularity ablation。

---

### 接近论文级版本

- RoBERTa-large；
- 5 seeds；
- 完整 GLUE subset；
- LLM instruction tuning；
- 多 rank；
- 多 target modules；
- 完整消融实验；
- 详细复现 gap 分析。

---

## 16. 最终建议路线

推荐优先做：

$$
\text{LoRA}
\rightarrow
\text{VeRA}
\rightarrow
\text{LoRA-XS}
\rightarrow
\text{Shared-}R\text{ LoRA-XS}
$$

其中：

- LoRA 是 baseline；
- VeRA 是 parameter sharing baseline；
- LoRA-XS 是 paper 核心；
- Shared-$R$ LoRA-XS 是你的研究改进点；
- VeLoRA 作为可选 activation memory 扩展，不应放在 parameter sharing 主线中。

最推荐的论文 / 报告标题：

> **LoRA, VeRA and LoRA-XS: A Comparative Study of Gradient Subspace Adaptation, Parameter Sharing, Resource Consumption and Reproducibility**

如果必须包含 VeLoRA，可以写：

> **LoRA, VeLoRA and LoRA-XS: A Comparative Study of Parameter Efficiency, Activation Memory, Implementation Complexity and Reproducibility**

但正文中必须说明：

> VeRA 和 LoRA-XS 主要比较 parameter sharing / trainable parameters；VeLoRA 主要比较 activation memory，二者优化目标不同。

---

## 17. 最终结论预判

整体复现要求：

$$
\text{中等偏高}
$$

分方法看：

- **LoRA**：低难度，适合作 baseline；
- **VeRA**：中等难度，重点是共享参数注册和参数统计；
- **LoRA-XS**：中等偏高，重点是 SVD 初始化、矩阵方向、checkpoint 统计；
- **Shared-$R$ LoRA-XS**：中高难度，但最有研究价值；
- **VeLoRA**：高难度，重点是 custom autograd 和 memory validation。

最终报告的关键不是证明“哪个方法 accuracy 最高”，而是形成完整的多维评估：

$$
\text{Performance}
$$

$$
\text{Trainable Parameters}
$$

$$
\text{Adapter Storage}
$$

$$
\text{Peak GPU Memory}
$$

$$
\text{Saved Activation Memory}
$$

$$
\text{Training Time}
$$

$$
\text{Implementation Complexity}
$$

$$
\text{Code Overlap}
$$

$$
\text{Reproduction Difficulty}
$$

这才符合“实验设计蓝图”和“复现难度评估”的目标。
