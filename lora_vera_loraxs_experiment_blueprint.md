# LoRA / VeRA / LoRA-XS / TinyLoRA 对比实验设计蓝图

> 版本说明：本蓝图已移除 VeLoRA。主实验对象为 **LoRA、VeRA、LoRA-XS**，新增 **TinyLoRA** 作为低参数极限与 RL reasoning 扩展实验。重点放在：参数效率、parameter sharing、SVD / random basis、target module 覆盖范围、梯度子空间视角、计算资源消耗、实现代码对比、完成时间对比、代码重合度与复现难度评估。

---

## 1. 实验定位

本实验不是单纯比较 accuracy，而是评估 LoRA 系 PEFT 方法的综合复现难度。GLUE 主线聚焦 LoRA / VeRA / LoRA-XS，TinyLoRA 作为低参数极限扩展：

\[
\text{LoRA} \quad vs \quad \text{VeRA} \quad vs \quad \text{LoRA-XS} \quad vs \quad \text{TinyLoRA}
\]

核心研究问题：

1. **性能问题**：LoRA-XS 是否能在更少可训练参数下达到或超过 LoRA / VeRA？
2. **参数效率问题**：LoRA、VeRA、LoRA-XS、TinyLoRA 的 trainable parameters、adapter checkpoint size 差异有多大？
3. **parameter sharing 问题**：VeRA 通过共享 frozen low-rank matrices 减参，这种共享机制和 LoRA-XS 的 SVD-informed basis 有何差异？
4. **梯度视角问题**：LoRA-XS 的 SVD 子空间是否能有效捕获 fine-tuning gradient 的主要方向？
5. **资源问题**：各方法的 peak GPU memory、step time、total training time、SVD initialization time 差异如何？
6. **工程复现问题**：各方法的实现复杂度、代码改动量、代码重合度、调试成本分别多高？
7. **target module 公平性问题**：LoRA-XS 在 paper-style 设置中比 LoRA 覆盖更多模块时，性能提升到底来自方法本身、参数效率，还是更大的模块覆盖范围？
8. **低参数极限问题**：TinyLoRA 能否把 LoRA-XS 的 \(R\in\mathbb{R}^{r\times r}\) 进一步压缩到 \(u\) 个参数，并通过 weight tying 达到单参数级别更新？

最终目标：形成一个可以支撑实验报告 / 课程项目 / 论文复现报告的系统化 benchmark。

---

## 2. 方法理论对比

### 2.1 LoRA

LoRA 在 frozen pretrained weight \(W\) 上加入低秩更新：

\[
W' = W + \Delta W
\]

\[
\Delta W = BA
\]

其中：

\[
A \in \mathbb{R}^{r \times d_{in}},\qquad B \in \mathbb{R}^{d_{out}\times r}
\]

对方阵 \(W\in\mathbb{R}^{n\times n}\)，如果有 \(L\) 层、每层 \(q\) 个 target modules，则 LoRA 可训练参数量为：

\[
P_{\text{LoRA}} = L \times q \times 2nr
\]

特点：

- 低秩更新，工程实现最成熟；
- 不改主模型结构，推理时可 merge；
- 参数量随 hidden dimension \(n\) 和 rank \(r\) 线性增长；
- 是本实验的 baseline。

---

### 2.2 VeRA

VeRA 的重点是 **parameter sharing**。

其基本思想是：

\[
\text{shared frozen random basis} + \text{layer-specific trainable scaling vectors}
\]

与 LoRA 每层训练自己的低秩矩阵不同，VeRA 使用跨层共享的 frozen low-rank matrices，只训练较小的 scaling vectors。

对方阵 \(W\in\mathbb{R}^{n\times n}\)，LoRA-XS paper 中给出的 VeRA 参数量近似为：

\[
P_{\text{VeRA}} = L \times q \times (n + r)
\]

特点：

- 参数共享机制强；
- 可训练参数通常显著少于 LoRA；
- 但参数量仍依赖 hidden dimension \(n\)；
- rank 往往需要设得较高，例如 paper 中 GLUE 对比里 VeRA 使用 rank 256；
- 本实验中 VeRA 是 parameter sharing baseline。

---

### 2.3 LoRA-XS

LoRA-XS 的核心是从 pretrained weight 的 SVD 中构造 frozen basis，只训练中间的小矩阵 \(R\)。

对权重矩阵做 SVD：

\[
W = U\Sigma V^T
\]

取 top-\(r\) singular components：

\[
A = U_r\Sigma_r
\]

\[
B = V_r^T
\]

LoRA-XS 更新为：

\[
W' = W + ARB
\]

也就是：

\[
W' = W + U_r\Sigma_r R V_r^T
\]

其中：

\[
R\in\mathbb{R}^{r\times r}
\]

只有 \(R\) 是 trainable。

因此对 \(L\) 层、每层 \(q\) 个 target modules：

\[
P_{\text{LoRA-XS}} = L \times q \times r^2
\]

与 LoRA / VeRA 的理论参数比为：

\[
\frac{P_{\text{LoRA}}}{P_{\text{LoRA-XS}}} = \frac{2n}{r}
\]

\[
\frac{P_{\text{VeRA}}}{P_{\text{LoRA-XS}}} = \frac{n+r}{r^2}
\]

当 \(n\gg r\) 时，LoRA-XS 的参数效率优势会非常明显。

特点：

- 只训练 \(r\times r\) 的 \(R\)；
- 参数量不直接依赖 hidden dimension \(n\)；
- 需要对每个 target weight 做 SVD 初始化；
- 工程难度高于 LoRA / VeRA；
- 是本实验的核心复现对象。

### 2.4 TinyLoRA

TinyLoRA 来自论文 **Learning to Reason in 13 Parameters**。它可以看作 LoRA-XS 的进一步压缩版本：仍然使用 frozen SVD basis，但不再直接训练完整的 \(R\in\mathbb{R}^{r\times r}\)，而是用极小的向量 \(v\in\mathbb{R}^{u}\) 通过固定随机投影生成 \(R\)。

\[
W' = W + U\Sigma R(v)V^T
\]

\[
R(v)=\sum_{i=1}^{u}v_iP_i
\]

其中：

\[
P_i\in\mathbb{R}^{r\times r}
\]

是 fixed random matrix，只有 \(v\) 是 trainable。若 \(L\) 层、每层 \(q\) 个模块、每个 trainable vector 被 \(n_{\text{tie}}\) 个模块共享，则：

\[
P_{\text{TinyLoRA}}
=
\left\lceil\frac{Lq}{n_{\text{tie}}}\right\rceil u
\]

当 \(n_{\text{tie}}=Lq\) 且 \(u=1\) 时，TinyLoRA 可以降到单个可训练参数。

TinyLoRA 和 LoRA-XS 的关系：

| Method | Trainable object | Per-module trainable params | 关键差异 |
|---|---|---:|---|
| LoRA-XS | \(R\in\mathbb{R}^{r\times r}\) | \(r^2\) | 直接训练完整小矩阵 |
| TinyLoRA | \(v\in\mathbb{R}^{u}\) | \(u\) | 用 fixed random projection 生成 \(R\) |
| TinyLoRA tied | shared \(v\) | \(\lceil Lq/n_{\text{tie}}\rceil u\) | 多模块共享同一个 tiny update |

重要限制：

- TinyLoRA paper 的强结果主要来自 **RLVR / GRPO + math reasoning**，不是 GLUE SFT；
- 因此不能直接用 TinyLoRA 的 GSM8K 结果证明它在 GLUE 分类任务上优于 LoRA-XS；
- 本蓝图中 TinyLoRA 应作为 **低参数极限实验** 与 **fixed random projection ablation**，而不是替代 GLUE 主线。

---

## 3. 梯度视角：LoRA-XS 的关键理论验证

LoRA-XS 的理论重点不是简单减参，而是：

> 如何选择一个低维子空间，使得 fine-tuning gradient 在该子空间内仍能近似 full fine-tuning update？

定义低维更新空间：

\[
S^r_{A,B}=\{AXB^T:X\in\mathbb{R}^{r\times r}\}
\]

其维度为：

\[
\dim(S^r_{A,B}) = r^2
\]

对 full gradient \(G\) 的投影为：

\[
p_{A,B}(G)=A[A^TGB]B^T
\]

这个式子有明确成立条件：

\[
A^TA=I,\qquad B^TB=I
\]

也就是 \(A\) 和 \(B\) 的列必须是标准正交列。若 \(A,B\) 只有满列秩但不是标准正交列，则正交投影应写成：

\[
p_{A,B}(G)
=
A(A^TA)^{-1}A^TGB(B^TB)^{-1}B^T
\]

若 \(A\) 或 \(B\) 数值上病态或不满秩，则用 Moore-Penrose pseudo-inverse：

\[
p_{A,B}(G)
=
A(A^TA)^{+}A^TGB(B^TB)^{+}B^T
\]

因此在 LoRA-XS 中需要区分两件事：

- 若计算梯度能量捕获率，建议使用 SVD 的正交基 \(U_r,V_r\)；
- 若使用 \(A=U_r\Sigma_r\)，则 \(A\) 的列通常不是标准正交列，不能直接套用 \(A(A^TGB)B^T\)，必须使用 Gram correction。

如果实际 full update：

\[
\Delta W = hG_1+hG_2+\cdots+hG_k
\]

接近 \(S^r_{A,B}\)，则 constrained update 可以近似 full fine-tuning：

\[
W + hp_{A,B}(G_1)+\cdots+hp_{A,B}(G_k)
= W + p_{A,B}(\Delta W)
\]

如果：

\[
\Delta W\in S^r_{A,B}
\]

则：

\[
p_{A,B}(\Delta W)=\Delta W
\]

### 3.1 梯度能量捕获率

为了验证 LoRA-XS 的 SVD basis 是否有效，建议记录：

\[
\rho_{\text{grad}}
=
\frac{\|U_rU_r^TG V_rV_r^T\|_F^2}{\|G\|_F^2}
\]

其中：

- \(G\)：某个 target weight 的 full gradient；
- \(U_r,V_r\)：来自 pretrained weight \(W\) 的 top-\(r\) SVD basis；
- \(\rho_{\text{grad}}\) 越高，说明 LoRA-XS 的 SVD 子空间越能捕获 gradient update。

### 3.2 低维梯度范数

LoRA-XS 中 \(R\) 的梯度可以写成：

\[
\nabla_R \mathcal{L}
= \Sigma_r U_r^T G V_r
\]

记录：

\[
G_R = \|\nabla_R\mathcal{L}\|_F
\]

并和 LoRA 的 \(\nabla_A,\nabla_B\) 范数比较：

\[
G_A=\|\nabla_A\mathcal{L}\|_F,\qquad
G_B=\|\nabla_B\mathcal{L}\|_F
\]

### 3.3 SVD basis ablation

做三个初始化对比：

| Init Type | 说明 |
|---|---|
| SVD of W | LoRA-XS 原版，使用 pretrained weight 的 top singular vectors |
| Random basis | 随机 frozen basis，只训练 \(R\) |
| Bottom singular vectors | 使用 bottom singular vectors，测试 top singular vectors 是否更有用 |

这部分用于证明 LoRA-XS 不是“随便冻结两个矩阵”，而是 SVD basis 确实有 gradient-aligned 的作用。

---

## 4. 实验设置

### 4.1 推荐主实验：GLUE / RoBERTa

主实验建议用 RoBERTa 系模型和 GLUE 子任务，因为 LoRA-XS paper 的主要表格包含 RoBERTa-large 在 GLUE 上对 LoRA、VeRA、LoRA-XS 的对比。

#### 最小可复现实验

| 项目 | 设置 |
|---|---|
| Base model | RoBERTa-base |
| Tasks | SST-2, MRPC, CoLA, QNLI |
| Seeds | 1 |
| 目标 | 跑通 GLUE 主线三种方法，验证性能、参数量、时间、显存、代码复杂度统计流程 |

#### 正式复现实验

| 项目 | 设置 |
|---|---|
| Base model | RoBERTa-large |
| Tasks | SST-2, MRPC, CoLA, QNLI, RTE, STS-B |
| Seeds | 3，若追求 paper-level 则 5 |
| 目标 | 对齐 LoRA-XS paper 的 GLUE 对比设置 |

---

### 4.2 Target modules 设计

为避免“不公平比较”，建议设置两套实验。

#### Setting A：Same-target fair comparison

GLUE 主线三种方法都只加到同样的模块：

\[
\mathcal{M}=\{W_q,W_v\}
\]

目的：比较方法本身，而不是 target modules 数量差异。

| Method | Target Modules | Rank |
|---|---|---|
| LoRA | q, v | 4, 8, 16 |
| VeRA | q, v | 64, 128, 256 |
| LoRA-XS | q, v | 4, 8, 12, 16, 20, 25 |

#### Setting B：Paper-style comparison

参考 LoRA-XS paper，LoRA-XS 使用更多 target modules：

\[
\mathcal{M}=\{W_q,W_v,W_o,FC_1\}
\]

| Method | Target Modules | Rank |
|---|---|---|
| LoRA | q, v | 8 |
| VeRA | q, v | 256 |
| LoRA-XS | q, v, attention output, FC1 | 4, 8, 12, 16, 20, 25 |

Setting B 用来复现 paper 的主要结论；Setting A 用来保证公平性。

#### Setting C：Coverage-matched control

为了回答“LoRA-XS 是否只是因为插入了更多 module 才更好”，需要增加 coverage-matched control：

\[
\mathcal{M}=\{W_q,W_v,W_o,FC_1\}
\]

| Method | Target Modules | Rank / Budget |
|---|---|---|
| LoRA | q, v, attention output, FC1 | parameter-matched rank |
| VeRA | q, v, attention output, FC1 | parameter-matched rank |
| LoRA-XS | q, v, attention output, FC1 | 4, 8, 12, 16, 20, 25 |

解释逻辑：

- Setting A：回答“方法本身是否更好”；
- Setting B：回答“paper-style 设置是否能复现”；
- Setting C：回答“模块覆盖范围是否是主要原因”。

---

### 4.3 Parameter-matched 实验

为了避免 rank 不同导致比较偏差，增加 parameter-matched 对比：

目标：让各方法的 trainable parameters 尽量接近。

例如设定预算：

\[
P_{\text{budget}} \in \{10K, 50K, 100K, 500K, 1M\}
\]

对每个方法选择 rank，使：

\[
P_{\text{method}} \leq P_{\text{budget}}
\]

然后比较 performance：

\[
M(P_{\text{budget}})
\]

输出 Pareto 曲线：

\[
\text{Task Score} \quad vs \quad \text{Trainable Parameters}
\]

### 4.4 针对当前疑问的实验拆解

#### 问题 1：LoRA-XS 是否靠更多 module 超过 LoRA？

不能只看 paper-style Setting B。因为 LoRA-XS 在 GLUE 中加入 \(W_q,W_v,W_o,FC_1\)，而 LoRA 只加 \(W_q,W_v\)，这会引入 coverage confound。

必须报告三组结果：

| 对比 | 能回答的问题 | 结论判断方式 |
|---|---|---|
| Same-target | LoRA-XS 是否在同样模块上优于 LoRA | 若 LoRA-XS 仍更好，说明方法 / basis 有贡献 |
| Paper-style | 是否复现原论文设置 | 若更好，只能说明 paper-style 更好，不能单独归因于方法 |
| Coverage-matched | 更多模块是否解释主要收益 | 若 LoRA 加到同样模块后追平，说明覆盖范围是关键因素 |

#### 问题 2：LoRA-XS 更好到底好在哪？

需要拆成三个指标，而不是一句“更好”：

| 可能原因 | 对应实验 | 判断标准 |
|---|---|---|
| 方法本身更好 | Same-target, same-rank | 同样 \(q,v\) 下 score 更高 |
| 参数效率更高 | Parameter-matched / Pareto | 同等参数预算下 score 更高，或同等 score 下参数更少 |
| 覆盖范围更广 | Coverage-matched | LoRA 扩到同样模块后是否追平 |

报告中应该分别写：

\[
\text{Performance gain}
\neq
\text{Parameter efficiency gain}
\neq
\text{Coverage gain}
\]

#### 问题 3：多少 module \(\times\) layer 能保持效果可观？

做 module-layer trade-off sweep：

| 维度 | Sweep |
|---|---|
| Layer subset | top-25%, top-50%, all layers |
| Module subset | qv, qvo, qvo+fc1, all attention+MLP |
| Rank / budget | fixed rank 与 parameter-matched 两套 |

建议用以下保持效果标准：

\[
\text{Retention}
=
\frac{M_{\text{subset}}-M_{\text{base}}}
{M_{\text{all}}-M_{\text{base}}}
\]

若：

\[
\text{Retention}\geq 0.95
\]

则认为该 module-layer 配置保持了相对可观的效果。

#### 问题 4：fixed random matrix 改成 trainable random matrix 会怎样？

以 TinyLoRA 为例：

\[
R(v)=\sum_i v_iP_i
\]

原版 TinyLoRA 中 \(P_i\) 是 fixed random matrix，只训练 \(v\)。如果把 \(P_i\) 也设为 trainable，则：

- 可训练参数从 \(u\) 增加到 \(u+ur^2\)；
- 训练 loss 通常会更快下降或达到更低 loss；
- 但这不再是同一个参数预算，不能直接说“收敛更好”；
- 需要同时报告 convergence speed 和 trainable parameter count。

因此实验应包含：

| Variant | Trainable params per module | 目的 |
|---|---:|---|
| TinyLoRA-fixed-P | \(u\) | 原版低参数极限 |
| TinyLoRA-trainable-P | \(u+ur^2\) | 检查随机投影可训练后是否更快收敛 |
| LoRA-XS-direct-R | \(r^2\) | 判断 trainable-P 是否只是接近直接训练 \(R\) |

#### 问题 5：TinyLoRA 参数不断减少时结果是否会突变？

TinyLoRA 的 tying 因子控制的是“多少个 module 共享同一个 \(v\)”：

\[
P_{\text{TinyLoRA}}
=
\left\lceil\frac{mn}{n_{\text{tie}}}\right\rceil u
\]

其中 \(m\) 是每层 adapted modules 数量，\(n\) 是层数。需要注意：

- \(\frac{mn}{n_{\text{tie}}}=0\) 在数学上不成立；实验中把它定义为 **zero-update baseline**，即不训练 adapter，参数量为 0；
- \(n_{\text{tie}}=mn\)：全模型所有 adapted modules 共享一个 \(v\)，参数量为 \(u\)；
- \(n_{\text{tie}}=m\)：每层内所有 adapted modules 共享一个 \(v\)，参数量为 \(nu\)；
- \(n_{\text{tie}}=1\)：不共享，每个 module 一个 \(v\)，参数量为 \(mnu\)。这不是 LoRA，只是 TinyLoRA no-tying；若要对齐 LoRA-XS，需要比较 \(R\) 直接可训练的 \(mnr^2\)。

推荐实验表：

| Variant | \(n_{\text{tie}}\) | Trainable params | 用途 |
|---|---:|---:|---|
| No adapter | \(\infty\) / N/A | 0 | 判断 base model 分数 |
| TinyLoRA full-model tie | \(mn\) | \(u\) | 极限共享，最小非零参数 |
| TinyLoRA per-layer tie | \(m\) | \(nu\) | 每层一个 tiny update |
| TinyLoRA no-tie | 1 | \(mnu\) | 每个 module 一个 tiny update |
| LoRA-XS direct-R | 1 | \(mnr^2\) | 直接训练 \(R\)，TinyLoRA 的容量上界 |
| LoRA | 1 | \(\sum_{\ell,j}r(d_{in,j}+d_{out,j})\) | 经典 LoRA baseline |

观察指标：

\[
\Delta M_k=M_k-M_{\text{zero}}
\]

\[
\text{Retention}_k
=
\frac{M_k-M_{\text{zero}}}
{M_{\text{no-tie}}-M_{\text{zero}}}
\]

若 trainable params 从 \(mnu\rightarrow nu\rightarrow u\rightarrow 0\) 时，Retention 出现大幅下降，例如：

\[
\text{Retention}_{\text{per-layer}}\geq 0.9
\quad\text{but}\quad
\text{Retention}_{\text{full-tie}}\leq 0.5
\]

则说明参数减少到 full-model tying 时发生了重大性能变化；反之，如果 full-model tie 仍保留大部分提升，则说明任务对 tiny update 非常鲁棒。

当前代码中对应脚本：

```text
comparsion and evaluation experinment/experiments/parameter_tradeoff.py
comparsion and evaluation experinment/experiments/projection_formula_check.py
comparsion and evaluation experinment/experiments/toy_convergence.py
comparsion and evaluation experinment/experiments/tinylora_tying_sweep.py
```

---

## 5. 评估指标

### 5.1 任务性能指标

| Task | Metric |
|---|---|
| SST-2 | Accuracy |
| MRPC | Accuracy + F1 |
| CoLA | Matthews Correlation Coefficient, MCC |
| QNLI | Accuracy |
| RTE | Accuracy |
| STS-B | Pearson Correlation |

主结果报告：

\[
\text{Mean} \pm \text{Std}
\]

如果做 5 seeds，可与 paper 一致报告 median：

\[
\text{Median}_{s\in Seeds}(M_s)
\]

---

### 5.2 参数效率指标

#### 可训练参数量

\[
P_{\text{train}}
=\sum_{p\in\Theta_{\text{train}}}\text{numel}(p)
\]

#### 可训练参数比例

\[
R_{\text{train}}
=\frac{P_{\text{train}}}{P_{\text{total}}}
\]

#### Adapter checkpoint size

\[
S_{\text{adapter}}
= P_{\text{saved}}\times \text{bytes(dtype)}
\]

注意：

- LoRA 保存 \(A,B\)；
- VeRA 保存 trainable scaling vectors，shared frozen basis 可选择重新生成或作为 buffer 保存；
- LoRA-XS 最理想情况下只保存 \(R\)，加载时从 base model weight 重新计算 SVD；若保存 \(A,B\) buffer，则实际 checkpoint size 会增大。
- TinyLoRA 最理想情况下只保存 \(v\) 和 tying metadata；fixed random projection \(P_i\) 可由 seed 重建。若保存 \(P_i\) 或将 \(P_i\) 设为 trainable，则 checkpoint size 会显著增大。

#### 参数效率得分

\[
E_{\text{param}}
=
\frac{M_{\text{task}}}{\log_{10}(P_{\text{train}}+1)}
\]

或者报告相对 LoRA 的参数节省率：

\[
Save_{\text{param}}
=
1-\frac{P_{\text{method}}}{P_{\text{LoRA}}}
\]

---

### 5.3 计算资源指标

#### Peak GPU memory

记录 PyTorch 和系统层面两类显存：

\[
VRAM_{\text{alloc-peak}}
=\max_t \texttt{torch.cuda.memory\_allocated}(t)
\]

\[
VRAM_{\text{reserved-peak}}
=\max_t \texttt{torch.cuda.memory\_reserved}(t)
\]

同时记录：

\[
VRAM_{\text{nvidia-smi}}
\]

#### 训练时间

总训练时间：

\[
T_{\text{total}}=T_{\text{init}}+T_{\text{train}}+T_{\text{eval}}
\]

每步时间：

\[
T_{\text{step}}=\frac{T_{\text{train}}}{N_{\text{steps}}}
\]

吞吐量：

\[
\text{samples/sec}=\frac{N_{\text{samples}}}{T_{\text{train}}}
\]

\[
\text{tokens/sec}=\frac{N_{\text{tokens}}}{T_{\text{train}}}
\]

#### LoRA-XS SVD 初始化时间

LoRA-XS 需要单独记录：

\[
T_{\text{SVD}}
\]

SVD overhead：

\[
Overhead_{\text{SVD}}
=\frac{T_{\text{SVD}}}{T_{\text{train}}+T_{\text{SVD}}}
\]

---

### 5.4 复现接近度指标

如果 paper 分数为 \(M_{\text{paper}}\)，复现分数为 \(M_{\text{rep}}\)：

\[
\Delta_M=|M_{\text{rep}}-M_{\text{paper}}|
\]

建议标准：

| 项目 | 合格复现 | 较好复现 |
|---|---:|---:|
| GLUE accuracy / correlation | \(\leq 2\) points | \(\leq 1\) point |
| Trainable parameters | 必须一致或明确解释差异 | 必须一致 |
| Checkpoint size | \(\leq 15\%\) 误差 | \(\leq 10\%\) 误差 |
| Training time | \(\leq 25\%\) 误差 | \(\leq 15\%\) 误差 |
| Peak VRAM | \(\leq 20\%\) 误差 | \(\leq 10\%\) 误差 |

---

## 6. 实现代码对比

### 6.1 代码结构建议

```text
project/
  colab_glue_experiment.ipynb
  colab_run_glue.py
  models/
    common.py
    basis.py
    lora.py
    vera.py
    loraxs.py
    tinylora.py
    inject.py
    subspace.py
  model_settings/
    lora.py
    vera.py
    loraxs.py
    tinylora.py
  experiment_settings/
    glue_same_target.py
    glue_paper_style.py
    glue_coverage_matched.py
    tinylora_tying.py
  experiments/
    colab_glue_runner.py
    list_settings.py
    parameter_tradeoff.py
    projection_formula_check.py
    toy_convergence.py
    tinylora_tying_sweep.py
    smoke_test.py  # development-only，不作为最终实验入口
  scripts/
    run_glue_lora.sh
    run_glue_vera.sh
    run_glue_loraxs.sh
    run_tinylora_toy.sh
  results/
    raw_logs/
    tables/
    figures/
```

当前仓库已补充的最小可执行代码位于：

```text
colab_glue_experiment.ipynb
colab_run_glue.py

comparsion and evaluation experinment/
  models/
    common.py
    basis.py
    lora.py
    vera.py
    loraxs.py
    tinylora.py
    adapters.py
    inject.py
    subspace.py
  model_settings/
    lora.py
    vera.py
    loraxs.py
    tinylora.py
  experiment_settings/
    glue_same_target.py
    glue_paper_style.py
    glue_coverage_matched.py
    tinylora_tying.py
  experiments/
    colab_glue_runner.py
    list_settings.py
    parameter_tradeoff.py
    projection_formula_check.py
    toy_convergence.py
    tinylora_tying_sweep.py
    smoke_test.py  # development-only
```

组织原则：

- **model 粒度**：每种 adapter 的实现和默认设定各自独立成 `.py`，例如 `models/lora.py` 与 `model_settings/lora.py`；
- **experiment 粒度**：每个实验组合独立成 `.py`，只引用已有 model setting，不在实验脚本中重复写方法配置；
- **Colab 入口**：最终运行接口可以使用 `colab_glue_experiment.ipynb` 或 `colab_run_glue.py`，通过 Hugging Face `datasets.load_dataset("glue", task)` 下载公开 GLUE 数据；
- `models/adapters.py` 只作为兼容导出入口，避免旧脚本失效。

### 6.2 方法实现要求

#### LoRA

实现要求：

- 低秩矩阵 \(A,B\)；
- base weight frozen；
- 支持 target modules 配置；
- 支持 merge/unmerge；
- 统计 trainable parameters。

预计实现难度：低。

#### VeRA

实现要求：

- 共享 frozen low-rank matrices；
- 每层 / 每模块有 trainable scaling vectors；
- 确认 shared basis 不被 optimizer 更新；
- 确认 checkpoint 只保存 trainable scaling vectors，或明确记录是否保存 frozen basis。

预计实现难度：中等。

#### LoRA-XS

实现要求：

- 对每个 target linear weight 执行 truncated SVD；
- 构造 frozen \(A=U_r\Sigma_r\)、\(B=V_r^T\)；
- 只注册 \(R\in\mathbb{R}^{r\times r}\) 为 trainable parameter；
- 记录 \(T_{\text{SVD}}\)；
- 处理 checkpoint：只保存 \(R\)，或保存 \(R+A+B\) 并注明存储口径；
- 注意 PyTorch `nn.Linear.weight` 的 shape 通常是 \((d_{out}, d_{in})\)。

预计实现难度：中等偏高。

#### TinyLoRA

实现要求：

- 复用 LoRA-XS 的 frozen SVD basis；
- 构造 fixed random projection tensor \(P\in\mathbb{R}^{u\times r\times r}\)；
- 只训练 \(v\in\mathbb{R}^{u}\)，并支持多个 module 共享同一个 \(v\)；
- 支持 ablation：`fixed_projection` vs `trainable_projection`；
- 统计 tying 后的真实 unique trainable parameters，而不是简单按 module 重复相加；
- 明确记录 random seed，因为 fixed projection 可由 seed 重建。

预计实现难度：中等。TinyLoRA 本身代码不难，难点在于 RLVR 训练框架、vLLM/merge 推理路径和 tiny update 的稳定性。

---

## 7. 代码重合度评估

### 7.1 方法专属代码行数

只统计 method-specific code，不统计训练框架、数据处理、评价代码。

\[
LOC_{\text{method}}
\]

建议统计：

```text
src/methods/lora.py
src/methods/vera.py
src/methods/loraxs.py
src/methods/tinylora.py
src/methods/subspace.py
```

### 7.2 公共代码复用率

\[
ReuseRate_m
=
\frac{LOC_{\text{common}}}
{LOC_{\text{common}}+LOC_{\text{method},m}}
\]

### 7.3 两两代码重合度

用 token Jaccard：

\[
Overlap(i,j)
=
\frac{|Tokens_i\cap Tokens_j|}{|Tokens_i\cup Tokens_j|}
\]

也可以用 normalized edit similarity：

\[
Sim_{\text{edit}}(i,j)=1-\frac{EditDistance(i,j)}{\max(|i|,|j|)}
\]

### 7.4 工程复杂度指标

| Metric | 说明 |
|---|---|
| Method LOC | 方法专属代码行数 |
| Files Changed | 改动文件数量 |
| Config Keys | 新增配置项数量 |
| Trainable Param Check Complexity | 参数冻结/训练检查复杂度 |
| Checkpoint Complexity | 保存和加载逻辑复杂度 |
| Debug Time | 从首次实现到稳定跑通所需时间 |

---

## 8. 结果表模板

### 8.1 主性能表

| Method | Model | Task | Rank | Target Modules | Seed | Score | Mean | Std |
|---|---|---|---:|---|---:|---:|---:|---:|
| LoRA | RoBERTa-large | SST-2 | 8 | q,v | 1 |  |  |  |
| VeRA | RoBERTa-large | SST-2 | 256 | q,v | 1 |  |  |  |
| LoRA-XS | RoBERTa-large | SST-2 | 16 | q,v,o,fc1 | 1 |  |  |  |
| TinyLoRA | Qwen2.5 / toy | GSM8K / toy | 2 | q,k,v,o,MLP | 1 |  |  |  |

### 8.2 参数与存储表

| Method | Rank | Target Modules | Trainable Params | Trainable % | Saved Params | Adapter Size MB | Buffer Size MB |
|---|---:|---|---:|---:|---:|---:|---:|
| LoRA | 8 | q,v |  |  |  |  |  |
| VeRA | 256 | q,v |  |  |  |  |  |
| LoRA-XS | 16 | q,v,o,fc1 |  |  |  |  |  |
| TinyLoRA | 2 | q,k,v,o,MLP |  |  |  |  |  |

### 8.3 显存和时间表

| Method | Rank | Batch | Seq Len | Peak Alloc GB | Peak Reserved GB | Step Time s | Total Train Time | Init Time | SVD Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 8 | 32 | 128 |  |  |  |  |  | 0 |
| VeRA | 256 | 32 | 128 |  |  |  |  |  | 0 |
| LoRA-XS | 16 | 32 | 128 |  |  |  |  |  |  |
| TinyLoRA | 2 | 32 | 128 |  |  |  |  |  |  |

### 8.4 实现复杂度表

| Method | Method LOC | Files Changed | Config Keys | Checkpoint Complexity | Debug Time h | Code Overlap with LoRA |
|---|---:|---:|---:|---:|---:|---:|
| LoRA |  |  |  | Low |  | 1.00 |
| VeRA |  |  |  | Medium |  |  |
| LoRA-XS |  |  |  | High |  |  |
| TinyLoRA |  |  |  | Medium-High |  |  |

### 8.5 梯度子空间表

| Method | Layer | Module | Rank | \(\rho_{grad}\) | \(\|\nabla_R\|_F\) | Comment |
|---|---|---|---:|---:|---:|---|
| LoRA-XS |  | q | 8 |  |  | SVD of W |
| LoRA-XS-random |  | q | 8 |  |  | Random basis |
| LoRA-XS-bottom |  | q | 8 |  |  | Bottom singular vectors |
| TinyLoRA-fixed-P |  | q | 2 |  |  | fixed random projection |
| TinyLoRA-trainable-P |  | q | 2 |  |  | trainable projection ablation |

---

## 9. 复现难度评分体系

定义总体复现难度：

\[
D
=
0.25D_{\text{impl}}
+
0.20D_{\text{compute}}
+
0.15D_{\text{metric}}
+
0.20D_{\text{stability}}
+
0.20D_{\text{paper-gap}}
\]

每项 1--5 分：

| 分数 | 含义 |
|---:|---|
| 1 | 现成实现，几乎无难点 |
| 2 | 少量工程改动 |
| 3 | 需要自定义模块和参数冻结检查 |
| 4 | 需要特殊初始化、特殊 checkpoint、较多调试 |
| 5 | 需要大算力、多 seed、复杂复现实验 |

预估难度：

| Method | Implementation | Compute | Metric | Stability | Paper Gap | Overall |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 1 | 2 | 1 | 2 | 1 | 1.4 |
| VeRA | 2.5 | 2 | 2 | 3 | 2.5 | 2.4 |
| LoRA-XS | 3.5 | 2.5 | 2.5 | 3 | 3.5 | 3.1 |
| TinyLoRA toy / code | 3 | 2 | 2 | 3 | 3 | 2.7 |
| TinyLoRA RLVR reproduction | 4 | 4.5 | 4 | 4 | 4 | 4.1 |
| Full comparison | 3.5 | 3 | 3 | 3.5 | 3.5 | 3.3--3.6 |

结论：

- LoRA：低难度 baseline；
- VeRA：中等难度，重点是 parameter sharing 和 scaling vectors；
- LoRA-XS：中等偏高，重点是 SVD 初始化、梯度子空间解释、checkpoint 存储口径；
- TinyLoRA：代码层面是 LoRA-XS 的压缩扩展，但 paper-level 复现依赖 RLVR / GRPO / reasoning evaluation，难度高于 GLUE 主线；
- 四者完整对比：中等偏高；若包含 TinyLoRA paper-level RLVR 复现，则主要难点会转移到 RL 训练框架和 reasoning evaluation。

---

## 10. 时间计划

### 10.1 7 天最小可交付版本

| Day | 任务 |
|---:|---|
| Day 1 | 搭建统一训练框架，跑通 LoRA baseline |
| Day 2 | 实现 / 接入 VeRA，确认共享矩阵与 scaling vectors 正确冻结 / 训练 |
| Day 3 | 实现 LoRA-XS SVD 初始化和 \(R\) 训练逻辑 |
| Day 4 | 跑 SST-2、MRPC 单 seed；记录参数、显存、时间 |
| Day 5 | 跑 CoLA、QNLI 单 seed；补充 gradient capture 统计 |
| Day 6 | 统计代码 LOC、代码重合度、debug time、checkpoint size |
| Day 7 | 整理表格、画 Pareto 曲线、写复现难度报告 |

### 10.2 14 天标准版本

| 阶段 | 时间 | 输出 |
|---|---:|---|
| 统一训练框架 | 2 天 | trainer/evaluator/resource tracker |
| 主线三种方法实现 | 4 天 | lora.py / vera.py / loraxs.py |
| GLUE 4-task 3-seed | 4 天 | performance + resource tables |
| 梯度子空间实验 | 2 天 | \(\rho_{grad}\)、SVD/random/bottom ablation |
| 代码复杂度统计 | 1 天 | LOC、overlap、debug time |
| 报告整理 | 1 天 | final report + figures |

### 10.3 Paper-level 版本

需要 3--6 周：

- RoBERTa-large；
- 6 个 GLUE tasks；
- 5 seeds；
- LoRA-XS 多 rank；
- VeRA rank 256；
- 完整参数量、显存、训练时间、SVD overhead；
- 可选 LLaMA2/LLaMA3 commonsense reasoning 扩展。

---

## 11. 最终图表清单

至少输出以下图表：

1. **Task Score vs Trainable Parameters**
2. **Task Score vs Adapter Checkpoint Size**
3. **Peak GPU Memory vs Method**
4. **Training Time vs Method**
5. **SVD Initialization Overhead for LoRA-XS**
6. **Code LOC / Files Changed Bar Chart**
7. **Code Overlap Heatmap**
8. **Reproduction Difficulty Radar Chart**
9. **Gradient Energy Capture \(\rho_{grad}\) vs Rank**
10. **SVD / Random / Bottom Basis Ablation**

---

## 12. 推荐最终实验组合

最建议采用如下组合：

| Level | Model | Tasks | Methods | Seeds | 目标 |
|---|---|---|---|---:|---|
| Minimum | RoBERTa-base | SST-2, MRPC | LoRA, VeRA, LoRA-XS | 1 | 跑通流程 |
| Main | RoBERTa-large | SST-2, MRPC, CoLA, QNLI | LoRA, VeRA, LoRA-XS | 3 | 主对比 |
| Paper-aligned | RoBERTa-large | SST-2, MRPC, CoLA, QNLI, RTE, STS-B | LoRA, VeRA, LoRA-XS | 5 | 接近论文复现 |
| Theory ablation | RoBERTa-base/large | CoLA, QNLI | LoRA-XS SVD/random/bottom | 3 | 验证梯度子空间解释 |
| TinyLoRA code ablation | Toy / small local model | Matrix regression / small classification | TinyLoRA fixed-P, trainable-P, LoRA-XS-R | 3 | 验证 fixed random projection 与 trainable projection |
| TinyLoRA paper-style | Qwen2.5 Instruct | GSM8K / MATH | TinyLoRA, LoRA-XS, LoRA | 3 | RLVR 低参数极限复现 |

---

## 13. 最终判断标准

### 合格复现

- GLUE 主线三种方法都能跑通；
- 至少 2 个 GLUE tasks；
- 有 task score、trainable params、GPU memory、training time；
- 能解释 LoRA / VeRA / LoRA-XS 的方法差异。

### 较好复现

- 4 个 GLUE tasks；
- 3 seeds；
- 有 parameter-matched 对比；
- 有代码复杂度和代码重合度统计；
- 有 SVD initialization overhead；
- 有复现难度评分。

### 高质量复现

- RoBERTa-large + 6 GLUE tasks；
- 5 seeds；
- 与 paper 表格数字进行误差比较；
- 做 LoRA-XS SVD / random / bottom singular vectors ablation；
- 画完整 Pareto 曲线；
- 对 LoRA-XS 梯度子空间捕获率进行定量分析。

---

## 14. 一句话总结

本实验应聚焦于：

\[
\text{LoRA as baseline}
\]

\[
\text{VeRA as parameter sharing baseline}
\]

\[
\text{LoRA-XS as SVD-based gradient-subspace adaptation}
\]

\[
\text{TinyLoRA as ultra-low-parameter SVD-projection adaptation}
\]

最终用性能、参数量、显存、训练时间、SVD overhead、代码复杂度、代码重合度、target module 覆盖范围、梯度子空间指标和 tiny update 收敛曲线共同评估四者的复现难度与研究价值。
