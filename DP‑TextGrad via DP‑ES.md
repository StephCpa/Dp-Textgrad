# DP‑TextGrad via DP‑ES：差分隐私进化策略改造方案

## 目标与动机
- **问题背景**：TextGrad 当前通过 “前向 LLM → 评价 LLM（产生文本梯度）→ 优化器 LLM” 的单轨循环更新变量，但文本梯度是自然语言，无法直接执行传统 DP‑SGD 中的 L2 裁剪与噪声注入。
- **核心想法**：将 TextGrad 的单点更新替换为 **差分隐私进化策略（DP‑Evolution Strategy, DP‑ES）**。我们维护一批候选文本，对它们进行评估、选择、变异，整个演化过程的敏感操作（尤其是依赖私有数据的评分）经过差分隐私化。
- **隐私对象**：敏感数据可能来自任务输入（如含隐私数据的提示、用户日志）、裁判/奖励模型使用的私有数据，或候选文本本身携带的敏感内容。目标是在保证 ε, δ 隐私约束下，提升任务表现。

## 整体架构

```
初始化候选 population
└─ 每轮迭代：
   1. 变量展开：把父代变量转换为候选（复用 TextGrad 的 Variable API）
   2. 生成反馈 / 评估：
      a. 任务评价（loss / reward）→ DP 处理
      b. 文本梯度提示（可选）→ 作为变异参考
   3. 差分隐私化评分：
      a. 裁剪（Clipping）
      b. 噪声注入（Gaussian / Laplace）
   4. 差分隐私化选择：
      a. 噪声 Top‑k / 指数机制选父代
   5. 变异 & 重生成：
      a. LLM 变异（提示同 TextualGradientDescent）
      b. 规则扰动 / crossover 组合
      c. 生成下一代 population
   6. 预算会计与终止检查
返回最优 / 最后候选
```

### 关键模块映射
| 模块 | 作用 | 与现有 TextGrad 对应 | 新增/改造要点 |
| --- | --- | --- | --- |
| `PopulationManager` | 维护候选 Variable 集合 | 无直接对应 | 支持批量候选、父代追踪 |
| `DPScorer` | 执行任务评估并 DP 处理 | `dp_textgrad.loss` | 包装现有 Loss Module，实现裁剪与噪声 |
| `DPSelector` | 根据 DP 评分选父代 | 无 | 实现 Noisy Top‑k / 指数机制 |
| `MutationEngine` | 生成新候选 | `TextualGradientDescent` | 扩展为批量提示 / crossover |
| `PrivacyAccountant` | 跟踪 ε, δ | 无 | RDP/Moments accountant |
| `EvolutionController` | 调度主循环 | `Optimizer` 基类 | 继承自 `Optimizer` 或新抽象 |

## 差分隐私机制设计

### 1. 敏感度控制
- **评分裁剪**：对每个候选的 raw score `s` 进行截断：`s_clipped = clamp(s, -C, C)` 或使用平滑函数（tanh/sigmoid）。`C` 由任务或历史统计决定。
- **批量裁剪**：若一次评估多个候选，可将向量 `s` 按 L2 或 L∞ 裁剪。
- **强化反馈**：文本文本梯度可作为私有数据的函数，若由敏感判据生成，也应裁剪其“强度”。做法是限制反馈长度/关键性，并通过温度采样或加噪。

### 2. 噪声注入
- **Gaussian 机制**：对裁剪后的得分加噪：`s_dp = s_clipped + N(0, σ^2)`，其中 `σ ≥ C·sqrt(2 log(1.25/δ))/ε`（经典 DP 界），也可使用 zCDP / RDP 公式。
- **指数机制**：直接通过指数机制进行选择，避免显式加噪，但需要计算 `exp(ε·utility/(2Δu))`，其中 Δu 为敏感度（由裁剪保证）。
- **文本反馈加噪**（可选）：对于直接暴露文本的步骤，可在提示中插入噪声 token，或使用 Dropout 指令降低对单个样本的敏感依赖。

### 3. DP 选择策略
- **Noisy Top‑k**：为每个候选得分加入 Gumbel 噪声后取 Top‑k，带来 (ε,0) 的隐私保证（本质为指数机制）。
- **Proportional Sampling**：根据 `exp(ε s_dp / (2C))` 对候选采样，可得到明确的 DP 界。
- **父代多样性**：可结合 tournament 选择 + 噪声，降低单个得分对结果的影响。

### 4. 隐私会计
- **次数追踪**：每次调用 DPScorer、DPSelector 记一次 `mechanism_call`，记录对应 ε, δ。
- **RDP/Moments**：实现基本 RDP 的合成规则：`ε_total = sqrt(2 log(1/δ) Σ ε_i^2)` 或使用现成公式。《Moments Accountant》可支持更多轮数。
- **终止条件**：若累计 ε 超过预算，提前结束或降低迭代强度（减少候选数/减少判分频率）。

## 演化策略细化

### Population 初始化
- **随机启动**：直接复制原变量多份，使用 `MutationEngine` 生成多样扰动（指令变体、随机拆词、模板交换）。
- **历史种子**：若已有历史版本，可混入 Warm Start。

### 候选变异
- **LLM 变异**：复用 `TextualGradientDescent` 提示，但把输入改为“父代 + 反馈 + 变异类型标签”，并生成多个候选。可采用多温度采样或 beam search。
- **交叉**：对两个候选进行段落级合并，或让 LLM 读取两份文本输出融合版本。
- **规则扰动**：定义轻量操作（同义词替换、模板插入、调参数），供快速探索。

### 评价与反馈
- **Loss 模块**：继续使用 `TextLoss` / `MultiFieldEvaluation` 等，但要封装在 `DPScorer` 中。
- **辅助梯度**：可选地保持 TextGrad 的梯度提示，用作启发式：反馈文本本身不直接 DP，但生成时需使用非敏感提示（或 DP 化损失）。如果反馈能泄露敏感信息，需要加截断 + 噪声。

### 控制流程
1. `PopulationManager` 输出当前候选列表：`List[Variable]`
2. `DPScorer` 对候选批量评估 → 返回 `(scores_dp, noise_metadata)`
3. `DPSelector` 使用 DP 评分选出父代索引
4. `MutationEngine` 基于父代生成下一代
5. `PrivacyAccountant` 更新预算；若超限则终止
6. `EvolutionController` 决定是否早停（任务达标或预算枯竭）

## 与现有 TextGrad 的集成
- **变量层面**：沿用 `Variable` 数据结构，新增可选字段：`parent_id`, `mutation_metadata`, `dp_score`, `noise_sigma`.
- **引擎层面**：继续使用 `engine` 抽象，允许不同阶段复用/切换模型。
- **日志**：在 `logger` 中新增 DP‑ES 专用字段，记录每轮噪声参数、ε 使用情况等，方便审计。
- **配置**：新增 `dp-textgrad/config_dp.py`（或扩展 `config.py`），集中管理人口规模、裁剪常数、总体 ε/δ、 accountant 类型等。

## 伪代码示例

```python
population = PopulationManager.init_from_variable(init_var, size=K)
accountant = PrivacyAccountant(target_epsilon, target_delta)

for t in range(max_iters):
    scores, metadata = dp_scorer.evaluate(population)
    accountant.consume(metadata['privacy_cost'])
    if accountant.exceeds_budget():
        break

    parents = dp_selector.select(population, scores, metadata)
    population = mutation_engine.generate_next(parents, feedback=metadata.get('feedback'))

best = population.best_member()  # 可按去噪后得分或实测评估
return best
```

## 待补充的实验与评估
- **隐私验证**：模拟插入/删除敏感样本，验证输出差异是否符合理论界。
- **性能折衷**：与非 DP TextGrad 对比，衡量性能随 ε、人口规模变化的趋势。
- **开销评估**：统计多候选评估带来的 token 成本，必要时引入候选分层或多臂带噪采样。

## TODO 清单（实现优先级）
1. **架构支撑**
   - [ ] 新增 `dp-textgrad/dp_es/population.py`：实现 `PopulationManager`
   - [ ] 新增 `dp-textgrad/dp_es/accountant.py`：RDP / Moments Accountant
   - [ ] 新增 `dp-textgrad/dp_es/scorer.py`：封装 DP 評分（继承或包装 `TextLoss`）
   - [ ] 新增 `dp-textgrad/dp_es/selector.py`：Noisy Top‑k / 指数机制实现
   - [ ] 新增 `dp-textgrad/dp_es/mutation.py`：批量 LLM 变异 + crossover 管理
   - [ ] 新增 `dp-textgrad/dp_es/controller.py`：主训练循环，继承 `Optimizer`

2. **与现有模块的接口**
   - [ ] 扩展 `Variable`：增加候选 ID、噪声记录字段（或新建 `Candidate` 数据类）
   - [ ] 在 `dp-textgrad/__init__.py` 暴露 `DPEvolutionStrategy` 入口
   - [ ] 更新 `dp-textgrad/config.py` / 新增 `config_dp.py` 支持 DP 参数
   - [ ] 复用/扩展 `dp-textgrad/optimizer/optimizer_prompts.py`，为批量变异提供提示模板

3. **测试与文档**
   - [ ] 单元测试：评分裁剪、噪声分布、accountant 累积、选择稳定性
   - [ ] 集成测试：端到端 DP‑ES 示例（小规模任务）
   - [ ] 性能基准：比较标准 TextualGradientDescent 与 DP‑ES 在不同 ε 下的性能
   - [ ] 文档/示例 notebook：解释使用方式、参数配置、隐私预算解读

4. **扩展探索（次优先级）**
   - [ ] 研究自适应噪声/自适应人口规模，优化隐私‑性能折衷
   - [ ] 引入多阶段机制：先进行非敏感预热，再切换到 DP‑ES
   - [ ] 结合隐私保护的回溯记忆（Gradient Memory 变体），在 DP 约束下累积安全反馈

> **提示**：以上 TODO 可按优先级逐步实现；先完成核心架构与 DPScorer，便能进行最小可行实验，再逐步加入高级功能。

### 当前代码落地情况
- `dp_textgrad/dp_es/feedback.py`：新增 `FeedbackSanitiser`，对评分阶段回传的文本或列表进行长度裁剪与字符过滤，避免隐私信息原样泄露。
- `dp_textgrad/dp_es/critique.py`：封装 `CritiquePipeline`，按照“生成 → DP 评分 → DP 选择”流程挑选批评，支撑 DP-TGD 架构。
- `dp_textgrad/dp_es/mutation.py`：`MutationEngine` 自动执行批评管线，把被选中的批评写入父代元数据，兼容旧版 `mutation_fn`。
- `dp_textgrad/dp_es/accountant.py`：新增 `AdvancedCompositionAccountant`，利用高级组合定理更节省 ε/δ 预算。

