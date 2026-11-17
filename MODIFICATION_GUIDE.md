# DSER++ Learned BASR V2：完整修订代码

本版基于 `MrLi002/DSER-QQ` 的提交 `983d929512d5af48c12960333438546e04a6daaa`，
修正上一版 Learned BASR 的粗对齐证据、直接检索监督和来源判断输入问题。
压缩包提供完整文本源码，不包含数据、模型权重或演示图片。
这是功能验证后的研究实现，尚未完成真实数据完整训练，不能宣称指标提升。

## 1. 本次修正

| 项目 | V1 | V2 |
|---|---|---|
| 残差偏移输入 | target、同坐标 source、事件统计 | target、E-PCD 候选采样聚合后的 source、事件统计、采样有效性 |
| 检索器训练 | 主损失及来源/门控损失的间接梯度 | 另加不受门控和来源权重抑制的检索 RGB 损失 |
| 来源判断输入 | 潜在特征及未经校准的特征差等 | 加入两端检索 RGB、当前尺度 RGB 预测及两组 RGB 差，去掉未经校准的潜在特征绝对差 |
| DCN 调制量 | 未使用 | 对 detach 的原始调制量做可学习映射，转为归一化弱先验 |
| 监控 | 损失 | 另记录来源使用量、门控、有效使用量、有效支持比例与 RGB 误差；保存 CSV |
| V1 权重兼容 | 无版本区分 | 只能通过 --pretrained 迁移，禁止把 V1 当 V2 直接推理或续训 |

第一阶段、第二阶段的原计算保持不变。第二阶段只通过可选返回值暴露 offset 和 modulation。
本次没有同时修改事件掩码、数据读取、原评测步长或其他论文与公开实现的差异。

## 2. 文件对应

| 文件 | 作用 |
|---|---|
| model/selective_basr.py | 两阶段采样、难度/top-k、RGB 证据、融合门控、直接检索损失和监控量 |
| model/blocks.py | FeaTNet 可选返回各尺度 offset 和 modulation |
| model/ourNet.py | 串联新增模块；保留 original 与 learned 模式 |
| model/trans_sys.py | 给两处 BASR 传递对应尺度的采样和调制信息 |
| model/checkpoints.py | 原版/V1/V2 识别、迁移、严格主干检查及推理参数恢复 |
| train_bsergb.py | 新增损失、日志、CSV 与训练配置 |
| model/ourModel.py | 权重加载；推理时排除 padding |
| interpolation.py | 按 checkpoint 恢复结构参数并评测 |

`model/patch_attention.py` 保留原 BASR。
`model/loss.py` 保留原损失，新增损失集中在 `selective_basr.py`。
V2 相对 V1 的逐行修改见 `changes_from_v1.patch`；相对原仓库的修改见 `changes_from_original.patch`。

## 3. 关键机制

### 3.1 难度预测与选区

难度头输入端点图像、`Ft.detach()` 和左右事件统计，预测 `[B,1,H,W]` 的误差图。
训练目标为 `mean_c(abs(Ft.detach() - GT))`，使用像素 Smooth L1 和 patch 平均误差 L1 监督。
GT 只参与损失，不进入网络 forward 或 top-k。默认每张图独立选取难度最高的 25% patch，patch 为 32。
`--repair-ratio 0` 关闭局部修复，`1` 让所有有效 patch 参与检索。
固定预算不意味着每张图都有同样多错误，是否使用外部支持仍由门控与来源权重决定。

### 3.2 先粗采样，再预测残差

E-PCD 返回：

```python
return out, {
    'offsets': [off_half, off_quarter, off_eighth],
    'modulation': [m_half, m_quarter, m_eighth],
}
```

第三阶段两处 BASR 分别使用 quarter、half 尺度的候选。
每个候选位置按 torchvision DCN 的 `(dy, dx)` 通道顺序计算：

```text
coarse_position = target_position + kernel_position + detached_epcd_offset
```

第一遍采样在这些位置检索 source 特征，按匹配注意力聚合，得到目标坐标网格的 `coarse_aligned`。
残差头输入：

```python
cat(target, coarse_aligned, temporal, coarse_valid)
```

然后在 `coarse_position + bounded_residual` 上第二遍采样。
不使用未经采样的同坐标 source 预测残差，也不把九个 offset 平均成光流。
聚合仅统计有效候选；无效候选不会参与注意力。
默认残差范围为该尺度的 +/-2 特征像素。若 E-PCD 候选完全偏离，局部残差不能保证找回远处支持。

采样只对选区内目标位置执行，默认每块处理 1024 个位置。两遍采样会增加计算，分块不消除训练时全部反向图的存储成本。
E-PCD offset 和 modulation 在该读取分支中 detach；这并不等于冻结整个前两阶段。

### 3.3 DCN 调制量作为弱先验

原代码的 m 是未经过 sigmoid 的卷积输出，可能为负，不能直接取 log 或当成可见性概率。
V2 使用：

```python
log_prior = log_softmax(modulation_projection(m.detach()), dim=1)
alpha = sigmoid(prior_logit)   # 初始为 0.1，可学习
logits = cosine_similarity / temperature + alpha * log_prior
```

再施加候选有效性掩码。该先验只表示学习到的候选偏好，不是真实可见性或已校准置信度。
`--disable-modulation-prior` 可以禁用它，评估是否有帮助。

### 3.4 来源判断使用明确的 RGB 证据

来源头输入包括：目标/两端支持特征、事件统计、难度、采样有效性和匹配分数，以及：

```python
color0, color1, base_rgb,
abs(color0 - base_rgb), abs(color1 - base_rgb)
```

其中 `base_rgb` 是当前尺度 decoder 的预测，不是 GT。
删除 V1 中的 `abs(target_feature - value(source_feature))`，避免把未经约束的潜在空间差当作误差证据。

输出三个来源权重及修复门控：

```python
support = w0 * support0 + w1 * support1 + wr * target_feature
delta = selected_low * sigmoid(gate_logits) * (support - target_feature)
output = base_high + selected_high * project(upsample(delta))
```

越界和 padding 支持被排除；两端均无有效候选时保留分支权重为 1。
输出投影仍为零初始化，避免新局部模块一开始大幅扰动 backbone。
直接检索损失绕开这个零投影，因此不需要等待它变为非零才开始训练匹配。
选区外不写入局部残差，但后续全局 decoder 仍可能传播修改，最终 RGB 不保证选区外逐像素完全不变。

### 3.5 直接检索光度损失

新增项独立于 gate、w0/w1/wr 和 to_high：

```python
error_i = mean_c(sqrt((retrieved_rgb_i - GT)**2 + 1e-6) - 1e-3)
loss_retrieve = mean_selected_valid(min_valid_side(error_0, error_1))
```

该分支不 detach 检索 RGB；只对选区且至少一端有效的位置计算。
无效方向不参与 min，两端均无效的位置跳过，空区域损失为零。
它能在初始化时给 Q/K、采样残差头、注意力温度和调制先验映射提供梯度，不依赖来源/门控头和输出投影。
`value` 投影仍主要由特征重建与来源/门控路径学习，RGB 损失不直接训练该投影。

route/gate 的标签继续在 no_grad 中构造，避免通过改变标签逃避监督。
这些标签是 RGB 支持有用性的代理，不是真实遮挡标签。
min 通常主要训练获胜方向；两端都不可见时也未必存在正确外观，因此本版本使用 Charbonnier 和较小权重，仍需真实数据验证。

### 3.6 事件时间统计

左右片段分别提取 log 活动量、加权平均 bin 时间和时间方差，再加左右活动平衡，共 7 通道。
这些是 abs(voxel) 的代理统计：存在极性抵消，片段时间独立归一化。
不能把它们直接当成真实事件计数、物理事件率或遮挡方向。

## 4. 损失与监控

总损失保留原有四项，再加：

| 新增项 | 默认系数 | 作用 |
|---|---:|---|
| difficulty | 1.0 | 难度图监督 |
| route | 0.05 | 三来源软目标监督 |
| gate | 0.05 | 支持改善程度监督 |
| retrieve | 0.1 | 不受门控抑制的直接 RGB 检索监督 |

系数为起始配置，未经过验证集调参。
训练输出及 `output-dir/epoch_metrics.csv` 记录：

| 指标 | 含义 |
|---|---|
| source_usage | w0+w1 的平均值 |
| gate_mean | sigmoid(gate_logits) 的平均值 |
| effective_usage | gate*(w0+w1) 的平均值 |
| support_valid | 选区中至少有一端有效支持的比例 |
| retrieval_error | 最佳有效方向的 RGB L1 误差 |
| base_error | 当前尺度 RGB 预测在相同统计区域的 L1 误差 |
| oracle_improvable | 最佳检索 RGB 误差小于当前预测误差的位置比例，仅训练监控 |

除 support_valid 外，以上平均只在选区且至少一端有效的位置统计，排除无效位置对均值的污染。
两个 BASR 等权汇总，再按训练 batch 平均。低 gate 或低 source_usage 本身不等于塌缩：应结合有效支持比例、检索误差及可改善比例判断。
CSV 是每 epoch 一行，包含损失、监控量、global_step 与学习率。

## 5. 使用命令

使用已经能运行原项目的环境即可。新增代码使用 PyTorch 标准算子，仍依赖原有 torchvision DCN。
较小依赖清单为 `requirements_basr.txt`；选择匹配的 torch/torchvision CUDA 构建。

从 DSER++ 原版权重，或上一版 Learned BASR 权重迁移：

```bash
python train_bsergb.py --data-root "/home/yuaobo/桌面/ljh/dataset/3_TRAINING" --pretrained checkpoints/bsergb_eventaid.pth --basr-mode learned --epochs 20 --lr 1e-4 --batch-size 1 --crop-size 256 --skips 1 3 --repair-ratio 0.25 --lambda-retrieve 0.1 --output-dir train/bsergb_learned_basr_v2
```

从头训练：

```bash
python train_bsergb.py --data-root "/home/yuaobo/桌面/ljh/dataset/3_TRAINING" --basr-mode learned --epochs 40 --lr 1e-4 --batch-size 1 --crop-size 256 --skips 1 3 --output-dir train/bsergb_learned_basr_v2_scratch
```

断点续训 V2：

```bash
python train_bsergb.py --data-root "/home/yuaobo/桌面/ljh/dataset/3_TRAINING" --resume train/bsergb_learned_basr_v2/last.pth --basr-mode learned --epochs 20 --batch-size 1 --crop-size 256 --skips 1 3 --repair-ratio 0.25 --lambda-retrieve 0.1 --output-dir train/bsergb_learned_basr_v2
```

保持原先总 epochs、数据设置、结构配置和损失系数，以延续学习率调度与实验条件。
V1 转 V2 必须使用 --pretrained；--resume 和直接 V2 推理会拒绝 V1 权重。
迁移时重建两处 BASR；保留匹配 backbone 和已有难度头。原版 DSER++ 则同时新建难度头。
同版本 V2 的 --pretrained、--resume 及推理正常加载全部匹配参数。
日志显示迁移版本及丢弃/新增键数量；非 BASR 主干不匹配仍报错，默认不需要 --allow-partial-pretrained。

测试：

```bash
python interpolation.py --dataset bsergb --data_root_path "/home/yuaobo/桌面/ljh/dataset/1_TEST" --checkpoints_path train/bsergb_learned_basr_v2/last.pth --save_root_path eval/learned_basr_v2
```

完整训练 checkpoint 会自动恢复 bins、patch、选区比例及消融开关；纯 state_dict 不包含实验配置。
旧 original 权重仍可用 --basr-mode original 评测，旧 learned V1 需用 V1 代码评测或先迁移训练为 V2。

## 6. 消融开关

| 参数 | 用途 |
|---|---|
| --basr-mode original | 原 BASR 对照 |
| --lambda-retrieve 0 | 去掉直接检索损失 |
| --repair-residual-radius 0 | 关闭候选残差微调 |
| --disable-modulation-prior | 不使用 DCN 调制先验 |
| --disable-event-temporal | 去掉显式事件时间统计；其余事件网络仍保留 |
| --disable-alignment-candidates | 围绕同坐标 3x3 位置检索，并同时禁用不再对应的调制先验 |
| --repair-ratio 1 | 全部有效 patch 参与检索 |
| --repair-ratio 0 | 不执行局部修复 |

如果单独研究 E-PCD 位置的作用，应让两组都禁用调制先验，避免混入先验变化。
同坐标消融的搜索范围更小，不宜解释为同等范围下的纯可靠性对照。

## 7. 验证范围

```bash
python -m unittest discover -s tests -v
```

验证环境为 Python 3.12、PyTorch 2.5.1+cpu、torchvision 0.20.1+cpu、NumPy 1.26.4、einops 0.8.1。
语法保持 Python 3.10 兼容。
本次执行记录见 VALIDATION.md，包括初始化时直接检索梯度、粗对齐输入、RGB 证据、先验有效性、边界回退与权重迁移。
没有在真实 BS-ERGB 上完整训练，没有验证 GPU/AMP/多卡运行或指标提升。
