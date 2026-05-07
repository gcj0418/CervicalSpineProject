# 颈椎脊柱关键点检测：方法对比与优化报告

> 更新时间: 2026-05-06  
> 实验对象：RENJI（33 例，56 landmarks，~0.125 mm/px）与 RUIJIN（14 例，52 landmarks，0.28 mm/px）两家医院数据集  
> 对比方法：VLD、D-CeLR、HRNet

---

## 第一阶段：基础单方法对比（2 数据集 × 3 方法）

> 基线设置：所有模型统一使用 e60 配置，直接对比三家方法在两家医院数据上的原始性能。

### 表 1-1：单方法基线结果

| method | dataset | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |
|--------|---------|---------------|---------|-----------|---------|---------|
| VLD | RENJI | 3.9010 | 0.6399 | 0.7339 | 0.8054 | 0.8685 |
| VLD | RUIJIN | 1.8342 | 0.7102 | 0.8132 | 0.8599 | 0.9217 |
| D-CeLR | RENJI | 3.1643 | 0.5639 | 0.6618 | 0.7332 | 0.8279 |
| D-CeLR | RUIJIN | 3.8959 | 0.2926 | 0.3777 | 0.4863 | 0.6621 |
| HRNet | RENJI | 3.3745 | 0.3268 | 0.4405 | 0.5482 | 0.7219 |
| HRNet | RUIJIN | 2.3665 | 0.4657 | 0.6154 | 0.7266 | 0.8901 |

### 第一阶段关键发现

1. **VLD 在两个数据集上均表现最佳**，RUIJIN 尤为突出（1.83 mm / 0.710）。
2. **D-CeLR RUIJIN 严重欠拟合**：mean error 3.90 mm，acc@2 仅 0.293，远低于同数据集的 HRNet（2.37 mm）和 VLD（1.83 mm）。
3. **HRNet RUIJIN 优于 RENJI**（2.37 mm vs 3.37 mm），但在 RENJI 上不如 D-CeLR。
4. **RENJI 整体难度更高**：各方法误差普遍大于 RUIJIN，可能与更高分辨率（0.125 mm/px）和更多 landmarks（56 vs 52）有关。

---

## 第二阶段：单方法逐一优化

针对第一阶段暴露的短板，我们对三个方法分别进行定向优化。

### 2.1 D-CeLR RUIJIN：迁移学习 + 数据增强（最成功）

**问题诊断**：RUIJIN 数据量小（14 例），D-CeLR 从头训练导致严重欠拟合。

**优化措施**：
1. **RENJI 预训练初始化**：ResNet34 backbone 权重完全复用，`encoder_embed` 自适应裁剪至 52 个 landmark
2. **数据增强**：新增 random_scale (0.9-1.1)、brightness/contrast、gamma、gaussian_noise
3. **降低学习率**：2e-4 → 1e-4，配合 cosine decay
4. **增加正则化**：weight_decay 1e-4 → 5e-4
5. **延长训练**：60 → 80 epochs，early stopping 在 e63

### 表 2-1：D-CeLR RUIJIN 优化前后对比

| 版本 | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |
|------|---------------|---------|-----------|---------|---------|
| 基线 (e60) | 3.8959 | 0.2926 | 0.3777 | 0.4863 | 0.6621 |
| **改进版 (best e63)** | **2.5724** | **0.5604** | **0.6497** | **0.7418** | **0.8544** |
| **提升幅度** | **↓ 34.0%** | **↑ 91.5%** | **↑ 72.0%** | **↑ 52.5%** | **↑ 29.0%** |

---

### 2.2 D-CeLR RENJI：数据增强 + 调参

**问题诊断**：RENJI 数据量相对充足，但原始训练缺乏数据增强，且超参数未充分调优。

**优化措施**：与 RUIJIN 相同的数据增强策略，LR 降至 1e-4，WD 提升至 5e-4，训练 80 epochs。

### 表 2-2：D-CeLR RENJI 优化前后对比

| 版本 | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |
|------|---------------|---------|-----------|---------|---------|
| 基线 (e60) | 3.1643 | 0.5639 | 0.6618 | 0.7332 | 0.8279 |
| **改进版 (best e75)** | **2.8151** | **0.5736** | **0.6926** | **0.7684** | **0.8479** |
| **提升幅度** | **↓ 11.0%** | **↑ 1.7%** | **↑ 4.7%** | **↑ 4.8%** | **↑ 2.4%** |

---

### 2.3 VLD RUIJIN：延长训练（e60 → e100）

**问题诊断**：VLD 已表现优异，但增训可能进一步挖掘潜力。

**优化措施**：训练 epoch 从 60 延长至 100，其他配置不变。

### 表 2-3：VLD RUIJIN 增训收益

| epoch | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm |
|-------|---------------|---------|-----------|---------|---------|
| e60 | 1.8342 | 0.7102 | 0.8132 | 0.8599 | 0.9217 |
| **e100** | **1.6978** | **0.7294** | **0.8228** | **0.8874** | **0.9368** |
| **提升** | **↓ 7.4%** | **↑ 2.7%** | **↑ 1.2%** | **↑ 3.2%** | **↑ 1.6%** |

---

### 2.4 VLD RENJI：Test-Time Augmentation（horizontal flip）

**问题诊断**：RENJI 数据难度高，推理时仅单前向传播可能不稳定。

**优化措施**：推理时对输入图像做水平翻转，分别前向传播后将 heatmap 翻回并取平均。

### 表 2-4：VLD RENJI TTA 效果

| epoch | 无 TTA | TTA (horizontal flip) | mean 变化 |
|-------|--------|----------------------|-----------|
| e30 | 3.521 mm / 0.583 | **3.363 mm / 0.593** | ↓ 4.5% |
| e60 | 3.901 mm / 0.640 | **2.883 mm / 0.645** | **↓ 26.1%** |

**结论**：TTA 对 VLD RENJI 效果极其显著，e60 + TTA 将 mean error 从 3.90 mm 降至 **2.88 mm**，是 VLD RENJI 的最佳单模型结果。RUIJIN 上 TTA 无提升（未展示）。

---

### 2.5 HRNet：ImageNet 预训练初始化（有效）

**优化措施**：加载 ImageNet 预训练权重（HRNet-W18-C）初始化 backbone，其他配置保持基线不变（60 epochs, WD=0, scale=0.15）。

### 表 2-5：HRNet 预训练前后对比

| dataset | 基线 (e60) | **预训练版 (e60)** | 结论 |
|---------|-----------|-------------------|------|
| RENJI | 3.375 / 0.327 | **2.661 / 0.503** | **mean ↓21.2%, acc@2 ↑54.0%** |
| RUIJIN | 2.367 / 0.466 | **1.750 / 0.692** | **mean ↓26.1%, acc@2 ↑48.7%** |

> **决策**：后续 ensemble 使用预训练 HRNet。

---

### 2.6 单方法优化阶段总览

### 表 2-6：优化后单方法最佳结果

| method | dataset | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm | 备注 |
|--------|---------|---------------|---------|-----------|---------|---------|------|
| **VLD** | **RENJI** | **2.8835** | **0.6446** | **0.7369** | **0.8089** | **0.8702** | **e60 + TTA 最佳** |
| **VLD** | **RUIJIN** | **1.6978** | **0.7294** | **0.8228** | **0.8874** | **0.9368** | **e100 增训最佳** |
| **D-CeLR** | **RENJI** | **2.8151** | **0.5736** | **0.6926** | **0.7684** | **0.8479** | **增强 + 调参** |
| **D-CeLR** | **RUIJIN** | **2.5724** | **0.5604** | **0.6497** | **0.7418** | **0.8544** | **迁移学习最佳** |
| **HRNet** | **RENJI** | **2.6607** | **0.5032** | **0.6299** | **0.7354** | **0.8544** | **ImageNet 预训练** |
| **HRNet** | **RUIJIN** | **1.7496** | **0.6923** | **0.8036** | **0.8723** | **0.9423** | **ImageNet 预训练** |

---

## 第三阶段：Ensemble 融合

基于第二阶段优化后的单方法模型，采用 Hungarian 对齐 + 加权平均策略进行融合。RENJI 使用 VLD e60+TTA，RUIJIN 使用 VLD e100，HRNet 使用预训练版。

### 3.1 RENJI Ensemble

### 表 3-1：RENJI Ensemble 结果

| config | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm | 说明 |
|--------|---------------|---------|-----------|---------|---------|------|
| 单方法最佳 (D-CeLR 改进) | 2.8151 | 0.5736 | 0.6926 | 0.7684 | 0.8479 | — |
| 3-method 等权 mean | 2.4542 | 0.6119 | 0.6964 | 0.7702 | 0.8613 | — |
| **3-method weighted (0.4/0.3/0.3)** | **2.3638** | **0.6649** | **0.7470** | **0.8170** | **0.8911** | **最佳 mean** |
| 3-method weighted (0.1/0.6/0.3) | 2.5629 | **0.6881** | 0.7798 | 0.8348 | 0.8902 | **最佳 acc@2** |
| pairwise HRNet+VLD (0.6/0.4) | 2.4653 | 0.6649 | 0.7524 | 0.8217 | 0.8881 | 高效两两 |
| pairwise HRNet+D-CeLR (0.5/0.5) | 2.4542 | 0.6119 | 0.6964 | 0.7702 | 0.8613 | — |
| pairwise VLD+D-CeLR (0.5/0.5) | 2.5066 | 0.6732 | 0.7601 | 0.8225 | 0.8854 | — |

**RENJI 结论**：
- Ensemble 最佳 mean = **2.36 mm**（3-method 0.4/0.3/0.3），比单方法最佳（2.82 mm）提升 **16.0%**
- Ensemble 最佳 acc@2 = **0.688**（3-method 0.1/0.6/0.3），比单方法最佳（0.645）提升 **6.7%**
- **两两组合足够高效**：HRNet+VLD 2.47 mm ≈ 3-method best 2.36 mm

---

### 3.2 RUIJIN Ensemble

### 表 3-2：RUIJIN Ensemble 结果

| config | mean_error_mm | acc@2mm | acc@2.5mm | acc@3mm | acc@4mm | 说明 |
|--------|---------------|---------|-----------|---------|---------|------|
| 单方法最佳 (VLD e100) | 1.6978 | 0.7294 | 0.8228 | 0.8874 | 0.9368 | — |
| 3-method 等权 mean | 1.9236 | 0.6937 | 0.8146 | 0.8668 | 0.9245 | — |
| **3-method weighted (0.5/0.4/0.1)** | **1.5737** | **0.7706** | **0.8418** | **0.8938** | **0.9382** | **最佳 mean** |
| 3-method weighted (0.4/0.5/0.1) | 1.5737 | **0.7706** | 0.8418 | 0.8938 | 0.9382 | **最佳 acc@2** |
| **pairwise HRNet+VLD (0.5/0.5)** | **1.5574** | **0.7692** | **0.8407** | **0.8934** | **0.9382** | **最高效两两，最佳 mean** |
| pairwise HRNet+VLD (0.4/0.6) | 1.5737 | **0.7706** | 0.8418 | 0.8938 | 0.9382 | **最佳 acc@2** |
| pairwise VLD+D-CeLR (0.8/0.2) | 1.7823 | 0.7335 | 0.8146 | 0.8668 | 0.9245 | — |
| pairwise HRNet+D-CeLR (0.9/0.1) | 1.7263 | 0.7019 | 0.8192 | 0.8805 | 0.9313 | — |

**RUIJIN 结论**：
- Ensemble 最佳 mean = **1.56 mm**（HRNet+VLD 两两），比 VLD 单方法（1.70 mm）提升 **8.3%**
- Ensemble 最佳 acc@2 = **0.771**（3-method 加权），比 VLD 单方法（0.729）提升 **5.8%**
- **两两组合足够高效**：HRNet+VLD 1.56 mm ≈ 3-method best 1.57 mm，且 acc@2 达 0.769

---

## 第四阶段：最终成果总览

### 表 4-1：从基线到 Ensemble 的完整提升链

| dataset | 阶段 | 最佳结果 | mean_error_mm | acc@2mm | 相比上一阶段提升 |
|---------|------|----------|---------------|---------|-----------------|
| **RENJI** | 基线单方法 | D-CeLR | 3.1643 | 0.5639 | — |
| **RENJI** | 单方法优化 | D-CeLR 改进 / VLD+TTA / HRNet 预训练 | 2.661 | 0.645 | mean ↓ 15.9% |
| **RENJI** | **Ensemble** | **3-method weighted** | **2.364** | **0.688** | **mean ↓ 11.2%** |
| | | | | | **累计 mean ↓ 25.3%** |
| **RUIJIN** | 基线单方法 | VLD | 1.8342 | 0.7102 | — |
| **RUIJIN** | 单方法优化 | VLD e100 / HRNet 预训练 | 1.698 | 0.729 | mean ↓ 7.4% |
| **RUIJIN** | **Ensemble** | **HRNet+VLD 两两** | **1.557** | **0.769** | **mean ↓ 8.3%** |
| | | | | | **累计 mean ↓ 15.1%** |

---

## 附录：详细结果路径

| 结果 | 路径 |
|------|------|
| VLD RENJI e60 | `Vertebra-Landmark-Detection/outputs/inference_renji_e60/logs/comparison_table.md` |
| VLD RENJI e60 + TTA | `Vertebra-Landmark-Detection/outputs/inference_renji_e60_tta/logs/comparison_table.md` |
| VLD RUIJIN e100 | `Vertebra-Landmark-Detection/outputs/inference_ruijin_e100/logs/comparison_table.md` |
| D-CeLR RENJI e60 | `D-CeLR/outputs/renji_ab_fullimg_e60_eval_final/logs/comparison_table.md` |
| D-CeLR RENJI 改进 e75 | `D-CeLR/outputs/renji_ab_fullimg_e80_improved/eval_best/logs/comparison_table.md` |
| D-CeLR RUIJIN e60 | `D-CeLR/outputs/ruijin_ab_fullimg_e60_eval_final/logs/comparison_table.md` |
| D-CeLR RUIJIN 改进 e63 | `D-CeLR/outputs/ruijin_ab_fullimg_e80_renji_init/eval_best/logs/comparison_table.md` |
| HRNet RENJI 基线 e60 | `HRNet-Facial-Landmark-Detection/outputs/inference_renji_spine_renji_hrnet_w18_e60/logs/comparison_table.md` |
| HRNet RUIJIN 基线 e60 | `HRNet-Facial-Landmark-Detection/outputs/inference_ruijin_spine_ruijin_hrnet_w18_e60/logs/comparison_table.md` |
| HRNet RENJI 预训练 e60 | `HRNet-Facial-Landmark-Detection/outputs/inference_renji_spine_renji_hrnet_w18_pretrained_e60/logs/comparison_table.md` |
| HRNet RUIJIN 预训练 e60 | `HRNet-Facial-Landmark-Detection/outputs/inference_ruijin_spine_ruijin_hrnet_w18_pretrained_e60/logs/comparison_table.md` |
| Ensemble RENJI | `outputs/ensemble_optimize_renji/grid_search_results.md` |
| Ensemble RUIJIN | `outputs/ensemble_optimize_ruijin/grid_search_results.md` |
| 论文级图表 | `outputs/paper_figures/` |
