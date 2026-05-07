# 最佳结果可视化目录

> 生成时间: 2026-05-06  
> 本目录汇总了中期报告前各医院、各策略下的最佳关键点检测结果可视化。

---

## 目录结构

```
best_results/
├── README.md                          # 本说明文件
│
├── RENJI/                             # 仁济医院数据（30例，56点，~0.125 mm/px）
│   ├── single_best_HRNet_pretrained/  # 单方法最佳：HRNet + ImageNet预训练
│   │   ├── compare/                   # GT(绿) vs Pred(红) 叠加图
│   │   └── side_by_side/              # GT 与 Pred 左右对比图
│   ├── ensemble_best/                 # 复合最佳：3-method加权融合
│   │   ├── compare/
│   │   └── side_by_side/
│   └── figures/                       # 统计图表（基于最新数据）
│       ├── bar_comparison.png         # Mean Error + SDR柱状对比
│       ├── cdf.png                    # 累积分布函数（阈值线标注）
│       ├── error_distribution.png     # 误差核密度估计
│       ├── bland_altman.png           # 三对方法Bland-Altman一致性分析
│       └── worst_samples.png          # Top-15困难样本误差对比
│
└── RUIJIN/                            # 瑞金医院数据（14例，52点，0.28 mm/px）
    ├── single_best_HRNet_pretrained/  # 单方法最佳：HRNet + ImageNet预训练
    │   ├── compare/
    │   └── side_by_side/
    ├── ensemble_best/                 # 复合最佳：HRNet+VLD两两加权融合
    │   ├── compare/
    │   └── side_by_side/
    └── figures/                       # 统计图表
        ├── bar_comparison.png
        ├── cdf.png
        ├── error_distribution.png
        ├── bland_altman.png
        └── worst_samples.png
```

---

## 最佳结果汇总

### RENJI（仁济医院）

| 策略 | 方法 | Mean Error (mm) | Acc@2mm | 权重配置 |
|------|------|----------------|---------|---------|
| **单方法最佳** | HRNet + ImageNet预训练 | **2.66** | 0.503 | — |
| **复合最佳** | 3-method加权融合 | **2.36** | 0.688 | HRNet:VLD:D-CeLR = 0.4:0.3:0.3 |

**关键发现**：
- ImageNet预训练对高分辨率脊柱X线片任务迁移效果显著（mean error ↓21.2%）
- Ensemble充分利用了三方法互补性：VLD精度高但偶有离群值，D-CeLR稳定但mean偏高，HRNet预训练后性能均衡

### RUIJIN（瑞金医院）

| 策略 | 方法 | Mean Error (mm) | Acc@2mm | 权重配置 |
|------|------|----------------|---------|---------|
| **单方法最佳** | HRNet + ImageNet预训练 | **1.75** | 0.692 | — |
| **复合最佳** | HRNet + VLD两两融合 | **1.56** | 0.769 | HRNet:VLD = 0.5:0.5 |

**关键发现**：
- HRNet预训练在低分辨率数据上同样有效（mean error ↓26.1%）
- 两两融合即可达到接近3-method的效果，计算效率更高
- VLD在RUIJIN上表现优异（1.83mm），与HRNet融合后进一步提升至1.56mm

---

## 图像标注说明

### compare/ 目录
每张图展示：
- **绿色圆点**：医生标注的Ground Truth关键点
- **红色圆点**：模型预测的关键点
- 叠加在同一X线片原图上，直观展示定位偏差

### side_by_side/ 目录
每张图分为左右两栏：
- **左侧**：仅标注GT（绿色）
- **右侧**：仅标注预测结果（红色）
- 便于单独观察某一侧的全貌

---

## 统计图表说明

| 图表 | 内容 | 用途 |
|------|------|------|
| `bar_comparison.png` | 左=Mean Error柱状图，右=SDR@2/2.5/3/4mm | 论文表格的图形化呈现 |
| `cdf.png` | 误差累积分布曲线 | 读取任意阈值下的检测成功率 |
| `error_distribution.png` | 核密度估计（KDE）曲线 | 观察误差分布形态与长尾 |
| `bland_altman.png` | 三对方法的差异-均值散点图 | 评估系统偏差与一致性界限 |
| `worst_samples.png` | Top-15困难样本的各方法误差 | 定位失败模式分析 |

---

## 数据来源路径

| 内容 | 原始路径 |
|------|---------|
| RENJI HRNet预训练预测 | `HRNet-Facial-Landmark-Detection/outputs/inference_renji_spine_renji_hrnet_w18_pretrained_e60/predictions/predictions.pth` |
| RENJI VLD TTA预测 | `Vertebra-Landmark-Detection/outputs/inference_renji_e60_tta/predictions/predictions.pth` |
| RENJI D-CeLR改进预测 | `D-CeLR/outputs/renji_ab_fullimg_e80_improved/eval_best/predictions/` |
| RUIJIN HRNet预训练预测 | `HRNet-Facial-Landmark-Detection/outputs/inference_ruijin_spine_ruijin_hrnet_w18_pretrained_e60/predictions/predictions.pth` |
| RUIJIN VLD预测 | `Vertebra-Landmark-Detection/outputs/inference_ruijin_e60/predictions/predictions.pth` |
| RUIJIN D-CeLR改进预测 | `D-CeLR/outputs/ruijin_ab_fullimg_e80_renji_init/eval_best/predictions/` |
