<div align="center">

# 颈椎侧位 X 光片关键点检测与测量

**本项目最终交付一个可直接运行的桌面应用 `cervical_app/`，基于 HRNet + VLD 融合模型实现颈椎侧位 X 光片多参数自动测量。**

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 桌面应用 cervical_app

基于优化后的关键点检测模型封装的单文件桌面工具，选择 PNG/JPG 颈椎侧位片即可自动完成椎体框定与角度测量。

### 功能

- **一键加载**：选择单张颈椎侧位片即可自动分析
- **智能方向统一**：自动识别椎体朝向，统一为标准侧位视角
- **Cobb 角**：C2-C7 前凸角、最大 Cobb 角，带正负号与自动诊断
- **SVA**：矢状面垂直轴
- **T1 斜率**：T1 slope
- **椎间隙**：C2-C7 各椎间隙高度
- **椎位移**：各椎体相对位移
- **关节突角**：关节突关节角度
- **可视化与导出**：C2-C7 椎体框 + 终板参考线 + 结果保存

### 快速运行

```bash
cd cervical_app
pip install -r requirements.txt
python main.py
```

<p align="center">
  <img src="cervical_app/image/901bffcefc7f44d9b9116ffad25475e0.png" width="70%">
  <br>
  <em>桌面应用主界面</em>
</p>

---

## 技术路线：从模型选型到应用落地

`cervical_app` 的背后经历了多轮模型选型、优化与架构迭代。以下按时间线记录关键决策过程。

### 第一阶段：三范式基线对比

复现并对比了三种主流关键点检测方法：

| 方法 | 范式 | RENJI (mm) | RUIJIN (mm) |
|:------:|:--------:|:----------:|:-----------:|
| VLD | Heatmap Regression | 3.90 | 1.83 |
| D-CeLR | CNN + Transformer | 3.16 | 3.90 |
| HRNet | High-Resolution Net | 3.37 | 2.37 |

### 第二阶段：单方法优化

- **D-CeLR**：迁移学习 + 数据增强，RUIJIN 从 3.90 mm → 2.57 mm
- **VLD**：horizontal-flip TTA，RENJI 从 3.90 mm → 2.88 mm
- **HRNet**：ImageNet 预训练，RENJI 从 3.37 mm → 2.66 mm

### 第三阶段：Ensemble 融合

将优化后的三方法做 Hungarian 对齐 + 加权融合：

| dataset | 最佳融合配置 | mean error | acc@2mm |
|---------|-------------|-----------|---------|
| RENJI | 3-method (0.4/0.3/0.3) | **2.36 mm** | **68.8%** |
| RUIJIN | HRNet+VLD (0.5/0.5) | **1.56 mm** | **76.9%** |

> 完整对比报告见 [`outputs/comparison_table_all.md`](outputs/comparison_table_all.md)

### 第四阶段：VLD 拆分改进（本阶段新增）

在准备将模型落地为应用时，发现原始 VLD 单模型同时预测 56 个点存在瓶颈：单模型任务过重，且 `dataset.py` 存在硬编码截断导致训练时实际只学到前 28 点。

**改进措施**：将 VLD 拆分为两个独立的 28 点模型分别训练，再拼接为 56 点：
1. **椎体模型**（vertebrae）：专精预测椎体 4 角
2. **关节突模型**（facets）：专精预测关节突 4 角

**拆分后效果**（RENJI test，完整 56 点评估）：

| 模型 | mean error | 2mm acc | 4mm acc |
|------|-----------|---------|---------|
| 原始 VLD 56 点 | 3.28 mm | 58.4% | 82.7% |
| **拆分融合 56 点** | **2.04 mm** | **68.4%** | **89.8%** |

> 当前 `cervical_app` 仍使用 HRNet + VLD 融合架构，后续计划将拆分后的双 VLD 模型集成进应用推理流程。

---

## 仓库结构

```
├── cervical_app/                   # 桌面应用（最终成果）
├── Vertebra-Landmark-Detection/    # VLD (CenterNet-based) 复现、训练与拆分实验
├── D-CeLR/                         # D-CeLR (ResNet34 + Transformer) 复现与训练
├── HRNet-Facial-Landmark-Detection/ # HRNet-W18 复现与训练
├── outputs/                        # 实验结果与可视化
├── draft_box/                      # 实验草稿与中间脚本
├── eval_fusion_56pts.py            # 拆分模型融合评估脚本
├── ensemble_optimize.py            # 融合权重网格搜索
├── convert_renji_to_vld_dataset.py # 数据格式转换
├── convert_ruijin_to_vld_dataset.py
└── README.md
```

---

## 训练与评估

### 环境配置

```bash
pip install torch torchvision opencv-python numpy scipy matplotlib seaborn yacs tensorboardX
```

### 拆分 VLD 训练（当前最优）

```bash
cd Vertebra-Landmark-Detection

# 椎体模型 (28 pts)
python main.py --phase train --dataset renji \
    --data_dir data_renji_vld_vertebrae --max_points 28 --K 7 \
    --num_epoch 100 --weights_dir weights_renji_vertebrae

# 关节突模型 (28 pts)
python main.py --phase train --dataset renji \
    --data_dir data_renji_vld_facets --max_points 28 --K 7 \
    --num_epoch 100 --weights_dir weights_renji_facets
```

### 融合评估

```bash
cd draft_box
python eval_fusion_56pts.py
```

---

## 文档

- [前期完整量化对比报告](evaluations/01_comparison_table_all.md)
- [VLD 拆分实验总结](evaluations/05_SPLIT_EXPERIMENT_SUMMARY.md)

---

## 开源协议

本项目仅供学术研究使用。各子模块的协议请参考其对应许可证。

---

<div align="center">

Cervical Spine Landmark Detection & Cobb Angle Measurement

</div>
