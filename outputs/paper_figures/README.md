# 可视化图表

> 生成时间: 2026-05-05  
> 数据集: RENJI (30 test samples, 56 landmarks) + RUIJIN (14 test samples, 52 landmarks)  
> 方法: VLD / D-CeLR / HRNet (统一 e60 checkpoint)

---

## 图表索引

### 1. Error Distribution (`*_error_distribution.png`)
- **内容**: 各方法定位误差的核密度估计（KDE）曲线
- **用途**: 直观展示误差分布形态（是否偏斜、是否有长尾）
- **关键发现**:
  - **RENJI**: D-CeLR 峰值最低（~1.5mm），但右尾较长；VLD 有极端离群值拉高 mean
  - **RUIJIN**: VLD 峰值最尖锐（~1.0mm），精度最高；D-CeLR 分布最宽

### 2. Cumulative Distribution Function (`*_cdf.png`)
- **内容**: 误差累积分布曲线，标注 2/2.5/3 mm 阈值线
- **用途**: 直接读取任意阈值下的 SDR（Successful Detection Rate）
- **关键发现**:
  - **RENJI**: median 最优约在 2.5mm（D-CeLR 略优于 VLD）
  - **RUIJIN**: VLD 在 2mm 处 SDR 约 71%，显著领先 HRNet (47%) 和 D-CeLR (31%)

### 3. Bland-Altman Analysis (`*_bland_altman.png`)
- **内容**: 三对方法（VLD-HRNet, VLD-D-CeLR, HRNet-D-CeLR）的一致性分析
- **用途**: 评估系统偏差（mean diff）和一致性界限（±1.96 SD）
- **关键发现**:
  - **RENJI**: VLD 与 D-CeLR 均值差异最小，一致性最好
  - **RUIJIN**: VLD 明显优于其他两种方法，差异较大

### 4. Bar Comparison (`*_bar_comparison.png`)
- **内容**: 左图 = mean error，右图 = SDR@2/2.5/3/4 mm
- **用途**: 论文表格的图形化呈现
- **关键发现**:
  - **RENJI**: D-CeLR mean=3.14 mm 最低，但 VLD acc@2=0.64 最高
  - **RUIJIN**: VLD 全面领先（mean=1.83, acc@2=0.71）

### 5. Worst Samples (`*_worst_samples.png`)
- **内容**: 按 VLD error 排序的前 15 个最差样本，展示三种方法的误差
- **用途**: 定位困难案例，分析失败模式
- **关键发现**:
  - 某些样本三种方法同时失效（图像质量差或体位异常）
  - 某些样本仅一种方法失效（说明方法间互补性强，ensemble 有价值）

---

## 文件列表

| 文件名 | 数据集 | 尺寸 |
|--------|--------|------|
| RENJI_error_distribution.png | RENJI | 147 KB |
| RENJI_cdf.png | RENJI | 137 KB |
| RENJI_bland_altman.png | RENJI | 317 KB |
| RENJI_bar_comparison.png | RENJI | 140 KB |
| RENJI_worst_samples.png | RENJI | 347 KB |
| RUIJIN_error_distribution.png | RUIJIN | 154 KB |
| RUIJIN_cdf.png | RUIJIN | 137 KB |
| RUIJIN_bland_altman.png | RUIJIN | 317 KB |
| RUIJIN_bar_comparison.png | RUIJIN | 141 KB |
| RUIJIN_worst_samples.png | RUIJIN | 102 KB |

---

## 补充说明

- 所有图表使用统一配色方案：VLD(红) / D-CeLR(蓝) / HRNet(绿)
- DPI = 300，适合直接插入论文
- 误差计算使用 Hungarian matching 对齐 prediction 到 GT，再乘以 pixel spacing 转换为 mm
