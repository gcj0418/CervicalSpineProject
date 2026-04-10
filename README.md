# CervicalSpineProject

本 README 只保留当前可复现且已验证有效的内容。

## 1. 当前结论（以 test 集为准）

当前推荐使用的模型权重：
- `outputs/training/checkpoints/best.pth`（legacy 回归模型）

已复核的 test 指标：
- `avg_mean_pixel_error = 26.0864`
- `avg_pck_10px = 0.1343`

对应结果文件：
- `outputs/inference_training_bestcmp/logs/test_summary.json`

## 2. 快速复现当前效果

### 2.1 推理（推荐，最快验证）

```bash
e:/CervicalSpineProject/.venv/Scripts/python.exe -u inference.py \
  --model_path outputs/training/checkpoints/best.pth \
  --output_dir outputs/inference_now \
  --device cuda \
  --split test \
  --data_dir data/ \
  --batch_size 8 \
  --target_size 512 \
  --train_size 0.7 \
  --val_size 0.15
```

推理后重点查看：
- 总体指标：`outputs/inference_now/logs/test_summary.json`
- 每样本误差：`outputs/inference_now/logs/test_case_metrics.csv`
- 每关键点误差：`outputs/inference_now/logs/test_keypoint_metrics.csv`
- 可视化图：`outputs/inference_now/visualizations/`

### 2.2 训练（仅当你要重新训练）

当前 `train.py` 支持两种模式：
- `--model_type legacy_regression`（老版本回归链路，用于复现实验）
- `--model_type heatmap`（热图链路，实验用）

复现老版本训练链路（推荐模板）：

```bash
e:/CervicalSpineProject/.venv/Scripts/python.exe -u train.py \
  --device cuda \
  --model_type legacy_regression \
  --output_dir outputs/training_legacy_reproduce \
  --batch_size 16 \
  --epochs 60 \
  --augmentation \
  --save_every 5 \
  --patience 12 \
  --allow_variable_keypoints
```

## 3. 项目结构（当前）

下面只列当前链路会用到的目录和文件：

```text
E:/CervicalSpineProject/
├── data/                                # 原始影像与标注（NIfTI + JSON）
├── outputs/
│   ├── training/                        # 当前基线训练产物
│   │   ├── checkpoints/                 # best.pth / latest.pth / epoch_xxx.pth
│   │   └── logs/                        # metrics.csv / config.json
│   ├── inference_training_bestcmp/      # 已复核的最佳推理结果
│   │   ├── logs/                        # test_summary / case_metrics / keypoint_metrics
│   │   ├── predictions/                 # 每病例预测关键点 .npy
│   │   └── visualizations/              # 预测可视化图
│   └── ...                              # 其他实验目录
├── data_loader.py                       # 读取图像与关键点、坐标转换
├── preprocess.py                        # 归一化/缩放/增强预处理
├── dataset.py                           # Dataset封装、split、可选56点过滤
├── model.py                             # heatmap模型 + legacy回归兼容模型
├── train.py                             # 训练入口（支持model_type切换）
├── inference.py                         # 推理与评估入口（自动识别checkpoint类型）
└── README.md
```

## 4. 核心文件功能（当前有效）

### data_loader.py
- 扫描并读取 `.nii/.nii.gz` 与关键点 JSON。
- 将标注从物理坐标转换到体素坐标。

### preprocess.py
- 对影像做归一化和 resize。
- 在训练阶段可选增强（旋转/噪声等）。
- 输出模型输入张量与变换后的关键点。

### dataset.py
- 负责 train/val/test 切分。
- 支持按关键点数过滤（如只保留 56 点样本）。
- 提供 DataLoader 所需的批处理组织。

### model.py
- 提供两类关键点模型：
  - heatmap 模式（实验链路）。
  - legacy_regression 模式（当前最佳基线兼容）。
- 支持直接加载历史 best checkpoint。

### train.py
- 训练主入口，输出 checkpoint、metrics.csv、config.json。
- 支持 `--model_type legacy_regression|heatmap`。
- 支持可选能力：56点过滤、困难样本重采样、关键点加权、skip_final_test。

### inference.py
- 推理主入口，支持 split 批量评估与单图推理。
- 自动判断 checkpoint 类型并选择兼容模型加载。
- 生成三类评估产物：
  - 总体指标 `test_summary.json`
  - 每样本指标 `test_case_metrics.csv`
  - 每关键点指标 `test_keypoint_metrics.csv`

## 5. 当前代码开关说明（只列有效）

### 5.1 非 56 点样本过滤

训练默认保留了“按关键点数量过滤”的能力：
- 默认：`--require_num_keypoints 56`
- 关闭过滤：`--allow_variable_keypoints`

说明：
- 若你要“严格复现老 best 路线”，建议使用 `--allow_variable_keypoints`。
- 若你做 56 点子集实验，可保留默认过滤。

### 5.2 快速 smoke 训练

为了避免长时间卡在最终测试：
- 可加 `--skip_final_test`

## 6. 当前推荐对比口径

做 checkpoint 对比时，请统一以下条件：
- 同一 `split=test`
- 同一 `data_dir / target_size / batch_size / train_size / val_size`
- 以 `test_summary.json` 为最终判断，不仅看训练日志的 val 曲线

## 7. 目录约定（当前会用到）

- 训练日志：`outputs/*/logs/metrics.csv`
- 训练权重：`outputs/*/checkpoints/*.pth`
- 推理汇总：`outputs/*/logs/test_summary.json`
- 推理可视化：`outputs/*/visualizations/`

补充：预处理阶段输出
- 预处理可视化：`outputs/preprocess/visualizations/preprocess_*.png`
- 预处理元数据：`outputs/preprocess/metadata/preprocess_*.json`
- 用途：质量检查、参数追踪、复现实验输入

## 8. 版本备注

- 本 README 已移除历史阶段性说明和未落地路线。
- 如果后续最佳效果更新，请只改“第 1 节当前结论”和对应命令示例。
