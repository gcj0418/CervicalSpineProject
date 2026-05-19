# VLD 拆分实验总结与下一步

## 一、数据结构（关键）

RENJI 原始标注有 **56 个点**，存储于：
- `Vertebra-Landmark-Detection/data_renji_vld/labels/{train,val,test}/*.mat`
- `.mat` 中 key 为 `p2`，shape `(56, 2)`

56 个点的语义：
| 点序号 | 内容 | 数量 |
|--------|------|------|
| 0~27 | **椎体** (vertebrae) 7个 × 4角 | 28 |
| 28~55 | **关节突** (facets) 7个 × 4角 | 28 |

> 之前错误理解成"每个椎体的前缘2点+后缘2点"，导致实验失败（+23%误差）。现已纠正。

## 二、已生成的数据集

```
Vertebra-Landmark-Detection/
├── data_renji_vld/                          # 原始 56 点数据集
├── data_renji_vld_vertebrae/                # 椎体 28 点数据集（新生成）
│   ├── data/{train,val,test}/*.png
│   └── labels/{train,val,test}/*.mat   (p2 shape = 28×2)
└── data_renji_vld_facets/                   # 关节突 28 点数据集（新生成）
    ├── data/{train,val,test}/*.png
    └── labels/{train,val,test}/*.mat   (p2 shape = 28×2)
```

拆分方式：对每个 `.mat` 文件：
- `vertebrae`：取 `p2[:28, :]`
- `facets`：取 `p2[28:56, :]`

## 三、VLD 代码状态

`Vertebra-Landmark-Detection/` 下的源码已通过 `git checkout HEAD` 恢复为原始状态，**无** `corners_per_vertebra` 修改。

关键文件：
- `main.py` — 入口，`--max_points` 默认 56，`--K` 默认 14
- `dataset.py` — `BaseDataset`，`load_gt_pts()` 会截断到 28 点（对 56 点数据只取前 28）
- `pre_proc.py` — `generate_ground_truth()`, `processing_train()`
- `decoder.py` — `ctdet_decode()`，输出 `num_obj × 11` (cenx,ceny,tl_x,tl_y,tr_x,tr_y,bl_x,bl_y,br_x,br_y,score)
- `train.py / test.py / eval.py` — 训练、推理、评估

**注意**：`dataset.py` 中 `load_gt_pts()` 有一行硬编码截断：
```python
if pts.shape[0] > 28:
    pts = pts[:28, :]   # 对 56 点数据会自动截断到 vertebrae！
```
对 vertebrae-only 和 facets-only 数据集无影响（它们本身就是 28 点）。

## 四、之前失败实验的遗留物（可清理）

```
Vertebra-Landmark-Detection/
├── data_renji_vld_front/          # 错误实验：前缘2点 × 14椎体
├── data_renji_vld_back/           # 错误实验：后缘2点 × 14椎体
├── weights_renji_front/           # 错误实验模型权重
└── weights_renji_back/            # 错误实验模型权重

draft_box/
├── split_vld_dataset.py           # 错误拆分的脚本
├── eval_split_vld_models.py       # 错误实验的评估脚本
├── vis_split_compare.py           # 错误实验的可视化
├── vis_split_datasets_preview.py  # 错误实验的数据集预览
└── vis_split_compare/             # 错误实验的对比图
└── vis_split_datasets_preview/    # 错误实验的数据集预览图
```

## 五、接下来要做的事

### 步骤 1：训练椎体模型
```bash
cd Vertebra-Landmark-Detection
python main.py --phase train \
    --dataset renji \
    --data_dir data_renji_vld_vertebrae \
    --max_points 28 \
    --K 7 \
    --num_epoch 50 \
    --weights_dir weights_renji_vertebrae
```

### 步骤 2：训练关节突模型
```bash
python main.py --phase train \
    --dataset renji \
    --data_dir data_renji_vld_facets \
    --max_points 28 \
    --K 7 \
    --num_epoch 50 \
    --weights_dir weights_renji_facets
```

### 步骤 3：分别评估两个模型
```bash
# 椎体模型
python main.py --phase eval \
    --dataset renji \
    --data_dir data_renji_vld_vertebrae \
    --max_points 28 --K 7 \
    --weights_dir weights_renji_vertebrae \
    --resume model_last.pth

# 关节突模型
python main.py --phase eval \
    --dataset renji \
    --data_dir data_renji_vld_facets \
    --max_points 28 --K 7 \
    --weights_dir weights_renji_facets \
    --resume model_last.pth
```

### 步骤 4：与原始 56 点模型对比
需要写一个新的评估脚本，将两个 28 点模型的预测拼接成 56 点，再与 GT 56 点做 Hungarian 匹配，计算整体误差。

### 步骤 5（可选）：融合到 cobb_app
如果拆分模型效果好，考虑在 `cobb_app/` 中实现：
1. 椎体模型预测 28 点
2. 关节突模型预测 28 点
3. 拼接为 56 点 → 计算 Cobb 角

## 六、关键参数对照表

| 配置 | max_points | K (num_vertebra) | corners_per_vertebra |
|------|-----------|------------------|---------------------|
| 原始 56 点 | 56 | 14 | 4 |
| 椎体 28 点 | 28 | 7 | 4 |
| 关节突 28 点 | 28 | 7 | 4 |
| 之前错误 2 点 | 28 | 14 | 2 |

## 七、预览图位置

`draft_box/vis_vertebrae_facets_split/` — 10 张三列对比图（原始 56 点 / 椎体 28 点 / 关节突 28 点）
