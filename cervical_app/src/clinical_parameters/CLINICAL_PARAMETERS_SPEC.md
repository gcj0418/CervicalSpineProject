# 颈椎临床矢状位参数接口规范

> 本文档为 `clinical_parameters` 包的接口说明，定义六种颈椎临床参数的输入格式、计算方法及解剖学约定。
>
> **模型推理、坐标转换、匈牙利匹配、方向统一等操作由外部模块（`inference.py` / `gui.py`）负责，本包仅执行纯几何计算。**

---

## 数据格式约定

### `pts_front` — 融合前缘 28 点（C2–T1）

| 属性 | 说明 |
|------|------|
| 形状 | `(28, 2)` |
| 来源 | `predict_fusion()` 的前缘输出（通过 HRNet **前缘**锚点进行匈牙利匹配） |
| 顺序 | C2 → C3 → C4 → C5 → C6 → C7 → T1 |
| 椎体 `i` | `pts_front[i*4 : (i+1)*4]` |

每个椎体 4 个角点，顺序固定为 `[tl, tr, bl, br]`：

| 索引 | 名称 | 解剖含义 |
|------|------|---------|
| 0 | `tl` | 左上 = 前上（anterior-superior） |
| 1 | `tr` | 右上 = 后上（posterior-superior） |
| 2 | `bl` | 左下 = 前下（anterior-inferior） |
| 3 | `br` | 右下 = 后下（posterior-inferior） |

> **方向保证**：`predict_fusion` 统一所有图像方向，使**椎体（前缘）始终位于图像左侧**，棘突/关节突（后缘）位于右侧。若原始图像前缘在右，则执行水平翻转，并在每个椎体内部交换 `tl↔tr`、`bl↔br`。

### `pts_back` — 融合后缘 28 点（C2J–T1J）

| 属性 | 说明 |
|------|------|
| 形状 | `(28, 2)` |
| 来源 | `predict_fusion()` 的后缘输出（通过 HRNet **后缘**锚点进行匈牙利匹配） |
| 顺序 | C2J → C3J → C4J → C5J → C6J → C7J → T1J |

与 `pts_front` 相同的每椎体四角顺序。每个椎体的 `tr`（右上）和 `br`（右下）分别对应**上关节突**和**下关节突**的定位点，后缘方向（`tr → br`）即为关节突走向的几何代理。专用于**关节突夹角**计算。

### `pixel_spacing`

标量，单位 mm/pixel。从 NIfTI 体素间距提取，普通图像默认为 `1.0`。所有基于距离的参数（SVA、椎间隙高度、椎体位移）均需乘以此值换算为物理距离。

---

## 1. C2–C7 Cobb 角（°）

### 定义
C2 下终板与 C7 下终板之间的夹角，量化颈椎整体曲度。

| 符号 | 含义 |
|------|------|
| 正值 | 前凸（lordosis，生理曲度） |
| 负值 | 后凸 / 反弓（kyphosis / reversed） |

### 计算方法
1. 从 `pts_front` 提取 C2（索引 0）和 C7（索引 5）的四角点。
2. 分别计算下终板角度：
   - `θ = arctan2(y_br - y_bl, x_br - x_bl)`（角度制）
3. 求差并归一化到 `(-180, 180]`：
   - `diff = θ_C2 - θ_C7`
   - 若 `diff > 180` 则 `diff -= 360`
   - 若 `diff <= -180` 则 `diff += 360`

### 可视化
采用**垂线法**：分别过 C2、C7 下终板中点作垂直于各自下终板的参考线，两垂线夹角即为 Cobb 角。

### 源码索引
- **文件**：`clinical_parameters/cobb_angle.py`
- **函数**：
  - `compute_c2c7_lordosis(pts)` — C2-7 Cobb 角
  - `compute_max_cobb(pts)` — C2-7 最大节段 Cobb 角
  - `draw_cobb(image, pts)` — 绘制参考线

---

## 2. C2–C7 SVA 矢状面轴向距离（mm）

### 定义
C2 椎体中心铅垂线到 C7 椎体后上角（posterior-superior corner）之间的**水平距离**，反映颈椎整体矢状位平衡。

| 符号 | 含义 |
|------|------|
| 正值 | C2 中心位于 C7 后上角前方（前失衡） |
| 负值 | C2 中心位于 C7 后上角后方（后失衡） |

### 计算方法
1. `center_C2 = mean(box_C2)`
2. `C7_ps = tr_C7`（C7 盒的右上点，即后上角）
3. `dx = x_C7_ps - x_C2_center`
4. `SVA = dx × pixel_spacing`

### 源码索引
- **文件**：`clinical_parameters/sva.py`
- **函数**：
  - `compute_c2c7_sva(pts, pixel_spacing)`
  - `draw_sva(image, pts, pixel_spacing)`

---

## 3. T1 斜率（°）

### 定义
T1 椎体上终板与水平参考线之间的夹角。T1 斜率是颈椎矢状位参数的重要解剖基线，与颈椎前凸角、SVA 等存在耦合关系。

### 计算方法
1. T1 对应第 7 个椎体（索引 6，C2=0, C3=1, …, T1=6）。
2. 上终板由 `tl → tr` 确定：
   - `θ_T1 = arctan2(y_tr - y_tl, x_tr - x_tl)`（角度制）
3. T1 斜率即 `θ_T1`。

### 源码索引
- **文件**：`clinical_parameters/t1_slope.py`
- **函数**：
  - `compute_t1_slope(pts)`
  - `draw_t1_slope(image, pts)`

---

## 4. 椎间隙高度（mm）

### 定义
相邻椎体终板中点之间的直线距离（disc height），评估椎间盘退变及椎间隙狭窄程度。共 6 个节段：C2/3、C3/4、C4/5、C5/6、C6/7、C7/T1。

### 计算方法
对节段（上位椎体 `i`，下位椎体 `i+1`）：
1. 上位椎体下终板中点：`mid_lower_i = (bl_i + br_i) / 2`
2. 下位椎体上终板中点：`mid_upper_i1 = (tl_i1 + tr_i1) / 2`
3. 欧氏距离：`dist = ||mid_lower_i − mid_upper_i1||_2`
4. 物理高度：`height = dist × pixel_spacing`
5. 所有节段高度的均值作为最终报告值：`mean_height = mean([h_C2/3, ..., h_C7/T1])`

### 源码索引
- **文件**：`clinical_parameters/disc_height.py`
- **函数**：
  - `compute_disc_heights(pts, pixel_spacing)` → 字典
  - `draw_disc_heights(image, pts, heights)`

---

## 5. 椎体位移（mm）

### 定义
相邻椎体间的相对水平位移量（spondylolisthesis / retrolisthesis），用于评估脊柱稳定性。共 6 个节段：C2/3 至 C7/T1。

| 符号 | 含义 |
|------|------|
| 正值 | 上位椎体前滑脱（anterior listhesis，向图像左侧滑移） |
| 负值 | 上位椎体后滑脱（retrolisthesis，向图像右侧滑移） |

### 计算方法
对节段（上位椎体 `i`，下位椎体 `i+1`）：
1. 上位椎体后下角：`upper_br = br_i`
2. 下位椎体后上角：`lower_tr = tr_i1`
3. 水平偏移：`dx = x_upper_br − x_lower_tr`
4. `displacement = dx × pixel_spacing`

### 源码索引
- **文件**：`clinical_parameters/vertebral_displacement.py`
- **函数**：
  - `compute_vertebral_displacement(pts, pixel_spacing)` → 字典
  - `draw_displacements(image, pts, displacements)`

---

## 6. 关节突夹角（°）

### 定义
相邻椎体关节突（facet joint）关节面之间的夹角，反映小关节退变、排列及活动度。共 6 个节段：C2/3 至 C7/T1。

### 数据源
`pts_back`（后缘 28 点，C2J–T1J，通过 HRNet **后缘**锚点匈牙利匹配）。

每个椎体的后缘方向由 `tr → br` 近似：
- `tr` = 后上（对应上关节突，C2J–C7J 的 superior facet）
- `br` = 后下（对应下关节突，C2J–T1J 的 inferior facet）

### 计算方法
对节段（上位椎体 `i`，下位椎体 `i+1`）：
1. 上位椎体后缘方向：`v_upper = br_i - tr_i`
2. 下位椎体后缘方向：`v_lower = br_i1 - tr_i1`
3. 夹角：`cosθ = (v_upper · v_lower) / (|v_upper| |v_lower|)`
4. `θ = arccos(cosθ)`，截断到 `[0, 180]`（角度制）

### 可视化说明
角度标签使用 PIL 渲染（而非 OpenCV `putText`），以支持 Unicode 度符号 `°`。

### 源码索引
- **文件**：`clinical_parameters/facet_joint_angle.py`
- **函数**：
  - `compute_facet_joint_angles(pts_back, pixel_spacing)` → 字典
  - `draw_facet_angles(image, pts_back)`

---

## 附录：外部模块职责

| 职责 | 所属模块 |
|------|----------|
| HRNet 推理、heatmap 解码 | `inference.py` |
| VLD（SpineNet）推理、椎体 bounding box 检测 | `vld_inference.py` |
| 匈牙利匹配（前缘 7 锚点 + 后缘 7 锚点） | `inference.py`（`fuse_hrnet_vld`） |
| 方向统一（水平翻转 + 四角交换） | `inference.py`（`predict_fusion`） |
| NIfTI 加载、体素间距提取 | `inference.py`（`load_medical_image`） |
| 参数计算调度、UI 叠加显示 | `gui.py` |
| **纯数学计算** | `clinical_parameters/*.py` |
