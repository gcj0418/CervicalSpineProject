import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from typing import Any
import re


def load_nifti_image(file_path: str) -> Any:
    return nib.load(file_path)  # type: ignore[attr-defined]


def _natural_label_key(label: str):
    """Sort labels like F-2 before F-10 while keeping prefix groups stable."""
    if not label:
        return ("", -1, "")
    match = re.match(r"^([A-Za-z]+)[-_]?(\d+)$", label.strip())
    if match:
        return (match.group(1), int(match.group(2)), "")
    return (label, 10**9, label)


def read_image(file_path):
    """读取 .nii 或 .nii.gz 图像"""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    img = load_nifti_image(str(p))
    return img.get_fdata().astype(np.float32)


def read_keypoints_from_json(file_path, img_nib=None):
    """
    读取 JSON 文件中的关键点（支持 3D Slicer markups 格式）
    如果提供 img_nib，使用其仿射矩阵转换为体素坐标
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = []
    labels = []
    
    # 3D Slicer markups 格式
    if 'markups' in data and len(data['markups']) > 0:
        for cp in data['markups'][0].get('controlPoints', []):
            pos = cp.get('position', [0, 0, 0])
            label = cp.get('label', '')
            points.append([pos[0], pos[1], pos[2]])
            labels.append(label)
    
    if not points:
        return np.array([]), []
    
    points = np.array(points)

    # 关键点顺序标准化，避免不同文件中 controlPoints 顺序不一致导致训练噪声
    if len(labels) == len(points):
        order = sorted(range(len(labels)), key=lambda i: _natural_label_key(labels[i]))
        points = points[order]
        labels = [labels[i] for i in order]
    
    # 如果提供图像，使用仿射矩阵转换到体素坐标
    if img_nib is not None:
        affine = img_nib.affine.copy()
        
        # 处理负缩放
        flip_x = affine[0, 0] < 0
        flip_y = affine[1, 1] < 0
        flip_z = affine[2, 2] < 0
        
        affine[0, 0] = abs(affine[0, 0])
        affine[1, 1] = abs(affine[1, 1])
        affine[2, 2] = abs(affine[2, 2])
        
        inv_affine = np.linalg.inv(affine)
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        voxel_coords = (inv_affine @ points_homo.T).T[:, :3]
        
        img_shape = img_nib.shape
        if flip_x:
            voxel_coords[:, 0] = img_shape[0] - 1 - voxel_coords[:, 0]
        if flip_y:
            voxel_coords[:, 1] = img_shape[1] - 1 - voxel_coords[:, 1]
        if flip_z:
            voxel_coords[:, 2] = img_shape[2] - 1 - voxel_coords[:, 2]
        
        points = voxel_coords
    
    return points, labels


def main():
    # 配置数据路径
    data_dir = Path(__file__).resolve().parent / "data" / "HE_CHUN_DX_颈椎左侧位_20231205_2"
    
    # 找 .nii 或 .nii.gz 文件
    nii_files = list(data_dir.glob("*.nii.gz")) + list(data_dir.glob("*.nii"))
    if not nii_files:
        print("Error: No .nii or .nii.gz files found!")
        return
    
    img_path = nii_files[0]
    print(f"Loading image: {img_path.name}")
    
    img_nib = load_nifti_image(str(img_path))
    img = img_nib.get_fdata().astype(np.float32)
    
    print(f"Image shape: {img.shape}")
    print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")
    
    # 找所有 JSON 文件（关键点标注）
    json_files = list(data_dir.glob("*.json"))
    all_keypoints = {}
    
    for json_file in json_files:
        kps, labels = read_keypoints_from_json(json_file, img_nib)
        if len(kps) > 0:
            all_keypoints[json_file.name] = (kps, labels)
            print(f"Loaded {len(kps)} keypoints from {json_file.name}")
    
    # 显示图像
    if img.shape[2] == 1:
        # 2D 图像
        slice_data = img[:, :, 0]
        print("\nDisplaying 2D image")
    else:
        # 3D 图像：取中间切片
        idx = img.shape[2] // 2
        slice_data = img[:, :, idx]
        print(f"\nDisplaying 3D image slice {idx}")
    
    # 绘制
    plt.figure(figsize=(16, 12))
    vmin = np.percentile(slice_data, 2)
    vmax = np.percentile(slice_data, 98)
    plt.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
    
    # 绘制所有关键点（使用不同的颜色）
    colors = ['red', 'lime', 'cyan', 'magenta', 'yellow', 'orange']
    color_idx = 0
    
    for json_name, (kps, labels) in all_keypoints.items():
        color = colors[color_idx % len(colors)]
        for i, (kp, label) in enumerate(zip(kps, labels)):
            # kp 是 (x, y, z) 体素坐标
            x = slice_data.shape[1] - 1 - kp[1]  # 翻转 y
            y = slice_data.shape[0] - 1 - kp[0]  # 翻转 x
            plt.scatter(x, y, c=color, s=100, marker='o', alpha=0.8)
            if label:
                plt.text(x+10, y+10, label, color=color, fontsize=8)
        
        color_idx += 1
    
    plt.title(f"Image: {img_path.name}")
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()