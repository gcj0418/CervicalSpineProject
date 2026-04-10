import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset
import torch
from typing import Any

from data_loader import read_keypoints_from_json
from preprocess import ImagePreprocessor


# 缓存每个 (image, json) 对应的关键点数量，避免 train/val/test 重复扫描。
_KEYPOINT_COUNT_CACHE = {}


def load_nifti_image(file_path: str) -> Any:
    return nib.load(file_path)  # type: ignore[attr-defined]


class CervicalSpineDataset(Dataset):
    """
    颈椎侧位 X 光影像 PyTorch Dataset
    
    自动扫描数据目录，加载图像和关键点，应用预处理
    """
    
    def __init__(self, 
                 data_dir='data/', 
                 split='train',
                 train_size=0.7,
                 val_size=0.15,
                 seed=42,
                 augmentation=False,
                 target_size=512,
                 required_num_keypoints=None,
                 skip_invalid_keypoint_count=False):
        """
        参数：
            data_dir: 数据根目录路径（包含多个案例子目录）
            split: 数据分割方式 ('train', 'val', 'test')
            train_size: 训练集比例（默认 0.7）
            val_size: 验证集比例（默认 0.15，剩下为测试集 0.15）
            seed: 随机种子（确保可重现）
            augmentation: 是否启用数据增强（仅训练集使用）
            target_size: 预处理目标尺寸
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.seed = seed
        self.augmentation = augmentation
        self.target_size = target_size
        self.required_num_keypoints = required_num_keypoints
        self.skip_invalid_keypoint_count = skip_invalid_keypoint_count
        
        # 检查数据目录
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # 扫描所有文件对（image, keypoints）
        self.file_list = self._scan_files()

        if self.skip_invalid_keypoint_count:
            if self.required_num_keypoints is None:
                raise ValueError("required_num_keypoints must be set when skip_invalid_keypoint_count=True")
            self.file_list = self._filter_by_keypoint_count(self.file_list, int(self.required_num_keypoints))
        
        if len(self.file_list) == 0:
            raise RuntimeError(f"No valid image-keypoint pairs found in {self.data_dir}")
        
        # 数据分割
        self.indices = self._split_data(train_size, val_size)
        
        # 初始化预处理器
        if augmentation and split == 'train':
            from preprocess import AugmentedPreprocessor
            self.preprocessor = AugmentedPreprocessor(
                target_size=target_size,
                augmentation=True,
                augmentation_params={
                    'rotation_range': 12,
                    'elastic_alpha': 30,
                    'elastic_sigma': 3,
                    'elastic_prob': 0.0,
                    'gamma_prob': 0.3,
                    'gamma_range': (0.9, 1.1),
                    'noise_prob': 0.2,
                    'noise_std': 0.01,
                }
            )
        else:
            self.preprocessor = ImagePreprocessor(target_size=target_size)
        
        print(f"CervicalSpineDataset initialized:")
        print(f"  Split: {split}")
        print(f"  Total cases: {len(self.file_list)}")
        print(f"  {split.upper()} samples: {len(self.indices)}")
        print(f"  Augmentation: {augmentation and split == 'train'}")
        if self.skip_invalid_keypoint_count:
            print(f"  Keypoint filter: keep only samples with {self.required_num_keypoints} points")

    def _get_num_keypoints(self, img_path, json_path):
        cache_key = (str(img_path.resolve()), str(json_path.resolve()))
        if cache_key in _KEYPOINT_COUNT_CACHE:
            return _KEYPOINT_COUNT_CACHE[cache_key]

        img_nib = load_nifti_image(str(img_path))
        keypoints, _ = read_keypoints_from_json(json_path, img_nib)
        kp_count = int(keypoints.shape[0])
        _KEYPOINT_COUNT_CACHE[cache_key] = kp_count
        return kp_count

    def _filter_by_keypoint_count(self, file_list, required_num_keypoints):
        filtered = []
        skipped = 0

        for img_path, json_path, case_name in file_list:
            try:
                kp_count = self._get_num_keypoints(img_path, json_path)
            except Exception as exc:
                skipped += 1
                print(f"[WARN] Skip sample due to keypoint count read error: {case_name} ({exc})")
                continue

            if kp_count == required_num_keypoints:
                filtered.append((img_path, json_path, case_name))
            else:
                skipped += 1

        print(
            f"Keypoint-count filtering: kept {len(filtered)}/{len(file_list)}, "
            f"skipped {skipped} samples (required={required_num_keypoints})"
        )
        return filtered
    
    def _scan_files(self):
        """
        递归扫描 data_dir，找到所有 .nii.gz/.nii 和对应的 .json 文件对
        
        返回：
            list of (img_path, json_path, case_name) 三元组
        """
        file_pairs = []

        # 支持多层级目录结构：例如 data/RENJI/0929/... 
        all_dirs = [self.data_dir] + [p for p in self.data_dir.rglob("*") if p.is_dir()]
        seen = set()

        for case_dir in sorted(all_dirs):
            # 查找当前目录下的 .nii.gz / .nii 和 .json
            nii_files = sorted(list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii")))
            json_files = sorted(list(case_dir.glob("*.json")))
            if not nii_files or not json_files:
                continue

            # 建立 json stem 索引，优先同名配对
            json_by_stem = {}
            for jp in json_files:
                json_by_stem[jp.stem.lower()] = jp

            for nii_path in nii_files:
                if nii_path.name.lower().endswith(".nii.gz"):
                    nii_stem = nii_path.name[:-7]
                else:
                    nii_stem = nii_path.stem

                json_path = json_by_stem.get(nii_stem.lower(), json_files[0])
                pair_key = (str(nii_path.resolve()), str(json_path.resolve()))
                if pair_key in seen:
                    continue

                seen.add(pair_key)
                case_name = nii_stem
                file_pairs.append((nii_path, json_path, case_name))
        
        return file_pairs
    
    def _split_data(self, train_size, val_size):
        """
        按比例将数据分割为 train/val/test
        
        参数：
            train_size: 训练集比例
            val_size: 验证集比例
            test_size: = 1 - train_size - val_size
        
        返回：
            self.split 对应的样本索引数组
        """
        np.random.seed(self.seed)
        n = len(self.file_list)
        
        # 生成打乱后的索引
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        # 计算分割点
        n_train = int(n * train_size)
        n_val = int(n * val_size)
        
        # 按 split 返回对应的索引
        if self.split == 'train':
            return indices[:n_train]
        elif self.split == 'val':
            return indices[n_train:n_train + n_val]
        elif self.split == 'test':
            return indices[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数：
            idx: 样本索引（在当前 split 中）
        
        返回：
            dict with keys:
                - 'image': torch.Tensor (1, 1, 512, 512)
                - 'keypoints': np.ndarray (56, 3)
                - 'case_name': str
                - 'original_shape': tuple (H, W, D)
        """
        # 获取这个 split 中的实际文件索引
        actual_idx = self.indices[idx]
        img_path, json_path, case_name = self.file_list[actual_idx]
        
        # 读取原始图像和关键点
        img_nib = load_nifti_image(str(img_path))
        image = img_nib.get_fdata().astype(np.float32)
        keypoints, labels = read_keypoints_from_json(json_path, img_nib)
        
        original_shape = image.shape
        
        # 应用预处理（包括可选的数据增强）
        img_tensor, transformed_kps = self.preprocessor.preprocess(image, keypoints)

        # 将关键点坐标系对齐到图像显示坐标（与 plt.imshow 的显示方向一致）。
        if transformed_kps.shape[0] > 0:
            target_h, target_w = self.preprocessor.target_size
            transformed_kps = transformed_kps.copy()
            transformed_kps[:, 0] = (target_h - 1) - transformed_kps[:, 0]
            transformed_kps[:, 1] = (target_w - 1) - transformed_kps[:, 1]
            transformed_kps = transformed_kps.astype(np.float32)
        
        return {
            'image': img_tensor,              # (1, 1, 512, 512)
            'keypoints': transformed_kps,     # (56, 3)
            'case_name': case_name,
            'original_shape': original_shape,
            'labels': labels                  # 关键点标签（可选）
        }


def collate_fn_cervical(batch):
    """
    自定义 collate_fn 用于 DataLoader
    处理可变长度的关键点（padding 到最大长度）
    
    参数：
        batch: list of dicts，每个 dict 来自 __getitem__
    
    返回：
        dict with batched tensors
    """
    images = torch.cat([item['image'] for item in batch], dim=0)  # (B, 1, 512, 512)
    
    # 关键点处理 - 支持可变长度
    # 找到最大关键点数量
    max_kps = max(item['keypoints'].shape[0] for item in batch)
    
    keypoints_list = []
    for item in batch:
        kps = item['keypoints']  # (N, 3)
        if kps.shape[0] < max_kps:
            # Pad 关键点到最大长度（用 0 或 NaN 填充）
            padding = np.zeros((max_kps - kps.shape[0], 3), dtype=np.float32)
            kps_padded = np.vstack([kps, padding])
        else:
            kps_padded = kps
        keypoints_list.append(torch.from_numpy(kps_padded).float())
    
    keypoints_tensor = torch.stack(keypoints_list)  # (B, max_kps, 3)
    
    case_names = [item['case_name'] for item in batch]
    original_shapes = [item['original_shape'] for item in batch]
    
    return {
        'image': images,                    # (B, 1, 512, 512)
        'keypoints': keypoints_tensor,      # (B, max_kps, 3)
        'case_names': case_names,
        'original_shapes': original_shapes
    }


# ============ 演示和测试 ============

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    from datetime import datetime
    from preprocess import save_visualization
    
    print("\n" + "=" * 80)
    print("CervicalSpineDataset Demo and Testing")
    print("=" * 80)
    
    # Create train, val, test datasets
    print("\n[1] Creating dataset objects")
    print("-" * 80)
    
    train_dataset = CervicalSpineDataset(
        data_dir='data/',
        split='train',
        train_size=0.7,
        val_size=0.15,
        augmentation=False
    )
    
    val_dataset = CervicalSpineDataset(
        data_dir='data/',
        split='val',
        train_size=0.7,
        val_size=0.15,
        augmentation=False
    )
    
    test_dataset = CervicalSpineDataset(
        data_dir='data/',
        split='test',
        train_size=0.7,
        val_size=0.15,
        augmentation=False
    )
    
    print("\n[2] Dataset statistics")
    print("-" * 80)
    print(f"Total cases: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")
    
    print("\n[3] Single sample retrieval")
    print("-" * 80)
    
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Keypoints shape: {sample['keypoints'].shape}")
    print(f"  Case name: {sample['case_name']}")
    print(f"  Original shape: {sample['original_shape']}")
    
    print("\n[4] Batch processing test (DataLoader)")
    print("-" * 80)
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_cervical
    )
    
    # Iterate one batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"  Batch images shape: {batch['image'].shape}")
    print(f"  Batch keypoints shape: {batch['keypoints'].shape}")
    print(f"  Batch size: {len(batch['case_names'])}")
    print(f"  Case names: {batch['case_names']}")
    
    print("\n[5] Preprocessing verification")
    print("-" * 80)
    
    sample = train_dataset[0]
    img = sample['image']
    kps = sample['keypoints']
    
    print(f"Image tensor:")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Value range: [{img.min():.3f}, {img.max():.3f}]")
    
    print(f"Keypoints array:")
    print(f"  Shape: {kps.shape}")
    print(f"  Dtype: {kps.dtype}")
    print(f"  X range: [{kps[:, 0].min():.1f}, {kps[:, 0].max():.1f}]")
    print(f"  Y range: [{kps[:, 1].min():.1f}, {kps[:, 1].max():.1f}]")
    
    print("\n[6] Sample visualization")
    print("-" * 80)
    
    # Visualization
    fig = plt.figure(figsize=(10, 8))
    img_display = sample['image'][0, 0].numpy()
    
    plt.imshow(img_display, cmap='gray')
    
    # Plot keypoints
    kps_display = sample['keypoints']
    plt.scatter(kps_display[:, 1], kps_display[:, 0], 
               c='red', s=30, alpha=0.6, edgecolors='yellow', linewidth=0.5)
    plt.title(f"Sample: {sample['case_name']}")
    plt.colorbar()
    plt.tight_layout()
    
    # Save visualization
    from preprocess import save_visualization
    vis_path = save_visualization(fig, sample['case_name'], tag="dataset_sample")
    print(f"Sample visualization saved to: {vis_path}")
    
    print("\n[7] Data augmentation test (if enabled)")
    print("-" * 80)
    
    # Create training dataset with augmentation
    train_dataset_aug = CervicalSpineDataset(
        data_dir='data/',
        split='train',
        augmentation=True
    )
    
    # Get same sample multiple times to see augmentation effect
    print(f"Getting same sample (idx=0) multiple times (augmentation enabled):")
    for i in range(3):
        sample = train_dataset_aug[0]
        print(f"  Augmented version {i+1}: image mean={sample['image'].mean():.4f}")
    
    print("\n" + "=" * 80)
    print("Dataset testing completed!")
    print("=" * 80 + "\n")
