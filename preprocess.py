import numpy as np
from PIL import Image
from PIL.Image import Resampling
from pathlib import Path
import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合脚本环境
import matplotlib.pyplot as plt
from datetime import datetime
import json


# ============ 输出管理工具 ============

def create_output_dirs():
    """创建输出目录结构"""
    output_base = Path(__file__).resolve().parent / "outputs"
    preprocess_dir = output_base / "preprocess"
    visualizations_dir = preprocess_dir / "visualizations"
    metadata_dir = preprocess_dir / "metadata"
    
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    return preprocess_dir, visualizations_dir, metadata_dir


def get_safe_filename(case_name):
    """将案例名称转换为安全的文件名（去除特殊字符）"""
    # 仅保留字母、数字、下划线、短横线
    safe_name = "".join(c if c.isalnum() or c in '_-' else '_' for c in case_name)
    return safe_name


def save_visualization(fig, case_name, tag="preprocess"):
    """
    规范化保存可视化图像
    
    参数：
        fig: matplotlib figure 对象
        case_name: 案例名称
        tag: 图像标签（例如 "preprocess", "augmentation" 等）
    
    返回：
        saved_path: 保存的文件路径
    """
    _, vis_dir, _ = create_output_dirs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_case_name = get_safe_filename(case_name)
    filename = f"{tag}_{safe_case_name}_{timestamp}.png"
    
    output_path = vis_dir / filename
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def save_metadata(case_name, image_shape, keypoints_shape, preprocessing_params, tag="preprocess"):
    """
    保存预处理的元数据和参数
    
    参数：
        case_name: 案例名称
        image_shape: 图像形状
        keypoints_shape: 关键点形状
        preprocessing_params: 预处理参数字典
        tag: 元数据标签
    """
    _, _, metadata_dir = create_output_dirs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_case_name = get_safe_filename(case_name)
    filename = f"{tag}_{safe_case_name}_{timestamp}.json"
    
    metadata = {
        "case_name": case_name,
        "timestamp": timestamp,
        "image_shape": image_shape,
        "keypoints_shape": keypoints_shape,
        "preprocessing_params": preprocessing_params,
        "output_image_shape": [1, 1, 512, 512],
        "output_keypoints_shape": [keypoints_shape[0], 3]
    }
    
    output_path = metadata_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return output_path


# ============ 预处理类定义 ============

class ImagePreprocessor:
    """医学X光图像预处理器"""
    
    def __init__(self, 
                 target_size=512,
                 normalize_method='percentile',
                 percentile_range=(2, 98),
                 percentile_sample_size=0,
                 resize_backend='pil'):
        """
        参数：
            target_size: 目标尺寸（假设正方形）或元组 (H, W)
            normalize_method: 'percentile' 或 'minmax'
            percentile_range: 用于百分位数归一化的范围
        """
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = tuple(target_size)
        
        self.normalize_method = normalize_method
        self.percentile_range = percentile_range
        self.percentile_sample_size = int(percentile_sample_size)
        self.resize_backend = resize_backend

    def _estimate_percentiles(self, image):
        """Estimate percentiles with optional random sampling for speed."""
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            return 0.0, 1.0

        sample_size = self.percentile_sample_size
        if sample_size > 0 and finite_values.size > sample_size:
            # Sample without replacement to keep percentile stats stable.
            rng = np.random.default_rng()
            indices = rng.choice(finite_values.size, size=sample_size, replace=False)
            percentile_values = finite_values[indices]
        else:
            percentile_values = finite_values

        vmin = float(np.percentile(percentile_values, self.percentile_range[0]))
        vmax = float(np.percentile(percentile_values, self.percentile_range[1]))

        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-8:
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1.0

        return vmin, vmax
    
    def normalize_image(self, image):
        """
        图像强度归一化
        
        参数：
            image: numpy array (H, W) 或 (H, W, 1)
        
        返回：
            归一化后的 numpy array，值域 [0, 1]
        """
        # 展平为 2D
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.ndim != 2:
            raise ValueError(f"Expected 2D or (H, W, 1) image, got {image.shape}")
        
        image = image.copy().astype(np.float32)
        
        if self.normalize_method == 'percentile':
            # 百分位数归一化（适合医学影像）
            vmin, vmax = self._estimate_percentiles(image)
            image = np.clip(image, vmin, vmax)
            image = (image - vmin) / (vmax - vmin + 1e-8)
        
        elif self.normalize_method == 'minmax':
            # MinMax 归一化
            vmin = image.min()
            vmax = image.max()
            image = (image - vmin) / (vmax - vmin + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalize method: {self.normalize_method}")
        
        return np.clip(image, 0, 1).astype(np.float32)
    
    def resize_image(self, image, return_scale=False):
        """
        调整图像大小到目标尺寸
        
        参数：
            image: numpy array (H, W) 或 (H, W, 1)
            return_scale: 是否返回缩放因子
        
        返回：
            resized_image: numpy array (target_size[0], target_size[1])
            scale_factors: (scale_h, scale_w) 如果 return_scale=True
        """
        # 展平为 2D
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        elif image.ndim != 2:
            raise ValueError(f"Expected 2D or (H, W, 1) image, got {image.shape}")
        
        orig_h, orig_w = image.shape
        target_h, target_w = self.target_size

        resized = None
        if self.resize_backend == 'opencv':
            try:
                import cv2

                interpolation = cv2.INTER_AREA if (target_h < orig_h or target_w < orig_w) else cv2.INTER_LINEAR
                resized = cv2.resize(image.astype(np.float32), (target_w, target_h), interpolation=interpolation).astype(np.float32)
            except Exception:
                resized = None

        if resized is None:
            # PIL 回退路径
            pil_img = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8))
            pil_img = pil_img.resize((target_w, target_h), Resampling.LANCZOS)
            resized = np.array(pil_img).astype(np.float32)

            # 如果原始图片是 [0, 1]，转换回相同范围
            if image.max() <= 1:
                resized = resized / 255.0
        
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        
        if return_scale:
            return resized, (scale_h, scale_w)
        else:
            return resized
    
    def transform_keypoints(self, keypoints, scale_factors):
        """
        根据图像 resize 的缩放因子调整关键点坐标
        
        参数：
            keypoints: numpy array (N, 3)，其中 (x, y, z) 为体素坐标
            scale_factors: (scale_h, scale_w) 或 (scale_x, scale_y, scale_z)
        
        返回：
            transformed_keypoints: numpy array (N, 3)
        """
        if keypoints.shape[0] == 0:
            return keypoints.copy()
        
        keypoints = keypoints.copy().astype(np.float32)
        
        if len(scale_factors) == 2:
            # (scale_h, scale_w) -> (scale_x, scale_y, scale_z)
            # x 对应 height, y 对应 width, z 不变
            scale_h, scale_w = scale_factors
            keypoints[:, 0] *= scale_h  # x 轴（height方向）
            keypoints[:, 1] *= scale_w  # y 轴（width方向）
            # z 坐标保持不变
        
        elif len(scale_factors) == 3:
            scale_x, scale_y, scale_z = scale_factors
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y
            keypoints[:, 2] *= scale_z
        
        else:
            raise ValueError(f"Expected 2 or 3 scale factors, got {len(scale_factors)}")
        
        return keypoints
    
    def preprocess(self, image, keypoints):
        """
        完整的预处理流程：归一化 -> resize -> 关键点转换
        
        参数：
            image: numpy array
            keypoints: numpy array (N, 3)
        
        返回：
            processed_image: torch tensor (1, H, W)
            transformed_keypoints: numpy array (N, 3)
        """
        # 1. 归一化
        normalized_img = self.normalize_image(image)
        
        # 2. Resize
        resized_img, scale_factors = self.resize_image(normalized_img, return_scale=True)
        
        # 3. 关键点转换
        transformed_kps = self.transform_keypoints(keypoints, scale_factors)
        
        # 4. 转换为 torch tensor (1, 1, H, W) - PyTorch (B, C, H, W) 标准格式
        img_tensor = torch.from_numpy(resized_img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return img_tensor, transformed_kps


class AugmentedPreprocessor(ImagePreprocessor):
    """带数据增强的预处理器"""
    
    def __init__(self, 
                 target_size=512,
                 normalize_method='percentile',
                 percentile_range=(2, 98),
                 augmentation=False,
                 augmentation_params=None):
        """
        参数：
            augmentation: 是否启用数据增强
            augmentation_params: 增强参数字典
        """
        super().__init__(target_size, normalize_method, percentile_range)
        self.augmentation = augmentation
        
        if augmentation_params is None:
            augmentation_params = {
                'rotation_range': 15,  # 度数
                'elastic_alpha': 30,  # 弹性变形强度
                'elastic_sigma': 3,
                # 默认关闭弹性形变：若不同时变换关键点，会引入标签噪声
                'elastic_prob': 0.0,
                'gamma_prob': 0.3,
                'gamma_range': (0.9, 1.1),
                'noise_prob': 0.2,
                'noise_std': 0.01,
            }
        
        self.augmentation_params = augmentation_params
    
    def _rotate_keypoints(self, keypoints, matrix, image_shape):
        """使用与图像相同的旋转矩阵变换关键点。"""
        if keypoints.shape[0] == 0:
            return keypoints

        kps = keypoints.copy().astype(np.float32)

        # 关键点格式为 (x, y, z)，其中 x=行坐标，y=列坐标
        # OpenCV 旋转矩阵作用于 (col, row)
        coords = np.stack([kps[:, 1], kps[:, 0], np.ones(kps.shape[0], dtype=np.float32)], axis=1)
        rotated = (matrix @ coords.T).T

        kps[:, 1] = rotated[:, 0]  # y <- col
        kps[:, 0] = rotated[:, 1]  # x <- row

        h, w = image_shape
        kps[:, 0] = np.clip(kps[:, 0], 0, h - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, w - 1)

        return kps

    def apply_augmentation(self, image, keypoints):
        """
        应用数据增强（旋转 + 强度扰动）
        
        参数：
            image: numpy array (H, W)
            keypoints: numpy array (N, 3)
        
        返回：
            augmented_image: numpy array (H, W)
            augmented_keypoints: numpy array (N, 3)
        """
        import cv2
        
        image = image.copy()
        keypoints = keypoints.copy().astype(np.float32)
        
        # 随机旋转
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.augmentation_params['rotation_range'],
                                     self.augmentation_params['rotation_range'])
            h, w = image.shape
            center = (w / 2, h / 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            keypoints = self._rotate_keypoints(keypoints, matrix, (h, w))

        # Gamma 亮度对比度扰动（不改变几何）
        if np.random.random() < self.augmentation_params.get('gamma_prob', 0.0):
            gamma_min, gamma_max = self.augmentation_params.get('gamma_range', (0.9, 1.1))
            gamma = np.random.uniform(gamma_min, gamma_max)
            image = np.power(np.clip(image, 0, 1), gamma)

        # 高斯噪声（不改变几何）
        if np.random.random() < self.augmentation_params.get('noise_prob', 0.0):
            noise_std = float(self.augmentation_params.get('noise_std', 0.01))
            image = np.clip(image + np.random.normal(0, noise_std, size=image.shape).astype(np.float32), 0, 1)
        
        # 弹性变形
        return image, keypoints
    
    def preprocess(self, image, keypoints):
        """
        带增强的完整预处理
        """
        # 1. 归一化
        normalized_img = self.normalize_image(image)
        
        # 2. 数据增强（可选）
        if self.augmentation:
            normalized_img, keypoints = self.apply_augmentation(normalized_img, keypoints)
        
        # 3. Resize
        resized_img, scale_factors = self.resize_image(normalized_img, return_scale=True)
        
        # 4. 关键点转换
        transformed_kps = self.transform_keypoints(keypoints, scale_factors)
        
        # 5. 转换为 torch tensor (1, 1, H, W)
        img_tensor = torch.from_numpy(resized_img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return img_tensor, transformed_kps


def preprocess_batch(image_list, keypoints_list, preprocessor):
    """
    批量预处理
    
    参数：
        image_list: 图像 numpy array 列表
        keypoints_list: 关键点 numpy array 列表
        preprocessor: ImagePreprocessor 实例
    
    返回：
        images: torch tensor (B, 1, H, W)
        keypoints: list of numpy arrays
    """
    images = []
    processed_keypoints = []
    
    for img, kps in zip(image_list, keypoints_list):
        img_tensor, transformed_kps = preprocessor.preprocess(img, kps)
        images.append(img_tensor)
        processed_keypoints.append(transformed_kps)
    
    # 堆叠图像 - 每个 img_tensor 是 (1, 1, H, W)
    images_batch = torch.cat(images, dim=0)  # (B, 1, H, W)
    
    return images_batch, processed_keypoints


# ============ 演示和测试 ============

if __name__ == "__main__":
    from data_loader import read_keypoints_from_json, load_nifti_image
    import matplotlib.pyplot as plt
    
    # 自动发现数据目录
    data_base = Path(__file__).resolve().parent / "data"
    data_dirs = sorted([d for d in data_base.iterdir() if d.is_dir()])
    
    if not data_dirs:
        print("Error: No data directories found in ./data/")
        exit(1)
    
    # 使用第一个数据目录
    data_dir = data_dirs[0]
    
    # 找到 .nii.gz 文件
    nii_files = list(data_dir.glob("*.nii.gz")) + list(data_dir.glob("*.nii"))
    if not nii_files:
        print(f"Error: No .nii files found in {data_dir.name}!")
        exit(1)
    
    img_path = nii_files[0]
    img_nib = load_nifti_image(str(img_path))
    image = img_nib.get_fdata().astype(np.float32)
    
    # 加载关键点
    json_files = list(data_dir.glob("*.json"))
    json_path = json_files[0] if json_files else None
    
    if json_path:
        keypoints, labels = read_keypoints_from_json(json_path, img_nib)
    else:
        keypoints = np.array([])
    
    print(f"Original image shape: {image.shape}")
    print(f"Original keypoints shape: {keypoints.shape}")
    
    # 创建预处理器
    preprocessor = ImagePreprocessor(target_size=512, normalize_method='percentile')
    
    # 执行预处理
    img_tensor, transformed_kps = preprocessor.preprocess(image, keypoints)
    
    print(f"\nAfter preprocessing:")
    print(f"  Processed image shape: {img_tensor.shape}")
    print(f"  Processed image range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    print(f"  Transformed keypoints shape: {transformed_kps.shape}")
    
    # 可视化
    if img_tensor.shape[0] > 0 and img_tensor.shape[1] > 0:
        processed_img = img_tensor[0, 0, :, :].numpy()  # (1, 1, H, W) -> (H, W)
        
        fig = plt.figure(figsize=(14, 6))
        
        # 原始图像
        plt.subplot(1, 2, 1)
        original_display = image[:, :, 0] if image.ndim == 3 else image
        vmin = np.percentile(original_display, 2)
        vmax = np.percentile(original_display, 98)
        plt.imshow(original_display, cmap='gray', vmin=vmin, vmax=vmax)
        if len(keypoints) > 0:
            plt.scatter(512 - keypoints[:, 1], 512 - keypoints[:, 0], 
                       c='red', s=50, alpha=0.6)
        plt.title(f"Original ({original_display.shape[0]}×{original_display.shape[1]})")
        plt.colorbar()
        
        # 预处理后的图像
        plt.subplot(1, 2, 2)
        plt.imshow(processed_img, cmap='gray')
        if len(transformed_kps) > 0:
            plt.scatter(512 - transformed_kps[:, 1], 512 - transformed_kps[:, 0], 
                       c='lime', s=50, alpha=0.6, edgecolors='red', linewidth=1)
        plt.title(f"Preprocessed ({processed_img.shape[0]}×{processed_img.shape[1]})")
        plt.colorbar()
        
        plt.tight_layout()
        
        # 规范化保存可视化
        case_name = img_path.parent.name  # 从目录名提取案例名称
        vis_path = save_visualization(fig, case_name, tag="preprocess")
        
        # 保存元数据
        preprocessing_params = {
            "target_size": preprocessor.target_size,
            "normalize_method": preprocessor.normalize_method,
            "percentile_range": preprocessor.percentile_range,
        }
        meta_path = save_metadata(
            case_name, 
            image.shape, 
            keypoints.shape,
            preprocessing_params,
            tag="preprocess"
        )
        
        print(f"\nVisualization saved to: {vis_path.relative_to(Path(__file__).resolve().parent)}")
        print(f"Metadata saved to: {meta_path.relative_to(Path(__file__).resolve().parent)}")
