"""
完整验证 dataset.py 的功能有效性

包含 8 个测试套件：
1. 文件扫描和发现
2. 数据分割正确性
3. 单样本加载
4. 批处理功能
5. 数据类型和范围验证
6. 数据增强效果
7. 可重现性（seed）
8. 错误处理和边界情况
"""

import numpy as np
import torch
from dataset import CervicalSpineDataset, collate_fn_cervical
from torch.utils.data import DataLoader
import sys

def print_header(title):
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")

def print_test(name, result, details=""):
    status = "[PASS]" if result else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        print(f"      -> {details}")

def print_section(name):
    print(f"\n[{name}]")
    print("-"*80)

# ============================================================================
# Test Suite 1: 文件扫描和发现
# ============================================================================

def test_file_discovery():
    print_section("Test 1: 文件扫描和发现")
    
    try:
        dataset = CervicalSpineDataset(data_dir='data/', split='train')
        
        # [1.1] 检查是否找到文件
        found_files = len(dataset.file_list)
        test_1_1 = found_files > 0
        print_test("1.1 能否找到数据文件", test_1_1, f"找到 {found_files} 个案例")
        
        # [1.2] 检查每个文件对是否完整
        all_valid = True
        for img_path, json_path, case_name in dataset.file_list:
            if not img_path.exists() or not json_path.exists():
                all_valid = False
                break
        print_test("1.2 所有文件对是否存在", all_valid, f"验证 {found_files} 个文件对")
        
        # [1.3] 检查案例名称是否正确
        case_names_valid = all(isinstance(name, str) and len(name) > 0 
                               for _, _, name in dataset.file_list)
        print_test("1.3 案例名称是否有效", case_names_valid)
        
        return all([test_1_1, all_valid, case_names_valid])
    
    except Exception as e:
        print_test("1.1 能否找到数据文件", False, f"错误: {str(e)}")
        return False

# ============================================================================
# Test Suite 2: 数据分割正确性
# ============================================================================

def test_data_splitting():
    print_section("Test 2: 数据分割正确性")
    
    try:
        # [2.1] 三个分割的数据总和是否等于总数
        train_ds = CervicalSpineDataset(data_dir='data/', split='train', seed=42)
        val_ds = CervicalSpineDataset(data_dir='data/', split='val', seed=42)
        test_ds = CervicalSpineDataset(data_dir='data/', split='test', seed=42)
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        total_cases = len(train_ds.file_list)
        test_2_1 = total == total_cases
        print_test("2.1 分割总和是否等于总数", test_2_1, 
                  f"{len(train_ds)}+{len(val_ds)}+{len(test_ds)}={total} vs {total_cases}")
        
        # [2.2] 分割比例是否正确（70:15:15）
        expected_train = int(total_cases * 0.7)
        expected_val = int(total_cases * 0.15)
        expected_test = total_cases - expected_train - expected_val
        
        test_2_2 = (len(train_ds) == expected_train and 
                    len(val_ds) == expected_val and 
                    len(test_ds) == expected_test)
        print_test("2.2 分割比例是否正确 (0.7:0.15:0.15)", test_2_2,
                  f"Train:{len(train_ds)}/{expected_train}, Val:{len(val_ds)}/{expected_val}, Test:{len(test_ds)}/{expected_test}")
        
        # [2.3] 不同分割的索引是否互不重叠
        train_indices = set(train_ds.indices)
        val_indices = set(val_ds.indices)
        test_indices = set(test_ds.indices)
        
        no_overlap = (len(train_indices & val_indices) == 0 and
                      len(train_indices & test_indices) == 0 and
                      len(val_indices & test_indices) == 0)
        print_test("2.3 索引是否互不重叠", no_overlap)
        
        # [2.4] Seed 是否保证可重现性
        train_ds2 = CervicalSpineDataset(data_dir='data/', split='train', seed=42)
        reproducible = np.array_equal(train_ds.indices, train_ds2.indices)
        print_test("2.4 Seed 是否保证可重现性", reproducible)
        
        # [2.5] 不同 seed 是否产生不同分割
        train_ds_diff = CervicalSpineDataset(data_dir='data/', split='train', seed=123)
        different = not np.array_equal(train_ds.indices, train_ds_diff.indices)
        print_test("2.5 不同 seed 是否产生不同分割", different)
        
        return all([test_2_1, test_2_2, no_overlap, reproducible, different])
    
    except Exception as e:
        print_test("2.1 分割总和是否等于总数", False, f"错误: {str(e)}")
        return False

# ============================================================================
# Test Suite 3: 单样本加载
# ============================================================================

def test_single_sample_loading():
    print_section("Test 3: 单样本加载")
    
    try:
        dataset = CervicalSpineDataset(data_dir='data/', split='train')
        
        # [3.1] 能否加载样本
        sample = dataset[0]
        test_3_1 = sample is not None
        print_test("3.1 能否加载样本", test_3_1)
        
        # [3.2] 返回值是否为字典
        test_3_2 = isinstance(sample, dict)
        print_test("3.2 返回值是否为字典", test_3_2, f"类型: {type(sample)}")
        
        # [3.3] 字典是否包含必要的键
        required_keys = {'image', 'keypoints', 'case_name', 'original_shape', 'labels'}
        has_all_keys = required_keys.issubset(sample.keys())
        print_test("3.3 是否包含必要的键", has_all_keys, 
                  f"键: {list(sample.keys())}")
        
        # [3.4] image 是否为正确的形状和类型
        correct_image_shape = sample['image'].shape == torch.Size([1, 1, 512, 512])
        correct_image_dtype = sample['image'].dtype == torch.float32
        test_3_4 = correct_image_shape and correct_image_dtype
        print_test("3.4 image 形状和类型是否正确", test_3_4,
                  f"形状: {sample['image'].shape}, 类型: {sample['image'].dtype}")
        
        # [3.5] keypoints 是否为正确的形状和类型
        correct_kp_shape = sample['keypoints'].shape == (56, 3)
        correct_kp_dtype = sample['keypoints'].dtype == np.float32
        test_3_5 = correct_kp_shape and correct_kp_dtype
        print_test("3.5 keypoints 形状和类型是否正确", test_3_5,
                  f"形状: {sample['keypoints'].shape}, 类型: {sample['keypoints'].dtype}")
        
        # [3.6] case_name 是否为字符串
        test_3_6 = isinstance(sample['case_name'], str) and len(sample['case_name']) > 0
        print_test("3.6 case_name 是否有效", test_3_6, f"名称: {sample['case_name']}")
        
        # [3.7] original_shape 是否为元组
        test_3_7 = isinstance(sample['original_shape'], tuple) and len(sample['original_shape']) >= 2
        print_test("3.7 original_shape 是否有效", test_3_7, 
                  f"形状: {sample['original_shape']}")
        
        # [3.8] labels 是否为列表
        test_3_8 = isinstance(sample['labels'], list) and len(sample['labels']) == 56
        print_test("3.8 labels 是否有效", test_3_8, 
                  f"长度: {len(sample.get('labels', []))}")
        
        return all([test_3_1, test_3_2, has_all_keys, test_3_4, test_3_5, test_3_6, test_3_7, test_3_8])
    
    except Exception as e:
        print_test("3.1 能否加载样本", False, f"错误: {str(e)}")
        return False

# ============================================================================
# Test Suite 4: 批处理功能
# ============================================================================

def test_batch_processing():
    print_section("Test 4: 批处理功能")
    
    try:
        dataset = CervicalSpineDataset(data_dir='data/', split='train')
        loader = DataLoader(dataset, batch_size=4, shuffle=False, 
                           collate_fn=collate_fn_cervical)
        
        # [4.1] 能否创建 DataLoader
        test_4_1 = loader is not None
        print_test("4.1 能否创建 DataLoader", test_4_1)
        
        # [4.2] 能否迭代
        batch = next(iter(loader))
        test_4_2 = batch is not None
        print_test("4.2 能否迭代 DataLoader", test_4_2)
        
        # [4.3] 批处理是否包含必要的键
        required_batch_keys = {'image', 'keypoints', 'case_names', 'original_shapes'}
        has_batch_keys = required_batch_keys.issubset(batch.keys())
        print_test("4.3 批处理是否包含必要的键", has_batch_keys,
                  f"键: {list(batch.keys())}")
        
        # [4.4] 批图像形状是否正确
        correct_batch_image_shape = batch['image'].shape == torch.Size([4, 1, 512, 512])
        print_test("4.4 批图像形状是否正确", correct_batch_image_shape,
                  f"形状: {batch['image'].shape}")
        
        # [4.5] 批关键点形状是否正确
        correct_batch_kp_shape = batch['keypoints'].shape == torch.Size([4, 56, 3])
        print_test("4.5 批关键点形状是否正确", correct_batch_kp_shape,
                  f"形状: {batch['keypoints'].shape}")
        
        # [4.6] case_names 是否为列表
        test_4_6 = isinstance(batch['case_names'], list) and len(batch['case_names']) == 4
        print_test("4.6 case_names 是否有效", test_4_6,
                  f"长度: {len(batch['case_names'])}")
        
        # [4.7] original_shapes 是否为列表
        test_4_7 = isinstance(batch['original_shapes'], list) and len(batch['original_shapes']) == 4
        print_test("4.7 original_shapes 是否有效", test_4_7,
                  f"长度: {len(batch['original_shapes'])}")
        
        # [4.8] 能否迭代完整的 epoch
        batch_count = 0
        for _ in loader:
            batch_count += 1
        test_4_8 = batch_count == np.ceil(len(dataset) / 4)
        print_test("4.8 能否迭代完整的 epoch", test_4_8,
                  f"批数: {batch_count}, 预期: {int(np.ceil(len(dataset) / 4))}")
        
        return all([test_4_1, test_4_2, has_batch_keys, correct_batch_image_shape,
                   correct_batch_kp_shape, test_4_6, test_4_7, test_4_8])
    
    except Exception as e:
        print_test("4.1 能否创建 DataLoader", False, f"错误: {str(e)}")
        return False

# ============================================================================
# Test Suite 5: 数据类型和范围验证
# ============================================================================

def test_data_validation():
    print_section("Test 5: 数据类型和范围验证")
    
    try:
        dataset = CervicalSpineDataset(data_dir='data/', split='train')
        
        # 收集多个样本的统计信息
        image_mins = []
        image_maxs = []
        kp_x_mins, kp_x_maxs = [], []
        kp_y_mins, kp_y_maxs = [], []
        
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            img = sample['image'].numpy()
            kps = sample['keypoints']
            
            image_mins.append(img.min())
            image_maxs.append(img.max())
            kp_x_mins.append(kps[:, 0].min())
            kp_x_maxs.append(kps[:, 0].max())
            kp_y_mins.append(kps[:, 1].min())
            kp_y_maxs.append(kps[:, 1].max())
        
        # [5.1] 图像值是否在 [0, 1] 范围内
        img_in_range = all(m >= 0 for m in image_mins) and all(m <= 1 for m in image_maxs)
        print_test("5.1 图像值是否在 [0, 1] 范围内", img_in_range,
                  f"范围: [{min(image_mins):.3f}, {max(image_maxs):.3f}]")
        
        # [5.2] 关键点 X 坐标是否在 [0, 512] 范围内
        kp_x_in_range = all(m >= 0 for m in kp_x_mins) and all(m <= 512 for m in kp_x_maxs)
        print_test("5.2 关键点 X 坐标是否在 [0, 512] 范围内", kp_x_in_range,
                  f"范围: [{min(kp_x_mins):.1f}, {max(kp_x_maxs):.1f}]")
        
        # [5.3] 关键点 Y 坐标是否在 [0, 512] 范围内
        kp_y_in_range = all(m >= 0 for m in kp_y_mins) and all(m <= 512 for m in kp_y_maxs)
        print_test("5.3 关键点 Y 坐标是否在 [0, 512] 范围内", kp_y_in_range,
                  f"范围: [{min(kp_y_mins):.1f}, {max(kp_y_maxs):.1f}]")
        
        # [5.4] 关键点 Z 坐标是否存在
        sample = dataset[0]
        kps = sample['keypoints']
        has_z = kps.shape[1] == 3
        print_test("5.4 关键点是否包含 Z 坐标", has_z,
                  f"形状: {kps.shape}")
        
        # [5.5] 是否存在 NaN 或 Inf
        no_nan_inf = True
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            if np.isnan(sample['image'].numpy()).any() or np.isinf(sample['image'].numpy()).any():
                no_nan_inf = False
            if np.isnan(sample['keypoints']).any() or np.isinf(sample['keypoints']).any():
                no_nan_inf = False
        print_test("5.5 数据中是否存在 NaN 或 Inf", no_nan_inf)
        
        return all([img_in_range, kp_x_in_range, kp_y_in_range, has_z, no_nan_inf])
    
    except Exception as e:
        print_test("5.1 图像值是否在 [0, 1] 范围内", False, f"错误: {str(e)}")
        return False

# ============================================================================
# Test Suite 6: 数据增强效果
# ============================================================================

def test_data_augmentation():
    print_section("Test 6: 数据增强效果")
    
    try:
        # [6.1] 能否创建启用增强的数据集
        aug_dataset = CervicalSpineDataset(data_dir='data/', split='train', augmentation=True)
        test_6_1 = aug_dataset.preprocessor is not None
        print_test("6.1 能否创建启用增强的数据集", test_6_1)
        
        # [6.2] 能否创建禁用增强的数据集
        no_aug_dataset = CervicalSpineDataset(data_dir='data/', split='train', augmentation=False)
        test_6_2 = no_aug_dataset.preprocessor is not None
        print_test("6.2 能否创建禁用增强的数据集", test_6_2)
        
        # [6.3] 增强数据集是否产生不同的结果
        samples_aug = [aug_dataset[0] for _ in range(3)]
        means_aug = [s['image'].mean().item() for s in samples_aug]
        augmentation_working = len(set([round(m, 4) for m in means_aug])) >= 2
        print_test("6.3 多次读取是否产生不同结果（增强有效）", augmentation_working,
                  f"平均值: {[f'{m:.4f}' for m in means_aug]}")
        
        # [6.4] 禁用增强数据集是否产生相同的结果
        samples_no_aug = [no_aug_dataset[0] for _ in range(3)]
        means_no_aug = [s['image'].mean().item() for s in samples_no_aug]
        no_augmentation_consistent = len(set([round(m, 6) for m in means_no_aug])) == 1
        print_test("6.4 禁用增强时多次读取结果一致", no_augmentation_consistent,
                  f"平均值: {[f'{m:.6f}' for m in means_no_aug]}")
        
        # [6.5] 验证集不应该有增强
        val_dataset = CervicalSpineDataset(data_dir='data/', split='val', augmentation=True)
        val_samples = [val_dataset[0] for _ in range(3)]
        val_means = [s['image'].mean().item() for s in val_samples]
        val_no_aug = len(set([round(m, 6) for m in val_means])) == 1
        print_test("6.5 验证集即使启用选项也不增强", val_no_aug,
                  f"平均值: {[f'{m:.6f}' for m in val_means]}")
        
        return all([test_6_1, test_6_2, augmentation_working, 
                   no_augmentation_consistent, val_no_aug])
    
    except Exception as e:
        print_test("6.1 能否创建启用增强的数据集", False, f"错误: {str(e)}")
        return False

# ============================================================================
# Test Suite 7: 可重现性（Seed）
# ============================================================================

def test_reproducibility():
    print_section("Test 7: 可重现性（Seed）")
    
    try:
        # [7.1] 相同 seed 是否产生相同分割
        ds1 = CervicalSpineDataset(data_dir='data/', split='train', seed=42)
        ds2 = CervicalSpineDataset(data_dir='data/', split='train', seed=42)
        same_split = np.array_equal(ds1.indices, ds2.indices)
        print_test("7.1 相同 seed 是否产生相同分割", same_split)
        
        # [7.2] 相同 seed 的样本是否相同
        no_aug_ds = CervicalSpineDataset(data_dir='data/', split='train', 
                                        seed=42, augmentation=False)
        sample1 = no_aug_ds[0]
        sample2 = no_aug_ds[0]
        same_content = torch.allclose(sample1['image'], sample2['image'])
        print_test("7.2 相同 seed 的样本是否相同", same_content)
        
        # [7.3] 禁用增强时多次读取是否相同
        samples = [no_aug_ds[0] for _ in range(3)]
        all_same = all(torch.allclose(samples[0]['image'], s['image']) for s in samples[1:])
        print_test("7.3 禁用增强时多次读取是否相同", all_same)
        
        return all([same_split, same_content, all_same])
    
    except Exception as e:
        print_test("7.1 相同 seed 是否产生相同分割", False, f"错误: {str(e)}")
        return False

# ============================================================================
# Test Suite 8: 错误处理和边界情况
# ============================================================================

def test_error_handling():
    print_section("Test 8: 错误处理和边界情况")
    
    try:
        # [8.1] 无效分割是否抛出错误
        try:
            invalid_ds = CervicalSpineDataset(data_dir='data/', split='invalid')
            test_8_1 = False
        except ValueError:
            test_8_1 = True
        except:
            test_8_1 = False
        print_test("8.1 无效分割是否抛出错误", test_8_1)
        
        # [8.2] 超出索引范围时是否抛出错误
        dataset = CervicalSpineDataset(data_dir='data/', split='train')
        try:
            _ = dataset[len(dataset)]
            test_8_2 = False
        except (IndexError, Exception):
            test_8_2 = True
        print_test("8.2 超出索引范围时是否抛出错误", test_8_2)
        
        # [8.3] 负数索引是否正常工作（Python 特性）
        try:
            sample = dataset[-1]
            test_8_3 = sample is not None
        except:
            test_8_3 = False
        print_test("8.3 负数索引是否正常工作", test_8_3)
        
        # [8.4] 不同分割的长度是否正确
        train_ds = CervicalSpineDataset(data_dir='data/', split='train')
        val_ds = CervicalSpineDataset(data_dir='data/', split='val')
        test_ds = CervicalSpineDataset(data_dir='data/', split='test')
        test_8_4 = len(train_ds) > len(val_ds) and len(val_ds) > 0 and len(test_ds) > 0
        print_test("8.4 不同分割的长度是否符合预期", test_8_4,
                  f"Train:{len(train_ds)}, Val:{len(val_ds)}, Test:{len(test_ds)}")
        
        return all([test_8_1, test_8_2, test_8_3, test_8_4])
    
    except Exception as e:
        print_test("8.1 无效分割是否抛出错误", False, f"错误: {str(e)}")
        return False

# ============================================================================
# 主测试函数
# ============================================================================

def run_all_tests():
    print_header("CervicalSpineDataset 完整功能验证测试")
    
    results = {
        '文件扫描和发现': test_file_discovery(),
        '数据分割正确性': test_data_splitting(),
        '单样本加载': test_single_sample_loading(),
        '批处理功能': test_batch_processing(),
        '数据类型和范围验证': test_data_validation(),
        '数据增强效果': test_data_augmentation(),
        '可重现性（Seed）': test_reproducibility(),
        '错误处理和边界情况': test_error_handling(),
    }
    
    # 打印总结
    print_header("测试总结")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\n总计: {passed}/{total} 个测试套件通过\n")
    
    if passed == total:
        print("===== 所有测试通过！dataset.py 功能完全有效！ =====\n")
        return True
    else:
        print(f"===== 有 {total - passed} 个测试套件失败 =====\n")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
