import os
import nibabel as nib

base_dir = r"E:\edge\图像处理\颈椎X片数据测试"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".nii"):
            nii_path = os.path.join(root, file)
            nii_gz_path = os.path.splitext(nii_path)[0] + ".nii.gz"
            # 如果目标文件已存在则跳过
            if os.path.exists(nii_gz_path):
                print(f"跳过已存在: {nii_gz_path}")
                continue
            try:
                img = nib.load(nii_path)
                nib.save(img, nii_gz_path)
                print(f"转换成功: {nii_path} -> {nii_gz_path}")
                # 可选：删除原文件
                # os.remove(nii_path)
            except Exception as e:
                print(f"转换失败 {nii_path}: {e}")