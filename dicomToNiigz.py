import os
import subprocess
from pathlib import Path

INPUT_FOLDER = r"E:\edge\图像处理\颈椎X片数据测试\test_images"
OUTPUT_FOLDER = r"E:\edge\图像处理\颈椎X片数据测试\nifti_output"
DCM2NIIX_PATH = r"E:\edge\dcm2niix_win\dcm2niix.exe"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

dicom_files = list(Path(INPUT_FOLDER).glob("*.dicom")) + list(Path(INPUT_FOLDER).glob("*.dcm"))

print(f"找到 {len(dicom_files)} 个DICOM文件，开始逐个转换...")

for dicom_file in dicom_files:
    cmd = [
        DCM2NIIX_PATH,
        "-z", "y",    
        "-b", "y",   
        "-f", "%s_%f",   
        "-o", OUTPUT_FOLDER, 
        str(dicom_file)      
    ]
    
    subprocess.run(cmd, capture_output=True)
    
    output_niigz = Path(OUTPUT_FOLDER) / f"{dicom_file.stem}.nii.gz"
    if output_niigz.exists():
        print(f"{dicom_file.name} → {output_niigz.name}")
    else:
        print(f"{dicom_file.name} 转换失败")

print(f"\n 共处理 {len(dicom_files)} 个文件")