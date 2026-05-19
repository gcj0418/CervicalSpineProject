import sys
import time
from pathlib import Path

sys.path.insert(0, 'hrnet-vld_portable')
from inference.roi_predictor import BuiltinROIPredictor

try:
    import SimpleITK as sitk
except ImportError:
    print("SimpleITK not available")
    sys.exit(1)

import numpy as np
import torch

test_list = Path('D-CeLR/data/renji_npy_direct/test.txt')
samples = []
with test_list.open() as f:
    for line in f:
        rel = line.strip()
        if not rel:
            continue
        img_path = Path('D-CeLR') / rel
        img_id = img_path.name[:-5]
        samples.append((img_id, img_path))
        if len(samples) >= 5:
            break

temp_dir = Path('temp_nii_test')
temp_dir.mkdir(exist_ok=True)
nii_paths = []
for img_id, img_path in samples:
    arr = np.load(img_path)
    arr_2d = arr[:, :, 0] if arr.ndim == 3 else arr
    img = sitk.GetImageFromArray(arr_2d.astype(np.float32))
    nii_out = temp_dir / f'{img_id}.nii.gz'
    sitk.WriteImage(img, str(nii_out))
    nii_paths.append(nii_out)

print(f'Testing nnU-Net on {len(nii_paths)} samples...')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch version: {torch.__version__}')

predictor = BuiltinROIPredictor(device='cuda')
mask_dir = Path('temp_mask_test')
mask_dir.mkdir(exist_ok=True)

t0 = time.time()
mask_paths = predictor.predict_masks(nii_paths, mask_dir)
t1 = time.time()

print(f'Masks generated: {len(mask_paths)}/{len(nii_paths)}')
print(f'Time: {t1-t0:.1f}s')
print(f'Speed: {(t1-t0)/len(nii_paths):.1f}s per image')
