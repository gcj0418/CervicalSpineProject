"""
Clinical parameters calculation package for cervical spine.

This package provides pure mathematical computation of six sagittal-plane
clinical parameters.  It assumes the caller has already:
  1. Run HRNet + VLD inference.
  2. Extracted fused landmarks for Cobb/SVA/etc:
       - pts (28, 2)  — VLD-selected 7-vertebra bounding boxes
  3. Kept VLD raw output for facet-joint calculation:
       - vld_pts (N*4, 2)  — all detected vertebrae, sorted top-to-bottom
  4. Converted NIfTI voxel spacing into a scalar mm/pixel value.

Model-output parsing, coordinate transformation, Hungarian matching,
and left/right orientation unification are handled externally
(inference.py / gui.py), NOT inside this package.
"""
from .cobb_angle import (
    compute_c2c7_lordosis,
    compute_max_cobb,
    diagnose,
    draw_cobb,
    normalize_corners,
    extract_vertebra_boxes,
)
from .sva import compute_c2c7_sva, draw_sva
from .t1_slope import compute_t1_slope, draw_t1_slope
from .disc_height import compute_disc_heights, draw_disc_heights
from .vertebral_displacement import compute_vertebral_displacement, draw_displacements
from .facet_joint_angle import compute_facet_joint_angles, draw_facet_angles
