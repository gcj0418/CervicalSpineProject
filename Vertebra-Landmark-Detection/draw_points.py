import cv2
import numpy as np

# Unified BGR palette (same as HRNet visualize_spine.py), extended to 14 colors
COLORS = [
    (195, 18, 251),
    (138, 209, 242),
    (20, 203, 39),
    (238, 229, 25),
    (187, 27, 251),
    (4, 183, 240),
    (138, 209, 115),
    (213, 249, 31),
    (185, 59, 76),
    (98, 242, 252),
    (255, 102, 0),
    (0, 128, 255),
    (128, 0, 128),
    (0, 200, 128),
]


def draw_landmarks_regress_test(pts0, ori_image_regress, ori_image_points):
    num_pts = min(len(pts0), len(COLORS))
    for i, pt in enumerate(pts0[:num_pts]):
        color_255 = COLORS[i % len(COLORS)]
        cv2.circle(ori_image_regress, (int(pt[0]), int(pt[1])), 3, color_255, -1, 1)
        # cv2.circle(ori_image, (int(pt[2]), int(pt[3])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[4]), int(pt[5])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[6]), int(pt[7])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[8]), int(pt[9])), 5, color_255, -1,1)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[4]), int(pt[5])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[6]), int(pt[7])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[8]), int(pt[9])), color_255, 2, 1,
                        tipLength=0.2)
        # cv2.circle(ori_image, (int(pt[0]), int(pt[1])), 6, (255,255,255), -1,1)
        cv2.circle(ori_image_points, (int(pt[2]), int(pt[3])), 3, color_255, -1, 1)
        cv2.circle(ori_image_points, (int(pt[4]), int(pt[5])), 3, color_255, -1, 1)
        cv2.circle(ori_image_points, (int(pt[6]), int(pt[7])), 3, color_255, -1, 1)
        cv2.circle(ori_image_points, (int(pt[8]), int(pt[9])), 3, color_255, -1, 1)
    return ori_image_regress, ori_image_points


def draw_landmarks_compare_test(gt_pts, pred_pts, ori_image_compare):
    gt_pts = np.asarray(gt_pts, np.float32)
    pred_pts = np.asarray(pred_pts, np.float32)

    for i, pt in enumerate(gt_pts):
        color_255 = (0, 255, 0)
        cv2.circle(ori_image_compare, (int(pt[0]), int(pt[1])), 4, color_255, 2, 1)

    for i, pt in enumerate(pred_pts):
        color_255 = COLORS[i % len(COLORS)]
        cv2.circle(ori_image_compare, (int(pt[0]), int(pt[1])), 3, color_255, -1, 1)
        cv2.arrowedLine(ori_image_compare, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color_255, 1, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_compare, (int(pt[0]), int(pt[1])), (int(pt[4]), int(pt[5])), color_255, 1, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_compare, (int(pt[0]), int(pt[1])), (int(pt[6]), int(pt[7])), color_255, 1, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_compare, (int(pt[0]), int(pt[1])), (int(pt[8]), int(pt[9])), color_255, 1, 1,
                        tipLength=0.2)

    return ori_image_compare


def draw_landmarks_side_by_side_test(gt_pts, pred_pts, left_image, right_image):
    gt_pts = np.asarray(gt_pts, np.float32)
    pred_pts = np.asarray(pred_pts, np.float32)

    left = left_image.copy()
    right = right_image.copy()

    for i, pt in enumerate(gt_pts):
        cv2.circle(left, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1, 1)

    for i, pt in enumerate(pred_pts):
        color_255 = COLORS[i % len(COLORS)]
        for corner_x, corner_y in ((pt[2], pt[3]), (pt[4], pt[5]), (pt[6], pt[7]), (pt[8], pt[9])):
            cv2.circle(right, (int(corner_x), int(corner_y)), 2, color_255, -1, 1)

    gap = np.full((left.shape[0], 24, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([left, gap, right], axis=1)
    return canvas



def draw_landmarks_pre_proc(out_image, pts):
    num_vertebra = min(pts.shape[0] // 4, len(COLORS))
    for i in range(num_vertebra):
        pts_4 = pts[4 * i:4 * i + 4, :]
        color_255 = COLORS[i % len(COLORS)]
        cv2.circle(out_image, (int(pts_4[0, 0]), int(pts_4[0, 1])), 3, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[1, 0]), int(pts_4[1, 1])), 3, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[2, 0]), int(pts_4[2, 1])), 3, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[3, 0]), int(pts_4[3, 1])), 3, color_255, -1, 1)
    return np.uint8(out_image)


def draw_regress_pre_proc(out_image, pts):
    num_vertebra = min(pts.shape[0] // 4, len(colors))
    for i in range(num_vertebra):
        pts_4 = pts[4 * i:4 * i + 4, :]
        pt = np.mean(pts_4, axis=0)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[0, 0]), int(pts_4[0, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[1, 0]), int(pts_4[1, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[2, 0]), int(pts_4[2, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[3, 0]), int(pts_4[3, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.putText(out_image, '{}'.format(i + 1), (int(pts_4[1, 0] + 10), int(pts_4[1, 1] + 10)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color_255, 1, 1)
    return np.uint8(out_image)