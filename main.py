import argparse
import os
import sys
from math import sqrt
from random import random
from time import time

import cv2
import numpy as np
from tqdm import tqdm

from projective import calc_affine_mat, calc_homography, from_hom, hom

# If a point was selected or not (bool)
pt_selected = False
# List of input points (user defined)
# all_points = [(597, 823), (614, 885), (708, 865), (699, 770),
#               (597, 823), (614, 885), (708, 865), (699, 770)]
# # Road image
# all_points = [(29, 437), (160, 387), (768, 460), (647, 399),
#               (235, 354), (312, 323), (599, 377), (513, 336)]
# # Windows image
# all_points = [(265, 108), (269, 550), (345, 119), (352, 540),
#               (429, 118), (746, 162), (432, 275), (759, 290)]
# Speaker image
all_points = [(193, 743), (92, 555), (721, 962), (769, 788), (120, 560), (690, 757), (224, 760), (633, 925)]
# all_points = []
# all_points = [(566, 623), (568, 752), (677, 755), (677, 628)]
# Correspondences in the second image
# all_points_cor = [(597, 823), (614, 885), (708, 865), (699, 770)]
all_points_cor = []
# all_points_cor = [(197, 75), (155, 313), (326, 319), (398, 54)]
# Images of points (after multiplying by the homography matrix)
all_points_ims = []
# First screen
image = np.zeros((720, 1280, 3), np.uint8)
# Second screen
image_fixed = np.zeros((720, 1280, 3), np.uint8)
# Index of next point to be set by the user (clicking)
selected_pt_idx = None
H = None
H_ = None
DEBUG = None

def dist(pt1, pt2):
    return sqrt(sum([(c1-c2)**2 for c1, c2 in zip(pt1, pt2)]))


def closest_pt(pt, points):
    point = None
    d = float('inf')
    for p in points:
        d_ = dist(pt, p)
        print(d_)
        if d_ < d:
            d = d_
            point = p
    return point, d


def draw_points(img, pts_l, color_l, pts_r, color_r, method):
    if method == 'affine1':
        lines_idxs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        for pt in pts_l:
            cv2.circle(img, pt, 5, color_l, -1)
        for i, j in lines_idxs:
            try:
                cv2.line(img, pts_l[i], pts_l[j], color_l, 1)
            except:
                pass
        if len(pts_l) == 8:
            (l1_p1, l1_p2, l2_p1, l2_p2,
             l3_p1, l3_p2, l4_p1, l4_p2) = [hom(p) for p in pts_l]

            line1 = np.cross(l1_p1, l1_p2)
            line2 = np.cross(l2_p1, l2_p2)
            line3 = np.cross(l3_p1, l3_p2)
            line4 = np.cross(l4_p1, l4_p2)

            # Intersections (image of infinity)
            inter1 = np.cross(line1, line2)
            inter1 = tuple([int(c) for c in inter1[:-1]/inter1[-1]])
            inter2 = np.cross(line3, line4)
            inter2 = tuple([int(c) for c in inter2[:-1]/inter2[-1]])

            # import pdb
            # pdb.set_trace()

            cv2.circle(img, inter1, 5, color_l, -1)
            cv2.circle(img, inter2, 5, color_l, -1)
            cv2.line(img, inter1, inter2, (0, 0, 255), 1)


def mouse_callback(event, x, y, flags, param):
    global selected_pt_idx
    global pt_selected
    global all_points
    global all_points_cor
    global all_points_ims
    global H
    global H_

    pts_left = None

    mouse_pos = (x, y)

    if param['method'] == 'affine1':
        pts_left = 8

    if param['screen'] == 'Screen':

        if event == cv2.EVENT_MOUSEMOVE and pt_selected:
            all_points[selected_pt_idx] = mouse_pos

        if event == cv2.EVENT_LBUTTONDOWN:
            pt, d = closest_pt(mouse_pos, all_points)
            selected_pt_idx = all_points.index(pt)
            print(f'Index: {selected_pt_idx}')
            all_points[selected_pt_idx] = mouse_pos
            pt_selected = True

        if event == cv2.EVENT_LBUTTONUP:
            pt_selected = False

        if event == cv2.EVENT_MBUTTONDOWN and len(all_points) < pts_left:
            all_points.append(mouse_pos)

    elif param['screen'] == 'ScreenFixed':
        if event == cv2.EVENT_MOUSEMOVE and pt_selected:
            all_points_cor[selected_pt_idx] = mouse_pos

        if event == cv2.EVENT_LBUTTONDOWN:
            pt, d = closest_pt(mouse_pos, all_points_cor)
            selected_pt_idx = all_points_cor.index(pt)
            print(f'Index: {selected_pt_idx}')
            all_points_cor[selected_pt_idx] = mouse_pos
            pt_selected = True

        if event == cv2.EVENT_LBUTTONUP:
            pt_selected = False

        if event == cv2.EVENT_MBUTTONDOWN and len(all_points_cor) < 4:
            all_points_cor.append(mouse_pos)

    if DEBUG or event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP]:
        if param['method'] == 'affine1' and len(all_points) == 8:
            H = calc_affine_mat(all_points)
            H_ = np.linalg.inv(H)
            print(f'Matrix H: {H}')
            print(f'Matrix H\': {H_}')
            # if len(all_points) == 4 and len(all_points_cor) == 4 and event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP]:
            #     H = calc_homography(all_points, all_points_cor)
            #     H_ = calc_homography(all_points_cor, all_points)
            #     all_points_ims = []
            #     for p1, p2 in zip(all_points, all_points_cor):
            #         p1 = np.array((p1[0], p1[1], 1))
            #         p2_ = np.matmul(H, p1)
            #         p2_ = (p2_[:-1]/p2_[-1])
            #         all_points_ims.append(p2_)

def transform(image_left, image_right, mat_l_to_r, mat_r_to_l, debug=False):
    # Calculate images of bounding boxes
    y_max_l, x_max_l = image_left.shape[:-1]
    bbox_left = [(0, 0), (0, y_max_l), (x_max_l, 0), (x_max_l, y_max_l)]
    x_min_r, y_min_r = float('inf'), float('inf')
    x_max_r, y_max_r = float('-inf'), float('-inf')
    bbox_right = []
    for x, y in bbox_left:
        pt_right = np.matmul(mat_l_to_r, np.array((x, y, 1)))
        pt_right = pt_right[:-1] / pt_right[-1]
        bbox_right.append(pt_right.tolist())
        x_min_r = min(x_min_r, pt_right[0])
        x_max_r = max(x_max_r, pt_right[0])
        y_min_r = min(y_min_r, pt_right[1])
        y_max_r = max(y_max_r, pt_right[1])

    # Coordinates of bounding box scaled to shape of image_right
    pts_r_scaled = []
    for xr in np.linspace(x_min_r, x_max_r, image_right.shape[1]):
        for yr in np.linspace(y_min_r, y_max_r, image_right.shape[0]):
            pts_r_scaled.append((xr, yr, 1))
    
    # Transform to left
    print('Multiplying...')
    pts_l = np.matmul(mat_r_to_l.astype(np.float32),
                      np.array(pts_r_scaled, dtype='int16').T)

    # Bring to z=1
    for i_col in range(pts_l.shape[1]):
        pts_l[:, i_col] /= pts_l[-1, i_col]

    # Draw in right image
    for (xl, yl, _), (xr, yr, _) in tqdm(list(zip(zip(*pts_l.tolist()), pts_r_scaled))):
        xr = (xr/x_max_r)*image_right.shape[1]
        yr = (yr/y_max_r)*image_right.shape[0]
        if(xl >= 0 and xl < image_left.shape[1] and yl >= 0 and yl < image_left.shape[0]):
            try:
                image_right = cv2.circle(image_right,
                                        (int(xr), int(yr)),
                                        radius=0,
                                        color=[int(c) for c in image_left[int(yl),
                                                                        int(xl)]],
                                        thickness=-1)
            except Exception as e:
                print(e)

    # Sanity check
    for (x, y), (x_l_old, y_l_old) in zip(bbox_right, bbox_left):
        xl, yl, zl = np.matmul(mat_r_to_l.astype(np.float32),
                               np.array((x, y, 1), dtype='int32').T).tolist()
        if not(abs(xl / zl - x_l_old) < 1 or abs(yl / zl - y_l_old) < 1):
            import ipdb; ipdb.set_trace()

    # Draw bbox in right image
    # print('Drawing bbox')
    # for (x, y), (x_l_old, y_l_old) in zip(bbox_right, bbox_left):
    #     cv2.circle(image_right,
    #                (int((x / x_max_r)*image_right.shape[1]),
    #                 int((y / y_max_r)*image_right.shape[0])),
    #                radius=5,
    #                color=(255,0,255),
    #                thickness=-1)
    #     xl, yl, zl = np.matmul(mat_r_to_l.astype(np.float32),
    #                        np.array((x, y, 1), dtype='int16').T).tolist()
    #     image_right = cv2.circle(image_right,
    #                             (int((x / x_max_r)*image_right.shape[1]),
    #                              int((y / y_max_r)*image_right.shape[0])),
    #                             radius=0,
    #                             color=[int(c) for c in image_left[int(yl/zl),
    #                                                               int(xl/zl)]],
    #                             thickness=-1)

if __name__ == '__main__':
    DEBUG = bool(os.environ.get('DEBUG'))
    if DEBUG:
        print('Debug mode enabled...')
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Image to be rectified')
    parser.add_argument('--method',
                        type=str,
                        help=('affine1: pair of lines,'
                              'affine2:  ,'
                              'affine3:  ,'
                              'affine4:  ,'
                              'affine5:  '),
                        choices=['affine1',
                                 'affine2',
                                 'affine3',
                                 'affine4',
                                 'affine5',
                                 'metric'],
                        default='affine1')
    args = parser.parse_args()

    print(f'Using method: {args.method}')

    base_image = cv2.imread(args.image)
    rectified_image = np.zeros(base_image.shape, dtype='uint8')
    was_rectified = False

    cv2.namedWindow('Screen', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_FREERATIO)
    cv2.namedWindow('ScreenFixed', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_FREERATIO)
    x, y = base_image.shape[:-1]
    print(x,y)
    cv2.moveWindow('ScreenFixed', x, y)
    cv2.setMouseCallback('Screen', mouse_callback,
                         {'screen': 'Screen', 'method': args.method})
    cv2.setMouseCallback('ScreenFixed', mouse_callback,
                         {'screen': 'ScreenFixed', 'method': args.method})

    img_right = base_image.copy()
    img_right.fill(0)

    while True:
        key = cv2.waitKey(10)
        if key == 27:
            exit(0)
        img_left = base_image.copy()

        draw_points(img_left,
                    all_points, (255, 255, 255),
                    all_points_cor, (255, 255, 255),
                    method=args.method)

        if H is not None:
            if DEBUG or key == 13:
                transform(img_left, img_right, H, H_)
                was_rectified = True
                print('Finished!')
            else:
                print(f'Key: {key}')

        cv2.imshow('Screen', img_left)
        cv2.imshow('ScreenFixed', img_right)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
