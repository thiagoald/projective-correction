from time import time
import cv2
from random import random
from math import sqrt
import numpy as np
import sys
from projective import calc_homography
from tqdm import tqdm

# If a point was selected or not (bool)
pt_selected = False
# List of input points (user defined)
# all_points = [(597, 823), (614, 885), (708, 865), (699, 770)]
all_points = []
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

def dist(pt1, pt2):
    return sqrt(sum([(c1-c2)**2 for c1, c2 in zip(pt1,pt2)]))

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

def draw_points(img, points, color):
    for pt in points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
        for pt1, pt2 in list(zip(points[1:], points[:-1])) + [(points[-1], points[0])]:
            cv2.line(img,pt1,pt2,color,1)

def mouse_callback(event, x, y, flags, param):
    global selected_pt_idx
    global pt_selected
    global all_points
    global all_points_cor
    global all_points_ims
    global H
    global H_

    mouse_pos = (x, y)

    if param == 'Screen':

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

        if event == cv2.EVENT_MBUTTONDOWN and len(all_points) < 4:
            all_points.append(mouse_pos)

    elif param == 'ScreenFixed':
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

    if len(all_points) == 4 and len(all_points_cor) == 4 and event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_MBUTTONUP]:
        H = calc_homography(all_points, all_points_cor)
        H_ = calc_homography(all_points_cor, all_points)
        all_points_ims = []
        for p1, p2 in zip(all_points, all_points_cor):
            p1 = np.array((p1[0], p1[1], 1))
            p2_ = np.matmul(H, p1)
            p2_ = (p2_[:-1]/p2_[-1])
            all_points_ims.append(p2_)


if __name__ == '__main__':
    base_image = cv2.imread(sys.argv[1])
    rectified_image = np.zeros(base_image.shape, dtype='uint8')
    was_rectified = False

    cv2.namedWindow('Screen', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_FREERATIO)
    cv2.namedWindow('ScreenFixed', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback('Screen', mouse_callback, 'Screen')
    cv2.setMouseCallback('ScreenFixed', mouse_callback, 'ScreenFixed')

    while True:
        start = time()
        image = base_image.copy()
        image_fixed = base_image.copy()
        if not was_rectified:
            image_fixed.fill(0)
        else:
            image_fixed = rectified_image.copy()

        if H is not None:
            key = cv2.waitKey(50)
            if key == 13:
                rectified_image.fill(0)
                coords = []
                for y in range(image.shape[0]):
                    for x in range(image.shape[1]):
                        coords.append((x,y,1))
                print('Multiplying...')
                new_pts = np.matmul(H_.astype(np.float32), np.array(coords, dtype='int16').T)
                print(new_pts.shape)
                for (x_, y_, w), (x,y,_) in tqdm(list(zip(zip(*new_pts.tolist()), coords))):
                    x_ = int(x_/w)
                    y_ = int(y_ / w)
                    # print((i_,j_))
                    if(y_ >= 0 and y_ < image.shape[0] and x_ >= 0 and x_ < image.shape[1]):
                        rectified_image = cv2.circle(rectified_image, (x, y), radius=0, color=[int(c) for c in image[y_, x_]], thickness=-1)
                was_rectified = True
                print('Finished!')

        draw_points(image, all_points, (255, 255, 255))
        draw_points(image_fixed, all_points_cor, (255, 255, 255))
        pts = [tuple([int(c) for c in p.tolist()]) for p in all_points_ims]
        draw_points(image_fixed, pts , (0, 0, 255))
        cv2.imshow('Screen', image)
        cv2.imshow('ScreenFixed', image_fixed)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
