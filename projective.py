import numpy as np
from sympy import Eq, Matrix, fraction, linsolve, symbols


def calc_homography(pts1, pts2):
    '''Calculate homography matrix using 4 points in each image'''
    mat = np.zeros((8, 9), dtype=np.float32)
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        xi = np.append(np.array(pt1), 1)
        (xi_, yi_, wi_) = [c for c in pt2] + [1]
        mat[2*i, :] = [0, 0, 0] + list(-wi_*xi) + list(yi_*xi)
        mat[2 * i + 1, :] = list(wi_ * xi) + [0, 0, 0] + list(-xi_ * xi)
    A = Matrix(mat)
    B = Matrix([0]*8)
    sols = linsolve((A, B))
    first_sol = list(sols)[0]
    return np.array(first_sol.subs('tau0', 1)).reshape(3, 3)


def hom(vec):
    '''Homogeneous coordinates'''
    return np.array(list(vec) + [1])


def from_hom(vec):
    return vec[:-1]/vec[-1]


def calc_affine_mat(pts_lines):
    '''
    Calculate matrix from two pairs of lines.
    '''
    (l1_p1, l1_p2, l2_p1, l2_p2,
     l3_p1, l3_p2, l4_p1, l4_p2) = [hom(p) for p in pts_lines]

    line1 = np.cross(l1_p1, l1_p2)
    line2 = np.cross(l2_p1, l2_p2)
    line3 = np.cross(l3_p1, l3_p2)
    line4 = np.cross(l4_p1, l4_p2)

    # Intersections (image of infinity)
    inter1 = np.cross(line1, line2)
    inter2 = np.cross(line3, line4)

    # Image of infinity line
    line_inf = np.cross(inter1, inter2)
    line_inf = line_inf / (line_inf.max()*10)
    a, b, c = list(line_inf)

    # TODO: If c==0
    mat = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [a, b, c]])
    return mat
