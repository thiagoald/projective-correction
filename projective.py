from sympy import Matrix
from sympy import symbols
from sympy import fraction
from sympy import Eq
from sympy import linsolve
import numpy as np

# Make an equation from a single coordinate of a symbolic and a numeric represantation of a point
def make_eq(p_im_sym, p_im):
    n1, d1 = fraction(p_im_sym)
    n2, d2 = fraction(p_im)
    return Eq(n1*d2 - n2*d1, 0)

def hom(pt):
    return pt.row_insert(2, Matrix([1]))

# Extract which number is multiplying a symbol
def coeff(symbol, expr):
    return expr.subs(symbol, 1) - expr.subs(symbol, 0)

# def calc_proj_mat(pts1, pts2):
#     a,b,c,d,e,f,g,h,i = symbols('a b c d e f g h i')
#     mat = Matrix([[a,b,c],
#                   [d,e,f],
#                   [g, h, i]])
#     pts1_sym = [mat * hom(Matrix(p)) for p in pts1]
#     # Resulting equations
#     eqs = []
#     rhs = []
#     for p_sym, p_im in zip(pts1_sym, pts2):
#         for i in range(2):
#             eqs.append(make_eq(p_sym[i], p_im[i]))
#             lhs_subs = eqs[-1].lhs
#             for s in [a, b, c, d, e, f, g, h, i]:
#                 lhs_subs = lhs_subs.subs(s, 0)
#             rhs.append(-lhs_subs)
#     print(rhs)
#     coeffs = []
#     i = symbols('i')
#     for eq in eqs:
#         coeffs.append([coeff(s, eq.lhs) for s in [a, b, c, d, e, f, g, h, i]])
#         print(eq)
#     system_mat = Matrix(coeffs[:-2])
#     print(system_mat)
#     solution = linsolve((system_mat, Matrix(rhs)), [a,b,c,d,e,f,g,h,i])
#     print(solution)

def calc_homography(pts1, pts2):
    mat = np.zeros((8, 9), dtype=np.float32)
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        xi = np.append(np.array(pt1), 1)
        (xi_, yi_, wi_) = [c for c in pt2] + [1]
        mat[2*i,:] = [0,0,0] + list(-wi_*xi) + list(yi_*xi)
        mat[2 * i + 1,:] = list(wi_ * xi) + [0, 0, 0] + list(-xi_ * xi)
    # u, s, vh = np.linalg.svd(mat)
    # print(f'U: {u}')
    # print(f'S: {s}')
    # print(f'vh: {vh}')

    # # print(vh[:, -1])
    # # print('\n\n')
    # # print(f'Lowest singular value: {s[-1]}')
    # # print(f'Vh: {vh}')
    # return vh[:, -1].reshape(3,3)
    return np.array(list(linsolve((Matrix(mat), Matrix([0]*8))))[0].subs('tau0', 1)).reshape(3,3)