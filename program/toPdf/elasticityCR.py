## PDE1
## 模型与剖分

import math
import numpy as np

from tool import PDE, getIsBdNode, uniform_refine, get_cr_node_cell
from tool import get_A1_A2_F, my_solve, H1Error, print_error, print_P
from tool import get_stiff_and_div_matrix, get_bb, drawer_uh_u

n   = 4  #剖分次数
n   = n + 1 
Lam = [1,1e1,1e2,1e3,1e4,1e5]
Mu  = Lam

H     = np.zeros(n)                              #步长
P     = np.zeros(len(Lam))                       #误差阶
E     = np.zeros((len(Lam),n), dtype=np.float64) #每个lambda(行)对应的误差(列)

for i in range(len(Lam)):
    pde = PDE(Mu[i], Lam[i])# / 2 + 10, Lam[i])
    node = np.array([
            (0,0),
            (1,0),
            (1,1),
            (0,1)], dtype=np.float64)
    cell = np.array([(1,2,0), (3,0,2)], dtype=np.int64)
    for j in range(n):
        NC = cell.shape[0]
        nn  = math.log(NC/2, 4)
        NN = int(3 * 2 * 4**nn - (3 * 2 * 4**nn - 4 * 2**nn) / 2)
        cm = np.ones(NC, dtype=np.float64) / NC
     
        cr_node, cr_cell = get_cr_node_cell(node,cell)

        # 单元刚度矩阵和单元载荷向量 
        A1, A2 = get_stiff_and_div_matrix(cr_node, cr_cell, cm)
        bb = get_bb(pde, node, cell, cm)
        
        A1, A2, F = get_A1_A2_F(A1, A2, bb, cr_node, cr_cell)
        A = pde.mu * A1 + (pde.lam + pde.mu) * A2
              
        uh = my_solve(A, F, cr_node, getIsBdNode)
        u = pde.solution(cr_node)
        H[j] = np.sqrt(2 * cm[0])
        # 计算误差
        E[i][j] = H1Error(u, uh)
        if j < n-1 :
            node, cell = uniform_refine(node, cell)
    uh_dir = "../../image/tmp/elaticity_uh_u/PDE1/uh_lam={}.png".format(Lam[i])
    u_dir = "../../image/tmp/elaticity_uh_u/PDE1/u_lam={}.png".format(Lam[i])
    drawer_uh_u(cr_node, uh, u, uh_dir, u_dir)

if n-1 > 1: 
    for i in range(len(Lam)):
        f = np.polyfit(np.log(H[1:]), np.log(E[i][1:]) ,1)
        P[i] = f[0]

print_error(Lam, H, E)
if n-1 > 1:
    print_P(Lam, P)