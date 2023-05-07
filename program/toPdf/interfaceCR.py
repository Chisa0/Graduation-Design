## 模型与剖分

import math
import numpy as np
import matplotlib.pyplot as plt

from tool import interfaceData, getIsBdNode, uniform_refine, get_cr_node_cell
from tool import get_A1_A2_F, my_solve, H1Error, print_error, print_P
from tool import get_stiff_and_div_matrix, get_bb, drawer_uh_u
from tool import getCellInOmega

n   = 3  #剖分次数
n   = n + 1 
Lam = np.array([[1e0, 2e0, 3e0, 4e0],
                [1e1, 2e1, 3e1, 4e1],
                [1e2, 2e2, 3e2, 4e2],
                [1e3, 2e3, 3e3, 4e3],
                [1e4, 2e4, 3e4, 4e4]])

#Lam = np.array([[1e0, 2e0, 3e0, 4e0]])

Mu  = Lam

H     = np.zeros(n)                                  #步长
P     = np.zeros(Lam.shape[0])                       #误差阶
E     = np.zeros((Lam.shape[0],n), dtype=np.float64) #每个lambda(行)对应的误差(列)
 
for i in range(Lam.shape[0]):
    pde = interfaceData(Lam[i], Mu[i])
    node = pde.node
    cell = pde.cell
    for j in range(n):
        NC = cell.shape[0]
        #print("cr_NC= ", NC)
        # nn 特定情况下剖分次数
        nn  = math.log(NC/2, 4)
        NN = int(3 * 2 * 4**nn - (3 * 2 * 4**nn - 4 * 2**nn) / 2)
        cm = np.ones(NC, dtype=np.float64) / NC
     
        cr_node, cr_cell = get_cr_node_cell(node,cell)

        # 单元刚度矩阵和单元载荷向量 
        A1, A2 = get_stiff_and_div_matrix(cr_node, cr_cell, cm)
        bb = get_bb(pde, node, cell, cm)
        cellInOmega = getCellInOmega(cr_node, cr_cell)
        kk = 1
        for k in range(4):
            A1[cellInOmega[k]] *= pde.mu[k]
            A2[cellInOmega[k]] *= pde.mu[k] + pde.lam[k]
        A1, A2, F = get_A1_A2_F(A1, A2, bb, cr_node, cr_cell)
        A = A1 + A2

        uh = my_solve(A, F, cr_node, getIsBdNode)
        u = pde.solution(cr_node, cr_node)
        H[j] = np.sqrt(2 * cm[0]) / 2
        E[i][j] = H1Error(u, uh)
        if j < n-1:
            node, cell = uniform_refine(node, cell)
    drawer_uh_u(cr_node, uh, u, "../../image/tmp/interface_uh_u/uh_lam={}.png".format(Lam[i]), "../../image/tmp/interface_uh_u/u_lam={}.png".format(Lam[i]))

# 画图 得到误差阶
if n-1 > 1: 
    # 画图
    for i in range(len(Lam)):
        fig = plt.figure()
        plt.plot(np.log(H[1:]), np.log(E[i][1:]))
        plt.title("lam={}".format(Lam[i]))
        plt.xlabel("log(h)")
        plt.ylabel("log(e)")
        plt.savefig(fname="interfaceCRFem/elasticityCRFemLam_{}.png"
                    .format(Lam[i]))
        plt.close(fig)
    # 求误差阶 
    # 得到 P
    for i in range(len(Lam)):
        f = np.polyfit(np.log(H[1:]), np.log(E[i][1:]) ,1)
        P[i] = f[0]

print_error(Lam, H, E)
if n-1 > 1:
    print_P(Lam, P)