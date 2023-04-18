import math
import numpy as np
from numpy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class PDE():
    def __init__(self, mu=1, lam=1):
        self.mu  = mu
        self.lam = lam
    
    def source(self, p):
        x   = p[..., 0]
        y   = p[..., 1]
        mu  = self.mu
        lam = self.lam
        
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float64)
        """
        val[..., 0] = -((2 * mu + lam) * y * (y - 1) * (2 * cos(x) - (x - 1) * sin(x))
                        + (mu + lam) * (2 * x - 1) * (sin(y) + (y - 1) * cos(y)) 
                        + 2 * mu * (x -1) * sin(x))
        val[..., 1] = -((2 * mu + lam) * x * (x - 1) * (2 * cos(y) - (y - 1) * sin(y))
                        + (mu + lam) * (2 * y - 1) * (sin(x) + (x - 1) * cos(x))
                        + 2 * mu * (y - 1) * sin(y))
        """
        frac_u1_x   = y * (y-1) * (2*cos(x) - (x-1) * sin(x))
        frac_u1_y   = 2 * (x-1) * sin(x)
        frac_u1_x_y = (2*y-1) * (sin(x) + (x-1) * cos(x))
        frac_u2_x   = 2 * (y-1) * sin(y)
        frac_u2_y   = x * (x-1) * (2*cos(y) - (y-1) * sin(y))
        frac_u2_x_y = (2*x-1) * (sin(y) + (y-1) * cos(y))

        val[..., 0] = -((2*mu+lam) * frac_u1_x + (mu+lam) * frac_u2_x_y + mu*frac_u1_y)
        val[..., 1] = -((2*mu+lam) * frac_u2_y + (mu+lam) * frac_u1_x_y + mu*frac_u2_x)

        return val
    
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        
        val = np.zeros(p.shape, dtype=np.float64)
        
        val[..., 0] = y * (x - 1) * (y - 1) * np.sin(x)
        val[..., 1] = x * (x - 1) * (y - 1) * np.sin(y)

        return val
    
class interfaceData():
    def __init__(self, mu=np.array([1,2,3,4]), lam=np.array([1,2,3,4])):
        self.mu = mu
        self.lam = lam
        self.node = np.array([(0,0),
                        (0.5,0),
                        (1,0),
                        (0,0.5),
                        (0.5,0.5),
                        (1,0.5),
                        (0,1),
                        (0.5,1),
                        (1,1)], dtype=np.float64)
        self.cell = np.array([[0,1,4],
                [0,4,3],
                [1,2,5],
                [1,5,4],
                [3,4,7],
                [3,7,6],
                [4,5,8],
                [4,8,7]], dtype=np.int64)
    
    def source(self, p):
        x   = p[..., 0]
        y   = p[..., 1]
        
        mu  = 1
        lam = 1
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float64)

        frac_u1_x   = y * (y-1) * (2*cos(x) - (x-1) * sin(x))
        frac_u1_y   = 2 * (x-1) * sin(x)
        frac_u1_x_y = (2*y-1) * (sin(x) + (x-1) * cos(x))
        frac_u2_x   = 2 * (y-1) * sin(y)
        frac_u2_y   = x * (x-1) * (2*cos(y) - (y-1) * sin(y))
        frac_u2_x_y = (2*x-1) * (sin(y) + (y-1) * cos(y))

        val[..., 0] = -((2*mu+lam) * frac_u1_x + (mu+lam) * frac_u2_x_y + mu*frac_u1_y)
        val[..., 1] = -((2*mu+lam) * frac_u2_y + (mu+lam) * frac_u1_x_y + mu*frac_u2_x)

        return val
    
    def solution(self, p, cr_node):
        x = p[..., 0]
        y = p[..., 1]
        
        val = np.zeros(p.shape, dtype=np.float64)
        
        val[..., 0] = y * (x - 1) * (y - 1) * np.sin(x)
        val[..., 1] = x * (x - 1) * (y - 1) * np.sin(y)

        crCell = getWhichCell(cr_node)
        #print("val= ", val)
        for i in range(4):
            val[crCell[i], :] /= self.lam[i]
        #print("val= ", val)

        return val

def error(u, uh):
    e = u - uh
    emax = np.max(np.abs(e))
    return emax

def print_error(Lam, H, E):
    for i in range(len(Lam)):
        print("---------------------Lam= {}---------------------".format(Lam[i]))
        #print("lam= ", Lam[i])
        n = H.shape[0]
        print()
        for j in range(n):
            print("h= ", H[j])
            print("e=", E[i][j])
            #print("e_rel= ", E_rel[i,j])
            print()
        print()
    
def print_P(Lam, P):
    print("---------------------误差阶---------------------")
    for i in range(len(Lam)):
        print("lam= ", Lam[i])
        print("p= ", P[i])
        print()
              
#判断 P (维度[2]) 是否在 cr_node, 是则放回其下标，否则 (val = 1 时) 将 P 加入 cr_node 并返回下标
def is_in_cr_node(p, cr_node):
    #p 不会为[0,0]
    #print("p= ", p)
    #print("cr_node[0]= ", cr_node[0])
    index = np.where((cr_node == p).all(axis=1))[0]
    if len(index):
        return index[0]
    else:
        in_index = np.where((cr_node == np.array([0,0])).all(axis=1))[0]
        if len(in_index) == 0:
            print("cr_node= ", cr_node)
            raise Exception("数组cr_node已满")
        cr_node[in_index[0]] = p
        return in_index[0]
    
#判断 P (维度[2]) 是否在 node, 是则放回其下标，否则将 P 加入 node 并返回下标
def is_in_node(p, node):
    #p 不会为[0,0]
    #print("p= ", p)
    #print("cr_node[0]= ", cr_node[0])
    index = np.where((node == p).all(axis=1))[0]
    if len(index):
        return index[0]
    else:
        in_index = np.where((node == np.array([0,0])).all(axis=1))[0]
        if len(in_index) == 1:
            print("node= ", node)
            raise Exception("数组node已满")
        node[in_index[1]] = p
        return in_index[1]
    
# a_cell 是否属于 cell，是则返回下标， 否则返回 -1
def is_in_cell(a_cell, cell):
    i = np.where((cell == a_cell).all(axis=1))[0]
    if len(i):
        return i[0]
    else: 
        return -1 

#将 a_cell (维数[3]) 放入new_cr_cell
def push_cr_cell(a_cell, new_cr_cell):
    in_index = np.where((new_cr_cell == np.array([0,0,0])).all(axis=1))[0]
    #print("in_index= ", in_index)
    if len(in_index) == 0:
        #print("in_index= ", in_index)
        raise Exception("数组cr_cell已满")
    new_cr_cell[in_index[0]] = a_cell
    return new_cr_cell
        

# 对单个三角形 a_cell_node (维度 [3, 2]) 求三条边中点 p1, p2, p3 并将其放入 new_cr_node 、 new_cr_cell
def a_creat(a_cell_node, new_cr_node, new_cr_cell):
    #print("a_cell_node= ", a_cell_node)
    p1 = (a_cell_node[0] + a_cell_node[1]) / 2
    p2 = (a_cell_node[0] + a_cell_node[2]) / 2
    p3 = (a_cell_node[1] + a_cell_node[2]) / 2
    
    p1_i = is_in_cr_node(p1, new_cr_node)
    p2_i = is_in_cr_node(p2, new_cr_node)
    p3_i = is_in_cr_node(p3, new_cr_node)
    
    #if is_in_cell([p1_i, p2_i, p3_i], new_cr_cell) == -1:
    push_cr_cell([p1_i, p2_i, p3_i], new_cr_cell)
    
    return new_cr_node, new_cr_cell
    
def refine_a_cell(a_cell, new_node, new_cell):
    p1 = (new_node[a_cell][0] + new_node[a_cell][1]) / 2
    p2 = (new_node[a_cell][0] + new_node[a_cell][2]) / 2
    p3 = (new_node[a_cell][1] + new_node[a_cell][2]) / 2

    p1_i = is_in_node(p1, new_node)
    p2_i = is_in_node(p2, new_node)
    p3_i = is_in_node(p3, new_node)

    push_cr_cell([p1_i, p2_i, p3_i], new_cell)
    push_cr_cell([a_cell[0], p1_i, p2_i], new_cell)
    push_cr_cell([p1_i, a_cell[1], p3_i], new_cell)
    push_cr_cell([p2_i, p3_i, a_cell[2]], new_cell)

    return new_node, new_cell

"""
# 由端点的 node 、cell 得到边中点的 cr_node cr_cell
def get_cr_node_cell(n, cr_node, cr_cell, node, cell):
    old_NN = cr_node.shape[0]
    old_NC = cr_cell.shape[0]
    
    mesh_text = TriangleMesh(node, cell)
    mesh_text.uniform_refine(n)
    #[NN,2] 剖分点及其编号(下标)
    node = mesh_text.entity('node')
    #print("node= ", node)
    #[NC,3] 剖分区间及其端点编号
    cell = mesh_text.entity('cell')
    NN = mesh_text.number_of_faces()
    NC = mesh_text.number_of_cells()
    #print("old_NN= ", old_NN)
    #print("NN= ", NN)
    #print("NC= ", NC)
    #print("cell.shape= ", cell.shape)
    
    new_cr_node = np.zeros((NN, 2), dtype=np.float64)
    new_cr_cell = np.zeros((NC, 3), dtype=np.int64)
    
    if n == 0:
        new_cr_node[0:old_NN, :] = cr_node
        new_cr_cell[0:old_NC, :] = cr_cell
        return new_cr_node, new_cr_cell
    for i in range(NC):
        #print(i)
        #print("cell= ", cell)
        new_cr_node, new_cr_cell = a_creat(node[cell[i]], new_cr_node, new_cr_cell)
    
    return new_cr_node, new_cr_cell
"""

# 从剖分node, cell得到 cr_node, cr_cell
# 单元数 NC
# 剖分次数 n : log_4(NC / 2) 
# 外边 out_edge : 4 * 2**n
# 总边 all_edge ： 3 * NC - (3 * NC - out_edge) / 2
def get_cr_node_cell(node, cell):
    NC = cell.shape[0]
    #print("cr_NC= ", NC)
    # n 特定情况下剖分次数
    n  = math.log(NC/2, 4)
    NN = int(3 * 2 * 4**n - (3 * 2 * 4**n - 4 * 2**n) / 2)
    #print("cr_NN= ", NN)
    cr_node = np.zeros((NN, 2), dtype=np.float64)
    cr_cell = np.zeros_like(cell)

    for i in range(NC):
        cr_node, cr_cell = a_creat(node[cell[i]], cr_node, cr_cell)

    return cr_node, cr_cell
    
# 返回 node 中是否为边界点的信息
def getIsBdNode(cr_node):
    is_BdNode = np.zeros(cr_node.shape[0], dtype=bool)
    for i in range(cr_node.shape[0]):
        a = np.min(np.abs(cr_node[i] - np.array([0,0])))
        b = np.min(np.abs(cr_node[i] - np.array([1,1])))
        if a < 1e-13 or b < 1e-13:
            is_BdNode[i] = True
    return is_BdNode

def uniform_refine(node, cell):
    old_NN = node.shape[0]
    old_NC = cell.shape[0]
    n  = math.log(old_NC/2, 4)
    NC = 4 * old_NC
    num_edge = int(3 * 2 * 4**n - (3 * 2 * 4**n - 4 * 2**n) / 2)
    #print("num_edge= ", num_edge)
    NN = old_NN + num_edge
    
    new_node = np.zeros((NN, 2), dtype=np.float64)
    new_cell = np.zeros((NC, 3), dtype=np.int64)
    new_node[:old_NN] = node
    
    for i in range(old_NC):
        new_node, new_cell = refine_a_cell(cell[i], new_node, new_cell)

    return new_node, new_cell

def get_cr_glam_and_pre(cr_node, cr_cell):
    NC = cr_cell.shape[0]
    NN = cr_node.shape[0]
    cr_node_cell = cr_node[cr_cell]
    ##求解CR元导数
    cr_node_cell_A = np.ones((NC, 3, 3), dtype=np.float64)
    #求解CR元导数的系数矩阵
    cr_node_cell_A[:, :, 0:2] = cr_node_cell
    #print("cr_node_cell_A= ", cr_node_cell_A[0])
    #用于求解CR元的值
    # cr_glam_x_y_pre [NC, 3, 3]
    cr_glam_x_y_pre = np.zeros((NC, 3, 3), dtype=np.float64)
    for k in range(NC):
        cr_glam_x_y_pre[k, :, :] = solve(cr_node_cell_A[k, :, :], np.diag(np.ones(3)))
    #print("cr_glam_x_y_pre= ", cr_glam_x_y_pre[0])
    #[NC,3,3]
    cr_glam_x_y = np.copy(cr_glam_x_y_pre)
    cr_glam_x_y = cr_glam_x_y[:, 0:2, :]
    #print("cr_glam_x_y= ", cr_glam_x_y)
    cr_glam_x_y = cr_glam_x_y.transpose((0,2,1))
    #print("cr_glam_x_y= ", cr_glam_x_y[0])
    return cr_glam_x_y, cr_glam_x_y_pre
    
# phi_val [NC,3(点),6(6个基函数),2(两个分量)]
def get_phi_val(node, cell, cr_glam_pre):
    NC = cell.shape[0]
    # cr_node_val [NC,3(点),3(三个cr元的值)] CR元在各顶点的值
    node_cell_A = np.ones((NC,3,3), dtype=np.float64)
    node_cell_A[:,:,0:2] = node[cell]
    cr_node_val = np.einsum("cij, cjk -> cik", node_cell_A, cr_glam_pre)
    #print("cr_node_vall= ", cr_node_val)
        
    # phi_node_val [NC,3(点),6(6个基函数),2(两个分量)]
    phi_node_val = np.zeros((NC,3,6,2), dtype=np.float64)
    phi_node_val[:,:,0:5:2,0] = cr_node_val
    phi_node_val[:,:,1:6:2,1] = cr_node_val
    #print("phi_node_val= ", phi_node_val[0,0])
    return phi_node_val

def get_phi_grad_and_div(cr_node, cr_cell):
    NC = cr_cell.shape[0]
    cr_glam_x_y, cr_glam_x_y_pre = get_cr_glam_and_pre(cr_node, cr_cell)
    #求 cr_phi_grad [NC,6(基函数),2(分量 x , y),2(导数)]
    cr_phi_grad = np.zeros((NC,6,2,2), dtype=np.float64)
    cr_phi_grad[:, 0:5:2, 0, :] = cr_glam_x_y
    cr_phi_grad[:, 1:6:2, 1, :] = cr_glam_x_y
    #print("cr_phi_grad= ", cr_phi_grad)
        
    # cr_phi_div [NC, 6]
    #cr_phi_div = np.einsum("cmij -> cm", cr_phi_grad)
    cr_phi_div = cr_glam_x_y.copy()
    cr_phi_div = cr_phi_div.reshape(NC, 6)
    #print("cr_phi_div= ", cr_phi_div)
    return cr_phi_grad, cr_phi_div

# 定义一个函数，计算向量的H_1范数
def H_1_norm(u, h):
    # 参数u是一个一维数组，表示离散向量
    # 参数h是一个浮点数，表示离散步长
    # 返回一个浮点数，表示向量的H_1范数

    # 计算向量的L_2范数，使用numpy.linalg.norm函数
    u_L2 = np.linalg.norm(u)

    # 计算向量的梯度，使用numpy.diff函数
    u_grad = np.diff(u) / h

    # 计算梯度的L_2范数，使用numpy.linalg.norm函数
    u_grad_L2 = np.linalg.norm(u_grad)

    # 计算向量的H_1范数，使用平方和开根号的公式
    u_H1 = np.sqrt(u_L2**2 + u_grad_L2**2)

    # 返回结果
    return u_H1

## 单元刚度矩阵， 单元质量矩阵 stiff, div [NC, 6, 6]
def get_stiff_and_div_matrix(cr_node, cr_cell, cm):
    cr_phi_grad, cr_phi_div = get_phi_grad_and_div(cr_node, cr_cell)
        
    ## 单元刚度矩阵
    # A1 A2 [NC, 6, 6]
    A1 = np.einsum("cnij, cmij, c -> cnm", cr_phi_grad, cr_phi_grad, cm)
    A2 = np.einsum("cn, cm, c -> cnm", cr_phi_div, cr_phi_div, cm)
    #print("A1= ", A1)
    #print("A2= ", A2)
    return A1, A2

## 单元载荷向量 bb [NC, 6]
def get_bb(pde, node, cell, cm):
    cr_node, cr_cell = get_cr_node_cell(node,cell)
    cr_glam_x_y, cr_glam_x_y_pre = get_cr_glam_and_pre(cr_node, cr_cell)
    # phi_val [NC,3(点),6(6个基函数),2(两个分量)]
    phi_node_val = get_phi_val(node, cell, cr_glam_x_y_pre)
        
    #print("node[cell]= ", node[cell].shape)
    # val [NC,3(点),2(分量)] 右端项在各顶点的值
    val = pde.source(node[cell])
    #print("val= ", val[0,0])
        
    # phi_val [NC,3,6] 基函数和右端项的点乘
    phi_val = np.einsum("cijk, cik -> cij", phi_node_val, val)
    #print("phi_val= ", phi_val.shape)
    # bb [NC,6]
    #print("type(phi_val)= ", type(phi_val))
    bb = phi_val.sum(axis=1) * cm[0] / 3
    return bb

#input 
#单元刚度矩阵 A1, A2 [NC, 6, 6], bb [NC,6]
# output
# 总刚度矩阵 A1, A2 [2*NN,2*NN], F [2*NN]
def get_A1_A2_F(A1, A2, bb, cr_node, cr_cell):
    NN = cr_node.shape[0]
    NC = cr_cell.shape[0]
    # cell_x_y [NC, 3(三个点), 2(x y 方向上基函数的编号)]
    cell_x_y = np.broadcast_to(cr_cell[:,:,None], shape=(NC, 3, 2)).copy()
    cell_x_y[:,:,0] = 2 * cell_x_y[:,:,0]       #[NC,3] 三个节点x方向上基函数在总刚度矩阵的位置
    cell_x_y[:,:,1] = 2 * cell_x_y[:,:,1] + 1   #[NC,3] 三个节点y方向上基函数在总刚度矩阵的位置
    #print("cr_cell= ", cr_cell)
    #print("cell_x_y= ", cell_x_y)
    cell_x_y = cell_x_y.reshape(NC, 6)
    #print("cell_x_y= ", cell_x_y)
    I = np.broadcast_to(cell_x_y[:, :, None], shape=A1.shape)
    J = np.broadcast_to(cell_x_y[:, None, :], shape=A2.shape)
    #print("I= ", I)
    #print("J= ", J)
        
    #S = csr_matrix((S.flat, (I.flat, J.flat)), shape=(2 * NN,2 * NN))
    #M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(2 * NN,2 * NN))
    A1 = csr_matrix((A1.flat, (I.flat, J.flat)), shape=(2 * NN,2 * NN))
    A2 = csr_matrix((A2.flat, (I.flat, J.flat)), shape=(2 * NN,2 * NN))
    F = np.zeros(2 * NN)
    np.add.at(F, cell_x_y, bb)
    return A1, A2, F

def my_solve(A, F, cr_node, getIsBdNode):
    NN = cr_node.shape[0]
    isBdNode    = getIsBdNode(cr_node)
    isInterNode = ~isBdNode
    #print("isInterNode= ", isInterNode)
    isInterNodeA = np.broadcast_to(isInterNode[:, None], shape=(NN, 2))
    isInterNodeA = isInterNodeA.reshape(2 * NN)
    #print("isInterNodeA= ", isInterNodeA)

    uh = np.zeros((2 * NN), dtype=np.float64)
    uh[isInterNodeA] = spsolve(A[:, isInterNodeA][isInterNodeA], F[isInterNodeA])
    #uh = spsolve(A, F)
    #print("uh= ", uh)
    uh = uh.reshape(NN, 2)
    return uh

## [4,NN] 返回各点属于哪个区间 
## \Omega_1 [0,0.5]   \times [0,0.5)
## \Omega_2 (0.5, 1]  \times [0,0.5]
## \Omega_3 [0,0.5)   \times [0.5,1]
## \Omega_4 [0.5,1] \times (0.5,1]
def getWhichCell(node):
    isWhichCellNode = np.zeros((4,node.shape[0]), dtype=bool)
    for i in range(node.shape[0]):
        a = node[i, 0] - 0
        b = node[i, 1] - 0
        if a <= 0.5 and b < 0.5:
            isWhichCellNode[0,i] = True
        if a > 0.5 and b <= 0.5:
            isWhichCellNode[1,i] = True
        if a < 0.5 and b >= 0.5:
            isWhichCellNode[2,i] = True
        if a >= 0.5 and b > 0.5:
            isWhichCellNode[3,i] = True
    return isWhichCellNode

## [4, NC] 返回各单元属于哪个区间
def getCellInOmega(node, cell):
    cellInOmega = np.zeros(cell.shape[0], dtype=bool)
    #print("node[cell]= ", node[cell])
    mid_p = node[cell].sum(axis=1) / 3
    #print("mid_p= ", mid_p)
    cellInOmega = getWhichCell(mid_p)
    return cellInOmega

## [4,NN] 
## \line_1 x=0.5, y<0.5
## \line_2 x=0.5, y>0.5
## \line_3 x<0.5, y=0.5
## \line_4 x>0.5, y=0.5
def getInterfaceCell(node):
    interfaceCell = np.zeros((4,node.shape[0]), dtype=bool)
    for i in range(node.shape[0]):
        a = node[i,0]
        b = node[i,1]
        if a == 0.5 and b < 0.5:
            interfaceCell[0,i] = True
        if a == 0.5 and b > 0.5:
            interfaceCell[1,i] = True
        if a < 0.5 and b == 0.5:
            interfaceCell[2,i] = True
        if a > 0.5 and b == 0.5:
            interfaceCell[3,i] = True
    return interfaceCell

def phiInWhichCell(whichCell):
    phiCell = np.broadcast_to(whichCell[:, :, None], shape=(4, whichCell.shape[1], 2))
    phiCell = phiCell.reshape(4, 2 * whichCell.shape[1])
    return phiCell

class MESH():
    def __init__(self, node, cell):
        """
        self.NN = node.shape[0]
        self.NC = cell.shape[0]
        self.cr_NC = self.NC
        # n 特定情况下剖分次数
        n  = math.log(self.NC/2, 4)
        self.cr_NN = int(3 * 2 * 4**n - (3 * 2 * 4**n - 4 * 2**n) / 2)
        """
        self.node = node # [NN,2]
        self.cell = cell # [NC,3]
        """
        self.cr_node, self.cr_cell = get_cr_node_cell(node, cell)   #[cr_NN,2] #[cr_NC,3]
        self.cm = np.ones(self.NC, dtype=np.float64) / self.NC      #[cr_NC,]
        #[NC,3(CR元),2(对x，y的偏导)] #[NC,3(3个系数),3(CR元)]
        self.cr_glam, self.cr_glam_pre = get_cr_glam_and_pre(self.cr_node, self.cr_cell) 
        # phi_val [NC,3(点),6(6个基函数),2(两个分量)]
        self.phi_val = get_phi_val(self.node, self.cell, self.cr_glam_pre)
        """
        
    def my_uniform_refine(self, n):
        if n != 0:
            for j in range(n):
                NN, NC = self.get_number_of_node_and_cell()
                #print("NC= ", NC)
                nn  = math.log(NC/2, 4)
                new_NC = 4 * NC
                num_edge = int(3 * 2 * 4**nn - (3 * 2 * 4**nn - 4 * 2**nn) / 2)
                #print("num_edge= ", num_edge)
                new_NN = NN + num_edge

                new_node = np.zeros((new_NN, 2), dtype=np.float64)
                new_cell = np.zeros((new_NC, 3), dtype=np.int64)
                new_node[:NN] = self.node

                for i in range(NC):
                    new_node, new_cell = refine_a_cell(self.cell[i], new_node, new_cell)
                self.node, self.cell = new_node, new_cell
                """
                self.cr_node, self.cr_cell = get_cr_node_cell(self.node, self.cell)
                self.NN = self.node.shape[0]
                self.NC = self.cell.shape[0]
                self.cr_NC = self.NC
                # n 特定情况下剖分次数
                nn  = math.log(self.NC/2, 4)
                self.cr_NN = int(3 * 2 * 4**nn - (3 * 2 * 4**nn - 4 * 2**nn) / 2)
                self.cm = np.ones(self.NC, dtype=np.float64) / self.NC
                self.cr_glam, self.cr_glam_pre = get_cr_glam_and_pre(self.cr_node, self.cr_cell) 
                self.phi_val = get_phi_val(self.node, self.cell, self.cr_glam_pre)
                """
                
    def get_number_of_node_and_cell(self):
        NN = self.node.shape[0]
        NC = self.cell.shape[0]
        return NN,NC
    
    def get_number_of_cr_node_and_cell(self):
        NC = self.cell.shape[0]
        cr_NC = NC
        # n 特定情况下剖分次数
        n  = math.log(NC/2, 4)
        cr_NN = int(3 * 2 * 4**n - (3 * 2 * 4**n - 4 * 2**n) / 2)
        return cr_NN, cr_NC
    
    def get_cr_node_cell(self):
        return get_cr_node_cell(self.node, self.cell)   #[cr_NN,2] #[cr_NC,3]
    
    def get_cm(self):
        NC = self.cell.shape[0]
        return np.ones(NC, dtype=np.float64) / NC
    
    def get_cr_glam_and_pre(self):
        cr_node, cr_cell = self.get_cr_node_cell()
        return get_cr_glam_and_pre(cr_node, cr_cell) 
    
    def get_phi_val(self):
        cr_glam, cr_glam_pre = self.get_cr_glam_and_pre()
        return get_phi_val(self.node, self.cell, cr_glam_pre)
    
    def get_phi_grad_and_div(self):
        cr_node, cr_cell = self.get_cr_node_cell()
        return get_phi_grad_and_div(cr_node, cr_cell)
    
    def get_stiff_and_div_matrix(self):
        cr_node, cr_cell = self.get_cr_node_cell()
        cm = self.get_cm()
        return get_stiff_and_div_matrix(cr_node, cr_cell, cm)
    
    def get_bb(self, pde):
        return get_bb(pde, self.node, self.cell, self.get_cm())

    def get_A1_A2_F(self, bb):
        A1, A2 = self.get_stiff_and_div_matrix()
        cr_node, cr_cell = self.get_cr_node_cell()
        return get_A1_A2_F(A1, A2, bb, cr_node, cr_cell)
    
    def getWhichCell(self):
        cr_node, cr_cell = self.get_cr_node_cell()
        return getWhichCell(cr_node)
    
    def phiInWhichCell(self):
        whichCell = self.getWhichCell()
        return phiInWhichCell(whichCell)
    
    def getCellInOmega(self):
        cr_node, cr_cell = self.get_cr_node_cell()
        return getCellInOmega(cr_node, cr_cell)


if __name__ == "__main__":
    """
    node = np.array([
            (0,0),
            (1,0),
            (1,1),
            (0,1)], dtype=np.float64)
    cell = np.array([(1,2,0), (3,0,2)], dtype=np.int64)

    # 剖分次数
    n = 0
    mesh = MESH(node, cell)
    mesh.my_uniform_refine(n)
    cr_node, cr_cell = mesh.get_cr_node_cell()
    #print("cr_node[cr_cell]= ", cr_node[cr_cell][0])
    cr_glam, cr_glam_pre = mesh.get_cr_glam_and_pre()
    #print("cr_glam= ", cr_glam)
    #print("cr_glam_pre= ", cr_glam_pre)
    
    a = np.ones((3,3), dtype=np.float64)
    a[:, :2] = cr_node[cr_cell][0]
    print("cr_glam_pre[0] @ a= ", cr_glam_pre[0] @ a)
    #print("phi_val= ", mesh.get_phi_val().shape)


    phi_grad, phi_div = mesh.get_phi_grad_and_div()
    #print("phi_grad= ", phi_grad)
    print("phi_div= ", phi_div)

    cm = mesh.get_cm()
    print("cm= ", cm[0])
    A1, A2 = mesh.get_stiff_and_div_matrix()
    print("A2= ", A2)
    """

    interfacePde = interfaceData()
    node, cell = interfacePde.node, interfacePde.cell

    n = 0
    mesh = MESH(node, cell)
    mesh.my_uniform_refine(n)

    node, cell = mesh.node, mesh.cell
    #print("cell= ", cell)

    cr_node, cr_cell = mesh.get_cr_node_cell()
    #print("cr_node= ", cr_node)
    #print("cr_cell= ", cr_cell)

    cellInOmega = mesh.getCellInOmega()
    print("cellInOmega", cellInOmega[0])

    A1, A2 = mesh.get_stiff_and_div_matrix()
    print("A1= ", A1[0:2])

    A1[cellInOmega[0]] *= 2
    print("A1= ", A1[0:2])