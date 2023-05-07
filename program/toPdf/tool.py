import math
import numpy as np
from numpy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# PDE1
# u1 = y(x-1)(y-1)sin(x)
# u2 = x(x-1)(y-1)sin(y)
# uh_dir = "../../image/tmp/elaticity_uh_u/PDE1/uh_lam={}.png".format(Lam[i])
# u_dir = "../../image/tmp/elaticity_uh_u/PDE1/u_lam={}.png".format(Lam[i])
class PDE():
    def __init__(self, mu=1, lam=1):
        self.mu  = mu
        self.lam = lam
        self.node = np.array([
            (0,0),
            (1,0),
            (1,1),
            (0,1)], dtype=np.float64)
        self.cell = np.array([(1,2,0), (3,0,2)], dtype=np.int64)
    
    def source(self, p):
        x   = p[..., 0]
        y   = p[..., 1]
        mu  = self.mu
        lam = self.lam
        
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
    
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        
        val = np.zeros(p.shape, dtype=np.float64)
        
        val[..., 0] = y * (x - 1) * (y - 1) * np.sin(x)
        val[..., 1] = x * (x - 1) * (y - 1) * np.sin(y)

        return val
    
# PDE2
# u1 = u2 = x^2 * sin(x-1) * y^2 * sin(y-1) 
# uh_dir = "../../image/tmp/elaticity_uh_u/PDE2/uh_lam={}.png".format(Lam[i])
# u_dir = "../../image/tmp/elaticity_uh_u/PDE2/u_lam={}.png".format(Lam[i])
class PDE2():
    def __init__(self, mu=1, lam=1):
        self.mu  = mu
        self.lam = lam
        self.node = np.array([
            (0,0),
            (1,0),
            (1,1),
            (0,1)], dtype=np.float64)
        self.cell = np.array([(1,2,0), (3,0,2)], dtype=np.int64)
    
    def source(self, p):
        x   = p[..., 0]
        y   = p[..., 1]
        mu  = self.mu
        lam = self.lam
        
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float64)

        ux         = x**2 * sin(x-1)
        uy         = y**2 * sin(y-1)
        frac_ux_x  = 2 * x * sin(x-1) + x**2 * cos(x-1)
        frac_ux_xx = 2 * sin(x-1) + 2 * x *cos(x-1) + 2 * x * cos(x-1) - x**2 * sin(x-1)
        frac_uy_y  = 2 * y * sin(y-1) + y**2 * cos(y-1)
        frac_uy_yy = 2 * sin(y-1) + 2 * y *cos(y-1) + 2 * y * cos(y-1) - y**2 * sin(y-1)

        frac_u1_x   = frac_ux_xx * uy
        frac_u1_y   = ux * frac_uy_yy
        frac_u1_x_y = frac_ux_x * frac_uy_y
        frac_u2_x   = frac_ux_xx * uy
        frac_u2_y   = ux * frac_uy_yy
        frac_u2_x_y = frac_ux_x * frac_uy_y

        val[..., 0] = -((2*mu+lam) * frac_u1_x + (mu+lam) * frac_u2_x_y + mu*frac_u1_y)
        val[..., 1] = -((2*mu+lam) * frac_u2_y + (mu+lam) * frac_u1_x_y + mu*frac_u2_x)

        return val
    
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float64)
        
        ux = x**2 * sin(x-1)
        uy = y**2 * sin(y-1)

        val[..., 0] = ux * uy
        val[..., 1] = ux * uy

        return val

# PDE3
# u1 = u2 = -(x-1) * (e^x-1) * (y-1) * (e^y-1)
# uh_dir = "../../image/tmp/elaticity_uh_u/PDE3/uh_lam={}.png".format(Lam[i])
# u_dir = "../../image/tmp/elaticity_uh_u/PDE3/u_lam={}.png".format(Lam[i])
class PDE3():
    def __init__(self, mu=1, lam=1):
        self.mu  = mu
        self.lam = lam
        self.node = np.array([
            (0,0),
            (1,0),
            (1,1),
            (0,1)], dtype=np.float64)
        self.cell = np.array([(1,2,0), (3,0,2)], dtype=np.int64)
    
    def source(self, p):
        x   = p[..., 0]
        y   = p[..., 1]
        mu  = self.mu
        lam = self.lam
        
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)

        ux = (x-1) * (exp(x) - 1)
        uy = (y-1) * (exp(y) - 1)
        frac_ux_x = x * exp(x) - 1
        frac_ux_xx = exp(x) * (x+1)
        frac_uy_y = y * exp(y) - 1
        frac_uy_yy = exp(y) * (y+1)

        frac_u1_x   = frac_ux_xx * uy
        frac_u1_y   = ux * frac_uy_yy
        frac_u1_x_y = frac_ux_x * frac_uy_y
        frac_u2_x   = frac_ux_xx * uy
        frac_u2_y   = ux * frac_uy_yy
        frac_u2_x_y = frac_ux_x * frac_uy_y

        val[..., 0] = -((2*mu+lam) * frac_u1_x + (mu+lam) * frac_u2_x_y + mu*frac_u1_y)
        val[..., 1] = -((2*mu+lam) * frac_u2_y + (mu+lam) * frac_u1_x_y + mu*frac_u2_x)

        return -val
    
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        
        val = np.zeros(p.shape, dtype=np.float64)
        
        ux = (x-1) * (np.exp(x) - 1)
        uy = (y-1) * (np.exp(y) - 1)

        val[..., 0] = ux * uy
        val[..., 1] = ux * uy

        return -val

# interfaceData1
# u1 = u2 = x(x-0.5)(x-1) * y(y-0.5)(y-1)
# uh_dir = "../../image/tmp/interface_uh_u/PDE1/uh_lam={}.png".format(Lam[i])
# u_dir = "../../image/tmp/interface_uh_u/PDE1/u_lam={}.png".format(Lam[i])
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
        val = np.zeros(p.shape, dtype=np.float64)

        frac_u1_x   = (6*x-3) * (y**3-(3/2)*y**2+(1/2)*y)
        frac_u1_y   = (x**3-(3/2)*x**2+(1/2)*x) * (6*y-3)
        frac_u1_x_y = (3*x**2-3*x+1/2) * (3*y**2-3*y+1/2)
        frac_u2_x   = frac_u1_x
        frac_u2_y   = frac_u1_y
        frac_u2_x_y = frac_u1_x_y

        val[..., 0] = -((2*mu+lam) * frac_u1_x + (mu+lam) * frac_u2_x_y + mu*frac_u1_y)
        val[..., 1] = -((2*mu+lam) * frac_u2_y + (mu+lam) * frac_u1_x_y + mu*frac_u2_x)

        return val
    
    def solution(self, p, cr_node):
        x = p[..., 0]
        y = p[..., 1]
        
        val = np.zeros(p.shape, dtype=np.float64)
        
        val[..., 0] = x * (x - 0.5) * (x - 1) * y * (y - 0.5) * (y - 1)
        val[..., 1] = x * (x - 0.5) * (x - 1) * y * (y - 0.5) * (y - 1)

        crCell = getWhichCell(cr_node)
        for i in range(4):
            val[crCell[i], :] /= self.lam[i]

        return val

def H1Error(u, uh):
    tmp = u - uh
    e = np.einsum("ni, ni -> n", tmp, tmp)
    sum = e.sum()
    return np.sqrt(sum)

def print_error(Lam, H, E):
    for i in range(len(Lam)):
        print("---------------------Lam= {}---------------------".format(Lam[i]))
        n = H.shape[0]
        print()
        for j in range(n):
            print("h= ", H[j])
            print("e=", E[i][j])
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
    if len(in_index) == 0:
        raise Exception("数组cr_cell已满")
    new_cr_cell[in_index[0]] = a_cell
    return new_cr_cell
        

# 对单个三角形 a_cell_node (维度 [3, 2]) 求三条边中点 p1, p2, p3 并将其放入 new_cr_node 、 new_cr_cell
def a_creat(a_cell_node, new_cr_node, new_cr_cell):
    p1 = (a_cell_node[0] + a_cell_node[1]) / 2
    p2 = (a_cell_node[0] + a_cell_node[2]) / 2
    p3 = (a_cell_node[1] + a_cell_node[2]) / 2
    
    p1_i = is_in_cr_node(p1, new_cr_node)
    p2_i = is_in_cr_node(p2, new_cr_node)
    p3_i = is_in_cr_node(p3, new_cr_node)
    
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

# 从剖分node, cell得到 cr_node, cr_cell
# 单元数 NC
# 剖分次数 n : log_4(NC / 2) 
# 外边 out_edge : 4 * 2**n
# 总边 all_edge ： 3 * NC - (3 * NC - out_edge) / 2
def get_cr_node_cell(node, cell):
    NC = cell.shape[0]
    # n 特定情况下剖分次数
    n  = math.log(NC/2, 4)
    NN = int(3 * 2 * 4**n - (3 * 2 * 4**n - 4 * 2**n) / 2)
    cr_node = np.zeros((NN, 2), dtype=np.float64)
    cr_cell = np.zeros_like(cell)

    for i in range(NC):
        cr_node, cr_cell = a_creat(node[cell[i]], cr_node, cr_cell)

    return cr_node, cr_cell
    
# 返回 node 中是否为边界点的信息
# isBdNode [NN] bool
def getIsBdNode(cr_node):
    is_BdNode = np.zeros(cr_node.shape[0], dtype=bool)
    for i in range(cr_node.shape[0]):
        a = np.min(np.abs(cr_node[i] - np.array([0,0])))
        b = np.min(np.abs(cr_node[i] - np.array([1,1])))
        if a < 1e-13 or b < 1e-13:
            is_BdNode[i] = True
    return is_BdNode

def getIsBdLineNode(cr_node):
    NN = cr_node.shape[0]
    isBdLineNode = getIsBdNode(cr_node)
    for i in range(NN):
        a = cr_node[i,0]
        b = cr_node[i,1]
        if a == 0.5 or b == 0.5:
            isBdLineNode[i] = True
    return isBdLineNode

def uniform_refine(node, cell):
    old_NN = node.shape[0]
    old_NC = cell.shape[0]
    n  = math.log(old_NC/2, 4)
    NC = 4 * old_NC
    num_edge = int(3 * 2 * 4**n - (3 * 2 * 4**n - 4 * 2**n) / 2)
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
    #用于求解CR元的值
    # cr_glam_x_y_pre [NC, 3, 3]
    cr_glam_x_y_pre = np.zeros((NC, 3, 3), dtype=np.float64)
    for k in range(NC):
        cr_glam_x_y_pre[k, :, :] = solve(cr_node_cell_A[k, :, :], np.diag(np.ones(3)))
    #[NC,3,3]
    cr_glam_x_y = np.copy(cr_glam_x_y_pre)
    cr_glam_x_y = cr_glam_x_y[:, 0:2, :]
    cr_glam_x_y = cr_glam_x_y.transpose((0,2,1))
    return cr_glam_x_y, cr_glam_x_y_pre
    
# phi_val [NC,3(点),6(6个基函数),2(两个分量)]
def get_phi_val(node, cell, cr_glam_pre):
    NC = cell.shape[0]
    # cr_node_val [NC,3(点),3(三个cr元的值)] CR元在各顶点的值
    node_cell_A = np.ones((NC,3,3), dtype=np.float64)
    node_cell_A[:,:,0:2] = node[cell]
    cr_node_val = np.einsum("cij, cjk -> cik", node_cell_A, cr_glam_pre)
        
    # phi_node_val [NC,3(点),6(6个基函数),2(两个分量)]
    phi_node_val = np.zeros((NC,3,6,2), dtype=np.float64)
    phi_node_val[:,:,0:5:2,0] = cr_node_val
    phi_node_val[:,:,1:6:2,1] = cr_node_val
    return phi_node_val

def get_phi_grad_and_div(cr_node, cr_cell):
    NC = cr_cell.shape[0]
    cr_glam_x_y, cr_glam_x_y_pre = get_cr_glam_and_pre(cr_node, cr_cell)
    #求 cr_phi_grad [NC,6(基函数),2(分量 x , y),2(导数)]
    cr_phi_grad = np.zeros((NC,6,2,2), dtype=np.float64)
    cr_phi_grad[:, 0:5:2, 0, :] = cr_glam_x_y
    cr_phi_grad[:, 1:6:2, 1, :] = cr_glam_x_y
        
    # cr_phi_div [NC, 6]
    #cr_phi_div = np.einsum("cmij -> cm", cr_phi_grad)
    cr_phi_div = cr_glam_x_y.copy()
    cr_phi_div = cr_phi_div.reshape(NC, 6)
    return cr_phi_grad, cr_phi_div

## 单元刚度矩阵， 单元质量矩阵 stiff, div [NC, 6, 6]
def get_stiff_and_div_matrix(cr_node, cr_cell, cm):
    cr_phi_grad, cr_phi_div = get_phi_grad_and_div(cr_node, cr_cell)
        
    ## 单元刚度矩阵
    # A1 A2 [NC, 6, 6]
    A1 = np.einsum("cnij, cmij, c -> cnm", cr_phi_grad, cr_phi_grad, cm)
    A2 = np.einsum("cn, cm, c -> cnm", cr_phi_div, cr_phi_div, cm)
    return A1, A2

## 单元载荷向量 bb [NC, 6]
def get_bb(pde, node, cell, cm):
    cr_node, cr_cell = get_cr_node_cell(node,cell)
    cr_glam_x_y, cr_glam_x_y_pre = get_cr_glam_and_pre(cr_node, cr_cell)
    # phi_val [NC,3(点),6(6个基函数),2(两个分量)]
    phi_node_val = get_phi_val(node, cell, cr_glam_x_y_pre)
        
    # val [NC,3(点),2(分量)] 右端项在各顶点的值
    val = pde.source(node[cell])
        
    # phi_val [NC,3,6] 基函数和右端项的点乘
    phi_val = np.einsum("cijk, cik -> cij", phi_node_val, val)
    # bb [NC,6]
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
    cell_x_y = cell_x_y.reshape(NC, 6)
    I = np.broadcast_to(cell_x_y[:, :, None], shape=A1.shape)
    J = np.broadcast_to(cell_x_y[:, None, :], shape=A2.shape)

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

def getInterLineNode(node):
    NN = node.shape[0]
    lineNode = np.zeros(NN, dtype=bool)
    for i in range(NN):
        a = node[i,0]
        b = node[i,1]
        if a == 0.5 or b == 0.5:
            lineNode[i] = True
    return lineNode

def phiInWhichCell(whichCell):
    phiCell = np.broadcast_to(whichCell[:, :, None], shape=(4, whichCell.shape[1], 2))
    phiCell = phiCell.reshape(4, 2 * whichCell.shape[1])
    return phiCell

# uh_dir = "../../image/tmp/elaticity_uh_u/uh_lam={}.png".format(Lam[i])"
# u_dir = "../../image/tmp/elaticity_uh_u/u_lam={}.png".format(Lam[i])
def drawer_uh_u(cr_node, uh, u, uh_dir, u_dir):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    x = cr_node[:,0]
    y = cr_node[:,1]

    ax.plot_trisurf(x, y, uh[:,0], cmap='rainbow')
    plt.savefig(fname=uh_dir)

    ax.plot_trisurf(x, y, u[:,0], cmap='rainbow')
    plt.savefig(fname=u_dir)
    plt.close(fig)