import numpy as np
from scipy.sparse.linalg import spsolve,cg
from scipy.sparse import coo_matrix
from sympy import *


def source_term(nx,ny,Lx,Ly):
    dx,dy = Lx/(nx-1),Ly/(ny-1)

    Q_in = np.zeros((nx,ny))
    u_a = np.zeros((nx,ny))

    # Use sympy to find derivatives and to populate Q_in

    x_s,y_s,pi_s,Lx_s,Ly_s = symbols('x_s y_s pi_s Lx_s Ly_s')

    func = sin(pi_s*x_s/Lx_s)*sin(2*pi_s*y_s/Ly_s)
    func_diff = -diff(func,x_s,x_s)-diff(func,y_s,y_s)
    print(func_diff)

    func = func.subs([(pi_s,np.pi),(Lx_s,Lx),(Ly_s,Ly)])
    func_diff = func_diff.subs([(pi_s,np.pi),(Lx_s,Lx),(Ly_s,Ly)])

    for j in range(ny):
        for i in range(nx):
            x = dx*i
            y = dy*j
            Q_in[i,j] = func_diff.subs([(x_s,x),(y_s,y)])
            u_a[i,j] = func.subs([(x_s,x),(y_s,y)])
            
            
    Q_in_flat = np.zeros(nx*ny)
    for j in range(ny):
        for i in range(nx):
            Q_in_flat[nx*j+i] = Q_in[i,j]

    return Q_in_flat,u_a

def create_matrix(nx,ny,Lx,Ly):
    
    dx,dy = Lx/(nx-1),Ly/(ny-1)
    
    row = []
    col = []
    data = []

    for j in range(ny):
        for i in range(nx):
            Adiag = j*nx+i
            if j==0:
                row.append(Adiag)
                col.append(Adiag)
                data.append(1)
            elif j==ny-1:
                row.append(Adiag)
                col.append(Adiag)
                data.append(1)
            elif i==0:
                row.append(Adiag)
                col.append(Adiag)
                data.append(1)    
            elif i==nx-1:
                row.append(Adiag)
                col.append(Adiag)
                data.append(1)
            else:
                a0=-2/dx**2-2/dy**2
                a1=1/dx**2
                a2=1/dx**2
                a3=1/dy**2
                a4=1/dy**2

                row.append(Adiag)
                col.append(Adiag)
                data.append(a0)

                row.append(Adiag)
                col.append(Adiag-1)
                data.append(a1)

                row.append(Adiag)
                col.append(Adiag+nx)
                data.append(a3)

                row.append(Adiag)
                col.append(Adiag+1)
                data.append(a2)

                row.append(Adiag)
                col.append(Adiag-nx)
                data.append(a4)

    As=coo_matrix((data, (row, col)), shape=(nx*ny, nx*ny))
    #As = As.tolil()
    As = As.tocsr()
    return As

def solve_pde(nx,ny,As,Q_in_flat):
    n = spsolve(As,-Q_in_flat)

    u = np.zeros((nx,ny))
    for j in range(ny):
        for i in range(nx):
            u[i,j]= n[nx*j+i]
    return u

def solve_pde_cg(nx,ny,As,Q_in_flat):
    n = cg(As,-Q_in_flat)

    u = np.zeros((nx,ny))
    for j in range(ny):
        for i in range(nx):
            u[i,j]= n[0][nx*j+i]
    return n

def L2_norm(u,u_a):
    l2_diff = np.sqrt(np.sum((u-u_a)**2)/np.sum(u_a**2))
    return l2_diff

def fit_lin(x,m,c):
    return m*x+c