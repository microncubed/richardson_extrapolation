import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from sympy import *


def manufacture_source_term(nx,ny,Lx,Ly):
    '''
    Takes a known solution to the Poisson equation sin(pi_s*x_s/Lx_s)*sin(2*pi_s*y_s/Ly_s),differentiates it twice to find the source term. Then substitutes in grid values to find the numerical values of the source term. 
    
    This is the method of manufactured solutions. 
    
    Note that the simulation domain extends between the points (0,0) and (Lx,Ly).
    
    Parameters
    ----------
    nx,ny int: the number of grid points in x,y
    Lx,Ly float: the limits in x and y of the domain
    
    Returns 
    ----------
    Q_in_flat np.ndarray: The flat (1d array) source term
    u_a np.ndarray: An array representation of the analytic solution
    
    '''
    dx,dy = Lx/(nx-1),Ly/(ny-1)
    
    #The source term, going to be found by SymPy
    Q_in = np.zeros((nx,ny))
    
    #The analytic solution
    u_a = np.zeros((nx,ny))
    x_s,y_s,pi_s,Lx_s,Ly_s = symbols('x_s y_s pi_s Lx_s Ly_s')

    #This is the chosen solution to the PDE
    func = sin(pi_s*x_s/Lx_s)*sin(2*pi_s*y_s/Ly_s)
    
    #Taking the second derivative
    func_diff = -diff(func,x_s,x_s)-diff(func,y_s,y_s)
    
    #Replacing symbolic values by actual values of constants
    func = func.subs([(pi_s,np.pi),(Lx_s,Lx),(Ly_s,Ly)])
    func_diff = func_diff.subs([(pi_s,np.pi),(Lx_s,Lx),(Ly_s,Ly)])

    #Populating the array
    for j in range(ny):
        for i in range(nx):
            x = dx*i
            y = dy*j
            Q_in[i,j] = func_diff.subs([(x_s,x),(y_s,y)])
            u_a[i,j] = func.subs([(x_s,x),(y_s,y)])
    
    #The PDE solver takes a flat input, so flattening to a 1-d array        
    Q_in_flat = np.zeros(nx*ny)
    for j in range(ny):
        for i in range(nx):
            Q_in_flat[nx*j+i] = Q_in[i,j]

    return Q_in_flat,u_a

def create_poisson_matrix(nx,ny,Lx,Ly):
    '''
    Creates the matrix used to solve the Poisson equation with Dirichlet boundary conditions. 
    
    Parameters
    ----------
    nx,ny int: the number of grid points in x,y
    Lx,Ly float: the limits in x and y of the domain
   
    Returns
    ----------
    As scipy.linalg sparse matrix in csr format: the LHS term in the matrix equation
    '''
    
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
    As = As.tocsr()
    return As

def solve_poisson(nx,ny,As,Q_in_flat):    
    '''
    Solves the Poisson equation directly for the given inputs.
    
    Parameters
    ----------
    nx,ny int: the number of grid points in x,y
    Lx,Ly float: the limits in x and y of the domain
    As scipy.linalg sparse matrix in csr format: the LHS term in the matrix equation
    Q_in_flat np.ndarray: The flat (1d array) source term
   
    Returns
    ----------
    u np.ndarray: the numerical solution
    '''
    # Perform the numerical solution
    n = spsolve(As,-Q_in_flat)

    # Turn into a 2-d array
    u = np.zeros((nx,ny))
    for j in range(ny):
        for i in range(nx):
            u[i,j]= n[nx*j+i]
    return u

def L2_norm(u,u_a):
    '''
    Finds the L2 norm between the numerical and analytic solutions
    
    Parameters
    ----------
    u np.ndarray: numerical solution
    u_a np.ndarray: analytic solution
    
    Returns
    ----------
    l2_diff float: the l2 norm
    '''
    l2_diff = np.sqrt(np.sum((u-u_a)**2)/np.sum(u_a**2))
    return l2_diff

def fit_lin(x,m,c):
    '''
    Function for the linear for
    '''
    return m*x+c