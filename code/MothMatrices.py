#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy as sp


# In[2]:


'''
Generates a Moth factor with provided parameters
arg b: int, base of the moth factor
arg k: int, dimension of the square matrix returned
Both arguments are powers of two
'''
def genRandMothFac(b:int, k:int, interval = [0,10]):
    return np.random.rand(b,b,k//b)

def genRandMothFacMat(n:int, b:int, k:int, interval = [0,10]):
    return np.random.rand(n//k,b,b,k//b)


# In[3]:


class MothFactorMatrix:
    def __init__(self, n, b, k, zeroes = False):
        '''
        Parameters:
        n: int, dimension of the square matrix return
        k: int, size of moth factors, power of 2 
        b: int, number of diagonals in moth factors (i.e bxb matrix of diagonal matrices), power of 2
        '''
        
        assert round(np.log2(n)) == np.log2(n), f"Dimensions should be a power of 2."
        assert round(np.log2(k)) == np.log2(k), f"Block size should be a power of 2."
        assert round(np.log2(b)) == np.log2(b), f"Block count should be a power of 2."
        
        self.n = n
        self.b = b
        self.k = k
        
        if zeroes:
            # Stores matrix in numpy array of shape (n/k, b, b, k/b)
            # All diagonals are flattened for space
            self.mat = np.zeros((n//k,b,b,k//b))
        else:
            self.mat = np.random.rand(n//k,b,b,k//b)
                        
    def get_s_matrix(self, i, j, l):
        ''' 
        Gets the s-matrix with index (i,j,l) as denoted in Definition 2.2.5 of Moth Matrix Paper
        >Insert Link<
        
        Output: 
        S: np.ndarray, bxb matrix
        '''
        
        # For slight speedup, will delete if insignificant
        b = self.b
        k = self.b
        mat = self.mat
        
        S = np.zeros((b, b))
        r = l % (k // b)
        q = int(np.floor(l // (k//b)))
        
        for m in range(0,b):
            for p in range(0,b):
                S[m,p] = mat[i, m*(b-1)+q, j*(b-1)+p,r]
        return S


# In[4]:


def moth_factor_multiply_np(A, C, n = None, b = None, k = None):
    '''
    Multiplies two Moth Factor Matrices as defined in class MothFactorMatrices
    
    Parameters:
    A: np.ndarray, Moth matrix with size n, base count b and block size k
    C: np.ndarray, Moth matrix with size n, base count b and block size k/b
    n,b,k: int, Size, base count and block size as defined above (optional)
    
    Output:
    G: np.ndarray, Moth matrix with size n, base count b^2 and block size k
    '''
            
    if b is None:
        assert round(np.log2(A.shape[1])) == np.log2(A.shape[1]), f"Base count should be a power of 2."
        assert A.shape[1] == C.shape[1], f"Both input moth matrices should have same base."
        b = A.shape[1]
    else:
        assert b == A.shape[1] and A.shape[1] == C.shape[1], f"Both input moth matrices should have base {b}."
    
    if k is None:
        k = int(A.shape[3] * b)
        assert round(np.log2(k)) == np.log2(k), f"Block size should be a power of 2."
        assert k//(b**2) == C.shape[3], f"A should have block size k and C should have block size k/b"
    else:
        assert k//(b**2) == C.shape[3], f"A should have block size k and C should have block size k/b"
    
    if n is None:
        n = int(A.shape[0] * k)
    assert round(np.log2(n)) == np.log2(n), f"{A.shape[1]} should be a power of 2."

    G = np.zeros((n//k,b**2,b**2,k//(b**2)))    #Initialize empty output matrix of proper dimensions
    
    for i in range(0, n//k):
        for j in range(0, b):
            for l in range(0, k//b):
                S = np.zeros((b, b))
                r = l % (k // (b**2))
                q = int(np.floor(l // (k//b**2)))
                
                a = np.zeros((b,1))
                c = np.zeros((b,1))
                
                for d in range(0,b):
                    a[d] = A[i,d,j,l]
                    c[d] = C[i*b+j,q,d,r]

                S = a @ c.T

                for m in range(0,b):
                    for p in range(0,b):
                        G[i,m*b+q,j*b+p,r] = S[m,p]
                        
    return G


# In[5]:


def flattened_moth_to_matrix(mat,n,b,k):
    '''
    Convers a Moth Factor Matrix in flattened form into a proper matrix.
    
    Parameters:
    mat: np.ndarray, Moth matrix with size n, base count b and block size k as ndarray with shape (n//k,b,b,k//b)
    n,b,k: int, Size, base count and block size as defined above (optional)
    
    Output:
    ret: nxn matrix    
    '''
    ret = np.zeros((n,n))
    shape = mat.shape
    for factor in range(shape[0]):
        start = factor*k    # The row/col to start inserting moth factor
        for row in range(shape[1]):
            for col in range(shape[2]):
                for diag_i in range(shape[3]):
                    ret[start + row*(k // b) + diag_i, start + col*(k // b) + diag_i] = mat[factor, row, col, diag_i]
    return ret


# In[17]:


def moth_to_matrix(mat):
    '''
    Convers a Moth Factor Matrix in flattened form into a proper matrix.
    
    Parameters:
    mat: np.ndarray, Moth matrix with size n, base count b and block size k as ndarray with shape (n//k,b,b,k//b)
    n,b,k: int, Size, base count and block size as defined above (optional)
    
    Output:
    ret: nxn matrix    
    '''
    b = mat.shape[1]
    assert round(np.log2(b)) == np.log2(b), f"Base count should be a power of 2."
    
    k = int(mat.shape[3] * b)
    assert round(np.log2(k)) == np.log2(k), f"Block size should be a power of 2."
    
    n = int(mat.shape[0] * k)
    assert round(np.log2(n)) == np.log2(n), f"{mat.shape[1]} should be a power of 2."

    ret = np.zeros((n,n))
    shape = mat.shape
    for factor in range(shape[0]):
        start = factor*k    # The row/col to start inserting moth factor
        for row in range(shape[1]):
            for col in range(shape[2]):
                for diag_i in range(shape[3]):
                    ret[start + row*(k // b) + diag_i, start + col*(k // b) + diag_i] = mat[factor, row, col, diag_i]
    return ret


# In[1]:


def moth_factorize(G, n = None, b_2=None, k=None):
    '''
    Factorizes a Moth Factor Matrix with minimum error.
    
    Parameters:
    G: np.ndarray, Moth matrix with size n, base count b^2 and block size k as ndarray with shape (n//k,b^2,b^2,k//b^2)
    n,b^2,k: int, Size, base count (that is a square) and block size as defined above (optional)
    
    Output:
    A: np.ndarray, Moth matrix with size n, base count b and block size k as ndarray with shape (n//k,b,b,k//b)
    C: np.ndarray, Moth matrix with size n, base count b and block size k/b as ndarray with shape (n//k,b,b,k//b^2)
    '''
    
    if b_2 is None:
        b_2 = G.shape[1]
    assert round(np.log2(b_2)) == np.log2(b_2), f"Base count should be a power of 2."
    assert round(np.sqrt(b_2)) == np.sqrt(b_2), f"b should be a square"    
    
    if k is None:
        k = int(A.shape[3] * b_2)
    assert round(np.log2(k)) == np.log2(k), f"Block size should be a power of 2."
    
    if n is None:
        n = int(A.shape[0] * k)
    assert round(np.log2(n)) == np.log2(n), f"{A.shape[1]} should be a power of 2."
    b = int(np.sqrt(b_2))
    print(n,k,b)
    A = np.zeros((n//k,b,b,k//b))
    C = np.zeros((n//(k//b),b,b,(k//(b_2))))
    
    Z = np.zeros((b, b))    # Zero matrix for comparisons
    
    for i in range(0,n//k):
        for j in range(0,b):
            for l in range(0,k//b):
                r = l % (k//(b_2))
                q = int(np.floor(l//(k//b_2)))

                S = np.zeros((b, b))
                a = np.zeros((b,1))
                c = np.zeros((b,1))

                for m in range(0,b):
                    for p in range(0,b):
                        S[m,p] = G[i, m*b+q, j*b+p,r]
                
                if (S!=Z).any():
                    u,e,v = np.linalg.svd(S)
                    a = u[:,0]*e[0]
                    c = v[:,0]

                for d in range(b):
                    A[i,d,j,l] = a[d]
                    C[(i-1)*b+j,q,d,r] = c[d]
    
    return A,C              


# In[4]:


class MothMatrixNode:
    def __init__(self, mat):
        self.A = None
        self.C = None
        self.parent = None
        self.mat = mat
    
    def __str__(self):
        return f"This: {self.mat} \n left subtree: {str(self.A)} \n right subtree:{str(self.C)} "

def robust_tree_factorize(t,G,n,b,k,limit=None, rebuild = False):
    '''
    Creates a tree of factors for G, with minimum error, upto base t.
    
    Parameters:
    t: int, target base for matrices on the leaf level, power of 2
    G: np.ndarray, Moth matrix with size n, base count b and block size k as ndarray with shape (n//k,b,b,k//b)
    limit: int, Number of levels to stop building tree at
    rebuild: boolean, Flag for if the tree returned should be rebuilt by multiplying leaves matrices till root
    Output:
    T: (MothMatrixNode, Array(np.ndarray) Root of factorization tree with base t on leaf level, Array containin all leaves
    '''
    
    T = MothMatrixNode(G)
    to_factorize = [(T,k)]
    curr_base = b
    levels = 1
    
    while int(np.sqrt(curr_base)) == np.sqrt(curr_base) and curr_base != t and levels != limit:
        next_level = []
        for G_fac,k_fac in to_factorize:
            A, C = moth_factorize(G_fac.mat,n,curr_base,k_fac)
            G_fac.A = MothMatrixNode(A)
            G_fac.C = MothMatrixNode(C)
            G_fac.A.parent = G_fac
            G_fac.C.parent = G_fac
            next_level.append((G_fac.A,k_fac))
            next_level.append((G_fac.C,int(k_fac//np.sqrt(curr_base))))
        levels += 1
        curr_base = int(np.sqrt(curr_base))
        to_factorize = next_level
    
    if rebuild:
        curr_level = [n[0] for n in to_factorize]
        while len(curr_level) != 1:
            next_level = []
            for i in range(0,len(to_factorize),2):
                curr_level[i].parent.mat = curr_level[i].mat @ curr_level[i+1].mat
                next_level.append(curr_level[i].parent)
            curr_level = next_level

    return T, [n[0].mat for n in to_factorize]    # Return tree and leaf moth factor matrices


# In[ ]:




