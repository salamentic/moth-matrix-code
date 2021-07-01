#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
import torch
import numpy as np


# In[27]:


class MothFactorMatrix(nn.Module):
    '''
    A lot of this is from from:
    https://github.com/HazyResearch/butterfly/blob/master/learning_transforms/butterfly_old.py
    '''
    
    def __init__(self, n, b, k, zeroes = False):
        '''
        Parameters:
        n: int, dimension of the square matrix returned
        k: int, size of moth factors, power of 2 
        b: int, number of diagonals in moth factors (i.e bxb matrix of diagonal matrices), power of 2
        '''
        super().__init__()
        
        assert round(np.log2(n)) == np.log2(n), f"Dimensions should be a power of 2."
        assert round(np.log2(k)) == np.log2(k), f"Block size should be a power of 2."
        assert round(np.log2(b)) == np.log2(b), f"Block count should be a power of 2."
        
        self.n = n
        self.b = b
        self.k = k
        
        if zeroes:
            # Stores matrix in numpy array of shape (n/k, b, b, k/b)
            # All diagonals are flattened for space
            self.mat = nn.Parameter(torch.zeros((n//k,b,b,k//b)))
        else:
            self.mat = nn.Parameter(torch.randn((n//k,b,b,k//b)))
    
    def matrix(self):
        '''
        Converts class into a np matrix
        '''
        
        # Initialize everything for faster access
        b = self.b
        k = self.k
        n = self.n
        mat = self.mat
        shape = mat.shape
        
        ret = torch.zeros((n,n))
        for factor in range(shape[0]):
            start = factor*k    # The row/col to start inserting factor
            for row in range(shape[1]):
                row_iter = start+row*(k // b)
                for col in range(shape[2]):
                    col_iter = start+col*(k // b)
                    for diag_i in range(shape[3]):
                        ret[row_iter+diag_i,col_iter+diag_i] = mat[factor, row, col, diag_i]
        return ret
    
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
    
    def forward(self, input):
        """
        Parameters:
            input: (..., size)
        Return:
            output: (..., size)
        """
        # TODO: Need to optimize this.
        output = input @ self.matrix().T()
        return output


# In[ ]:




