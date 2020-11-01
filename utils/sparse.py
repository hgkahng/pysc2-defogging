# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp


class Sparse3D(object):
    def __init__(self, *sparse_matrices):
        self.data = sp.vstack(sparse_matrices, "dok")
        self.shape = (len(sparse_matrices), *sparse_matrices[0].shape)
        
        self._dim1_jump = np.arange(0, self.shape[1] * self.shape[0], self.shape[1])
        self._dim1 = np.arange(self.shape[0])
        self._dim2 = np.arange(self.shape[1])

    def __getitem__(self, pos):
        if not type(pos) == tuple:
            if not hasattr(pos, "__iter__") and not type(pos) == slice:
                return self.data[self._dim1_jump[pos] + self._dim2]
            else:
                return Sparse3D(*(self[self._dim1[i]] for i in self._dim1[pos]))
        elif len(pos) > 3:
            raise IndexError("too many indices for array")
        else:
            if (not hasattr(pos[0], "__iter__") and not type(pos[0]) == slice or
                not hasattr(pos[1], "__iter__") and not type(pos[1]) == slice):
                if len(pos) == 2:
                    result = self.data[self._dim1_jump[pos[0]] + self._dim2[pos[1]]]
                else:
                    result = self.data[self._dim1_jump[pos[0]] + self._dim2[pos[1]], pos[2]].T
                    if hasattr(pos[2], "__iter__") or type(pos[2]) == slice:
                        result = result.T
                return result
            else:
                if len(pos) == 2:
                    return Sparse3D(*(self[i, self._dim2[pos[1]]] for i in self._dim1[pos[0]]))
                else:
                    if not hasattr(pos[2], "__iter__") and not type(pos[2]) == slice:
                        return sp.vstack([self[self._dim1[pos[0]], i, pos[2]]
                                          for i in self._dim2[pos[1]]]).T
                    else:
                        return Sparse3D(*(self[i, self._dim2[pos[1]], pos[2]] 
                                          for i in self._dim1[pos[0]]))

    def toarray(self):
        return np.array([self[i].toarray() for i in range(self.shape[0])])