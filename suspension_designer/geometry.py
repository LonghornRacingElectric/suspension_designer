"""geometry.py - Geometry Utility Functions"""
from __future__ import annotations

from typing import Collection

import numpy as np

import scipy.linalg as sla
from scipy.spatial.transform import Rotation

__all__ = ['lerp',                                  # interpolation
           'Line', 'Plane',                         # geometric objects
           'alignment_angles',                      # utilities
           'skew_symmetric_cross_matrix'            # helpers
           'idx_sequence_to_str', 'unique']         # "

# %% Interpolation
def lerp(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two points"""
    return p0*(1-t) + p1*t

# %% Geometric Objects
class Line():
    """A 1D line
    
    :param p0: First point defining line
    :type p0: numpy.ndarray
    
    :param p1: Second point defining line
    :type p1: numpy.ndarray
    """
    def __init__(self, p0: np.ndarray, p1: np.ndarray):
        """Initialize Line"""
        self.p = np.array([p0, p1])
        self.d = self.p.shape[-1]

    def __call__(self, xq: float, j: int = 0):
        """Evaluates line between two points given coordinate and index"""
        t = (xq - self.p[0][j]) / (self.p[1][j] - self.p[0][j])

        return lerp(self.p[0], self.p[1], t)

class Plane():
    """A 2D plane
    
    :param p0: First point defining plane
    :type p0: numpy.ndarray
    
    :param p1: Second point defining plane
    :type p1: numpy.ndarray

    :param p2: Third point defining plane
    :type p2: numpy.ndarray
    """
    def __init__(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
        """Initialize Plane"""
        self.p = np.array([p0, p1, p2])
        self.d = self.p.shape[-1]
        
        A = np.hstack((self.p, np.ones((3,1))))
        self.N = sla.null_space(A).squeeze()

        if len(self.N.shape) != 1:
            raise ValueError('Provided points do not create a valid plane')
        
    def __call__(self, xq: Collection[float, float], 
                        j: Collection[int, int] = (0,1)) -> np.ndarray:
        """Evaluate plane at query coordinates for given dimension indices"""
        j = unique(j)

        p = np.zeros((3,))
        for jj, xqj in zip(j, xq):
            p[jj] = xqj
        
        i = [i for i in range(3) if i not in j][0]
        p[i] = (-self.N[-1] - np.dot(self.N[[j]], xq)) / self.N[i] 

        return p

    def intersection(self, other: Plane) -> Line:
        """Compute linear intersection of two planes"""
        A = np.array([self.N, other.N])
        j = np.unravel_index(np.argmin(np.abs(A[:,:3])), A[:,:3].shape)[1]

        Ar = A[:,[jj for jj in range(3) if jj != j]]
        b0 = A[:,-1] + A[:,j]*0
        b1 = A[:,-1] + A[:,j]*100

        p0 = np.insert(np.linalg.solve(Ar, -b0), 0, 0)
        p1 = np.insert(np.linalg.solve(Ar, -b1), 0, 100)

        return Line(p0, p1)
        
# %% Rotation Conversions
def alignment_rotation(e: np.ndarray, a: np.ndarray) -> Rotation:
    """Provide a rotation to align the first vector onto the second"""
    v = np.cross(e, a)
    c = np.dot(e, a)
         
    v_skew = skew_symmetric_cross_matrix(v)
    return Rotation.from_matrix(np.eye(3) + v_skew + (v_skew @ v_skew) / (1 + c))

def alignment_sequence(e: np.ndarray, a: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Provide an intrinsic Euler sequence to align the first vector onto the second"""
    rot = alignment_rotation(e, a)
    ang = rot.as_euler(idx_sequence_to_str(s))
    return np.array([ang[i-1] for i in np.argsort(s)[1:]])

def matrix_from_euler(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Convert intrinsic Euler angle sequence to rotation matrix"""
    rot = Rotation.from_euler(idx_sequence_to_str(s), r)
    return rot.as_matrix()

def euler_from_matrix(R: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Convert rotation matrix """
    rot = Rotation.from_matrix(R)
    return rot.as_euler(idx_sequence_to_str(s))

# %% Helpers
def skew_symmetric_cross_matrix(v: np.ndarray) -> np.ndarray:
    r"""Creates skew symmetric cross-product matrix corresponding 
    to vector in :math:`\mathbb{R}^3`
    """
    return np.array([[ 0   , -v[2],  v[1]],
                        [ v[2],  0   , -v[0]],
                        [-v[1],  v[0],  0   ]])

def idx_sequence_to_str(s: np.ndarray) -> str:
    sequence_map = {0: '', 1: 'X', 2: 'Y', 3: 'Z'}
    return ''.join([sequence_map[ss] for ss in s])

def unique(col: Collection) -> tuple:
    seen = set()
    return tuple(x for x in col if not (x in seen or seen.add(x)))
