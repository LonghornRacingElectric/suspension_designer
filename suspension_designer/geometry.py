"""geometry.py - Geometry Utility Functions"""
from __future__ import annotations

import typing as typ
from typing import Any
from numpy._typing import NDArray
import numpy.typing as npt

import operator as op

import numpy as np

import scipy.linalg as sla
import scipy.spatial.transform as sptl

from suspension_designer.utilities import ordered_unique, sequence_to_index

__all__ = ['lerp',                                  # interpolation
           'Line', 'Plane',                         # linear subspaces
           'Rotation',
           'vector_alignment_rotation',             # rotations
           'skew_symmetric_cross_matrix']           # helpers

# %% Interpolation
def lerp(point_A: np.ndarray, point_B: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation between two points

    :param point_A: First point
    :type point_A: numpy.ndarray
    
    :param point_B: Second point
    :type point_B: numpy.ndarray
    
    :param alpha: Interpolation scalar
    :type alpha: float
    
    :return: Linear interpolation result
    :rtype: numpy.ndarray
    """
    return point_A*(1-alpha) + point_B*alpha

# %% Linear Subspaces
class Line():
    """A 1D line in :math:`\mathbb{R}^{dimension}`
    
    :param point_A: First point defining line
    :type point_A: numpy.ndarray
    
    :param point_B: Second point defining line
    :type point_B: numpy.ndarray
    """
    def __init__(self, point_A: np.ndarray, point_B: np.ndarray):
        """Initialize Line"""
        self.point = np.array([point_A, point_B])
        self.basis = (point_B - point_A) / np.linalg.norm(point_B - point_A)
        self.dimension = self.point.shape[-1]

    def __call__(self, query: float, index: int = 0) -> np.ndarray:
        """Evaluates line between two points given coordinate and index

        :param query: Query coordinate
        :type query: float
        
        :param index: Query coordinate index, defaults to 0
        :type index: int, optional
        
        :return: Linear interpolation result
        :rtype: numpy.ndarray
        """
        alpha = (query - self.point[0][index]) \
            / (self.point[1][index] - self.point[0][index])
        return lerp(self.point[0], self.point[1], alpha)

    def proj(self, point: np.ndarray) -> np.ndarray:
        """Orthogonal projection onto the line
        
        :param point: Point of interest
        :type point: numpy.ndarray
        
        :return: Point projection onto line
        :rtype: numpy.ndarray
        """
        return self.point[0] + np.dot(point - self.point[0], self.basis) * self.basis
    
    def perp(self, point: np.ndarray) -> np.ndarray:
        """Perpendicular component to the orthogonal projection onto the line
        
        :param point: Point of interest
        :type point: numpy.ndarray
        
        :return: Perpendicular component to the projection of point from the line
        :rtype: numpy.ndarray
        """
        return point - self.proj(point)

class Plane():
    """A 2D plane in :math:`\mathbb{R}^{3}`
    
    :param point_A: First point defining plane
    :type point_A: numpy.ndarray
    
    :param point_B: Second point defining plane
    :type point_B: numpy.ndarray

    :param point_C: Third point defining plane
    :type point_C: numpy.ndarray
    """
    def __init__(self, point_A: np.ndarray, point_B: np.ndarray, point_C: np.ndarray):
        """Initialize Plane"""
        self.point = np.array([point_A, point_B, point_C])
        self.dimension = self.point.shape[-1]
        if self.dimension != 3:
            raise ValueError('Provided points are not three dimensional')
        
        A = np.hstack((self.point, np.ones((3,1))))
        self.null_space = sla.null_space(A).squeeze()
        if len(self.null_space.shape) != 1:
            raise ValueError('Provided points do not create a valid plane')
        
        self.n = self.null_space[:3] / np.linalg.norm(self.null_space[:3])
        
    def __call__(self, query: typ.Collection[float, float], 
                 index: typ.Collection[int, int] = (0,1)) -> np.ndarray:
        """Evaluate plane at query coordinates for given dimension indices
        
        :param query: Query coordinates
        :type query: typing.Collection[float, float]

        :param index: Query coordinate indices, defaults to (0,1)
        :type index: typing.Collection[int, int], optional

        :return: Planar evaluation
        :rtype: numpy.ndarray
        """
        if index != ordered_unique(index) or len(index) != 2:
            raise ValueError('Invalid index set')

        point = np.empty((3,))
        for j, qj in zip(index, query):
            point[j] = qj
        
        i = [i for i in range(3) if i not in index][0]
        point[i] = (-self.null_space[-1] - np.dot(self.null_space[[index]], query)) \
            / self.null_space[i] 

        return point

    def intersection(self, other: Plane) -> Line:
        """Compute linear intersection of two planes
        
        :param other: Intersecting plane
        :type other: Plane
        
        :return: Intersection line
        :rtype: Line
        """
        A = np.array([self.null_space, other.null_space])
        j = np.unravel_index(np.argmin(np.abs(A[:,:3])), A[:,:3].shape)[1]

        Ar = A[:,[jj for jj in range(3) if jj != j]]
        b0 = A[:,-1]
        b1 = A[:,-1] + A[:,j]

        point_A = np.insert(np.linalg.solve(Ar, -b0), 0, 0)
        point_B = np.insert(np.linalg.solve(Ar, -b1), 0, 100)

        return Line(point_A, point_B)
    
    def proj(self, point: np.ndarray) -> np.ndarray:
        """Orthogonal projection onto the plane
        
        :param point: Point of interest
        :type point: numpy.ndarray
        
        :return: Point projection onto plane
        :rtype: numpy.ndarray
        """
        return point - self.perp(point)

    def perp(self, point: np.ndarray) -> np.ndarray:
        """Perpendicular component to the orthogonal projection onto the plane
        
        :param point: Point of interest
        :type point: numpy.ndarray
        
        :return: Perpendicular component to orthogonal projection onto plane
        :rtype: numpy.ndarray
        """
        return self.point[0] + np.dot(point - self.point[0], self.n) * self.n

# %% Rotations
class EulerRotation(np.ndarray):
    """Euler / Tait-Bryan angle based rotations. Subclasses :code:`numpy.ndarray` 
    and wraps a :code:`scipy.spatial.transform.Rotation` object for efficient
    rotation computations.

    :param rotation: Euler / Tait-Bryan angles
    :type rotation: numpy.typing.Arraylike

    :param sequence: Rotation angle sequence, defaults to 'ZYX' (intrinsic)
    :type sequence: str, optional

    :param degrees: Unit of rotation angles, defaults to True
    :type degrees: bool, optional
    """
    def __new__(cls, rotation: npt.ArrayLike, 
                sequence: str = 'ZYX', degrees: bool = True) -> EulerRotation:
        """See: https://numpy.org/doc/stable/user/basics.subclassing.html"""
        obj = np.asarray(rotation).view(cls)

        obj.sequence = sequence
        obj.degrees = degrees

        obj._operator_synced = False
        obj._operator = None

        return obj
    
    def __array_finalize__(self, obj: np.ndarray | None):
        """See: https://numpy.org/doc/stable/user/basics.subclassing.html"""
        if obj is None: return
        
        self._operator = None

        self._sequence = getattr(obj, '_sequence', 'ZYX')
        self.sequence = getattr(obj, 'sequence', self._sequence)

        self._index = getattr(obj, '_index', sequence_to_index(self.sequence))

        self.degrees = getattr(obj, 'degrees', True)
        
        # running self._set_operator() here incurs a recursion loop?
        self._operator_synced = getattr(obj, '_operator_synced', False)
        self._operator = getattr(obj, '_operator', None)
    
    def __setitem__(self, key: int | slice, value: float):
        """Invoke :code:`numpy.ndarray` `__setitem__()` dunder and then set 
        corresponding operator."""
        super().__setitem__(key, value)

    sequence: str = property(op.attrgetter('_sequence'))
    
    @sequence.setter
    def sequence(self, value: str):
        """Sets sequence field and computes correct corresponding indexing list

        :param value: Sequence value
        :type value: str
        """
        self._index = sequence_to_index(value)
        self._sequence = value
    
    def __str__(self) -> str:
        return f"EulerRotation([{self[0]}, {self[1]}, {self[2]}], {self.sequence}, degrees={self.degrees})"
    
    def __repr__(self) -> str:
        return str(self)

    def _set_operator(self):
        """Set :code:`scipy.spatial.transform.Rotation` operator attribute from 
        rotation angles, sequence, and degree flag."""
        self._operator = sptl.Rotation.from_euler(
            self.sequence, self[self._index], self.degrees)

    def apply(self, vector: npt.ArrayLike) -> np.ndarray:
        """Applies forward rotation to input vector(s)
        
        :param vector: Input vector(s) of shape :math:`(n,3)` where :math:`n` is
            the number of vectors to be rotated 
        :type vector: numpy.ndarray
        
        :return: Rotated vectors with shape :math:`(n,3)`
        :rtype: numpy.ndarray
        """
        if not self._operator_synced:
            self._set_operator()
        
        return self._operator.apply(vector)
    
    def inv(self, vector: npt.ArrayLike) -> np.ndarray:
        """Applies inverse rotation to input vector(s)
        
        :param vector: Input vector(s) of shape :math:`(n,3)` where :math:`n` is
            the number of vectors to be rotated 
        :type vector: numpy.ndarray
        
        :return: Inversely rotated vectors with shape :math:`(n,3)`
        :rtype: numpy.ndarray
        """
        if not self._operator_synced:
            self._set_operator()

        return self._operator.apply(vector, inverse=True)

def vector_alignment_rotation(vector_A: np.ndarray, vector_B: np.ndarray) -> sptl.Rotation:
    """Provide a rotation matrix to align the first vector onto the second
    Reference: https://math.stackexchange.com/a/476311

    :param vector_A: Misaligned vector
    :type vector_A: numpy.ndarray

    :param vector_B: Reference direction vector
    :type vector_B: numpy.ndarray

    :return: Alignment rotation
    :rtype: scipy.spatial.transform.Rotation
    """
    v = np.cross(vector_A, vector_B)
    c = np.dot(vector_A, vector_B)
         
    v_skew = skew_symmetric_matrix(v)
    R = np.eye(3) + v_skew + (v_skew @ v_skew) / (1 + c)
    return sptl.Rotation.from_matrix(R)
    
def two_angle_vector_alignment_rotation(
    direction: typ.Callable[[], np.ndarray], sequence: str = 'ZYX'):
    """Used for two force members with ball-ball joints to align without
    rotation along the free axis. The free axis should be the third in the sequence.
    
    :param sequence: Rotation sequence, defaults to 'ZYX' 
    :type sequence: str, optional
    
    :param direction: Alignment point
    :type direction: numpy.ndarray
    """
    index = sequence_to_index(sequence)

    # for i in index[:-1]:

# %% Helpers
def skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
    r"""Creates skew symmetric cross-product matrix corresponding 
    to vector in :math:`\mathbb{R}^3`

    :param v: Input vector
    :type v: numpy.ndarray

    :return: Skew symmetric cross-product matrix
    :rtype: numpy.ndarray
    """
    return np.array([[ 0   , -v[2],  v[1]],
                     [ v[2],  0   , -v[0]],
                     [-v[1],  v[0],  0   ]])