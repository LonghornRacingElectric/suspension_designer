"""Testing Utility Functions"""
import numpy as np

from suspension_designer.geometry import EulerRotation

def random_uniform(scale: float, size: int) -> np.ndarray:
    """Generates uniform random vectors in radians between :math:`[-s, s)`

    :return: Scaled uniform randoms
    :rtype: np.ndarray
    """
    return scale * np.random.uniform(-1,1,(size,))

def random_sequence() -> str:
    """Generates random rotation sequence

    :return: Rotation sequence
    :rtype: str
    """
    return ''.join(np.random.permutation(list('XYZ')))

def random_euler_rotation() -> EulerRotation:
    """Generates random EulerRotation

    :return: Euler rotation object
    :rtype: EulerRotation
    """
    return EulerRotation(random_uniform(360,3), random_sequence)