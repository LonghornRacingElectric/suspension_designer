"""Kinematics module tests"""
import pytest
import warnings

import itertools as itl

import numpy as np
import scipy.spatial.transform as sptl

# import cProfile
# import pstats
# import io

from suspension_designer.geometry import EulerRotation
from suspension_designer.kinematics import KinematicTransform, KinematicSystem

from testing_utilities import random_uniform, random_euler_rotation

__all__ = ['TestKinematicTransform']

NUM_TRANSFORMS = 5
NUM_VECTORS = 5

# @pytest.mark.parametrize("position, rotation", 
#                          [(random_uniform(10,3), random_euler_rotation()) for _ in range(NUM_TRANSFORMS)])    
# class TestKinematicTransform():
#     raise NotImplementedError
    
class TestKinematicSystem():
    def _system_A(self) -> KinematicSystem:
        system = KinematicSystem()

        system.add_edges_from([
            ('O','A'), ('A','B'), 
            ('A','D'),
            ('O','B'), ('B','C'),
            ('O','C')])   

        return system 