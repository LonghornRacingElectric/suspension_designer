"""Geometry module tests"""
import pytest

import numpy as np
import scipy.spatial.transform as sptl

from suspension_designer.geometry import Line, Plane, EulerRotation       

from testing_utilities import random_uniform, random_sequence

__all__ = ['TestLine', 'TestPlane', 'TestEulerRotation']

NUM_ROTATIONS = 10

# class TestLine():
#     raise NotImplementedError

# class TestPlane():
#     raise NotImplementedError

@pytest.mark.parametrize('rotation, sequence', 
    [(random_uniform(2*np.pi,3), random_sequence()) for _ in range(NUM_ROTATIONS)])
class TestEulerRotation():
    @staticmethod
    def rotation_pair_factory(rotation: np.ndarray, sequence: str, degrees: bool) \
            -> tuple[sptl.Rotation, EulerRotation]:
        """Creates pair of rotations for testing comparision

        :param rotation: X-Y-Z rotation angles 
        :type rotation: np.ndarray

        :param sequence: Rotation sequence
        :type sequence: str

        :param degrees: Flag for whether rotation angles are specified in degrees or radians
        :type degrees: bool

        :return: Rotation pair
        :rtype: tuple[sptl.Rotation, EulerRotation]
        """
        angles = np.array([rotation['XYZ'.find(j)] for j in sequence])

        return (sptl.Rotation.from_euler(sequence, angles, degrees), 
                EulerRotation(rotation, sequence, degrees))

    def test_initialization(self, rotation: np.ndarray, sequence: str):
        """Tests rotation initialization via Frobenius norm

        :param rotation: X-Y-Z rotation angles
        :type rotation: np.ndarray

        :param sequence: Rotation sequence
        :type sequence: str
        """
        ref, obj = self.rotation_pair_factory(rotation, sequence, degrees=False)

        ref_mat = ref.as_matrix()
        mat = obj.as_matrix()

        frob_error = np.linalg.norm(mat - ref_mat, 'fro') / np.linalg.norm(ref_mat, 'fro')
        assert frob_error <= 1e-14, f"Test Failed: {frob_error} > 1e-14"

    def test_sync_operator(self, rotation: np.ndarray, sequence: str):
        """Tests rotation initialization via Frobenius norm

        :param rotation: X-Y-Z rotation angles
        :type rotation: np.ndarray

        :param sequence: Rotation sequence
        :type sequence: str
        """
        ref, obj = self.rotation_pair_factory(rotation, sequence, degrees=False)

        ref_mat = ref.as_matrix()
        mat = obj.as_matrix()

        frob_error = np.linalg.norm(mat - ref_mat, 'fro') / np.linalg.norm(ref_mat, 'fro')
        assert frob_error <= 1e-14, f"Initial Test Failed: {frob_error} > 1e-14"

        new_rotation, new_sequence = random_uniform(2*np.pi,3), random_sequence()
        new_ref, _ = self.rotation_pair_factory(new_rotation, new_sequence, degrees=False)

        obj[:] = new_rotation
        obj.sequence = new_sequence

        new_ref_mat = new_ref.as_matrix()
        new_mat = obj.as_matrix()

        new_frob_error = np.linalg.norm(new_mat - new_ref_mat, 'fro') / np.linalg.norm(new_ref_mat, 'fro')
        assert new_frob_error <= 1e-14, f"Sync Test Failed: {new_frob_error} > 1e-14"