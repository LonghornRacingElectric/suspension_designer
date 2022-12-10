"""kinematics.py - Kinematic Frames, Transforms, and Systems

- poi_factory()
- euler_rotation()
- KinematicFrame()
- KinematicTransform()
- KinematicSystem()
"""

# %% Imports
from __future__ import annotations

import collections as col
import numpy.typing as npt

import numpy as np
import networkx as nx


# %% Code
def poi_factory(name: str = '', position: npt.ArrayLike = np.empty(3), style: str = 'k.' ) -> dict:
    """Default Point of Interest Factory"""
    return {'name': name, 'position': np.array(position), 'style': style}


def euler_rotation(angle: float, axis: int, point: npt.ArrayLike) -> npt.ArrayLike:
    """Applies a rotation transform about a primary frame axis"""
    ia: int = np.mod(axis,3)
    iap1: int = np.mod(axis+1,3)
    iam1: int = np.mod(axis-1,3)

    R = np.zeros((3,3))
    R[ia  , ia  ] = 1
    R[iam1, iam1] = np.cos(angle)
    R[iap1, iap1] = np.cos(angle)
    R[iam1, iap1] =  np.sin(angle)
    R[iap1, iam1] = -np.sin(angle)

    return np.matmul(R, point)


class KinematicFrame(col.UserDict):
    """Kinematic reference frame in 3D space"""
    def __init__(self, name: str='', poi: dict=poi_factory(), style: str='k'):
        """Allocate KinematicFrame instance properties"""
        super().__init__({'name': name, 'poi': poi, 'style': style})


class KinematicTransform(col.UserDict):
    """Kinematic tranformation between reference frames"""
    def __init__(self, 
            position: npt.ArrayLike=np.empty(3),
            rotation: npt.ArrayLike=np.empty(3),
            sequence: npt.ArrayLike=np.array([0,3,2,1]) ):
        """Allocate KinematicTransform instance properties"""
        super().__init__({
            'position': np.array(position), 
            'rotation': np.array(rotation),
            'sequence': np.array(sequence),
        })

    def transform(self, point: npt.ArrayLike, direction: str='f' ) -> npt.ArrayLike:
        """Streamline invoking kinematic affine transformations"""
        if any([direction == d for d in ['f', 'forward']]):
            return self.forward_transform(np.array(point))
        elif any([direction == d for d in ['r', 'i', 'reverse', 'inverse']]):
            return self.inverse_transform(np.array(point))
        else:
            raise ValueError('Transformation direction argument not valid') 

    def forward_transform(self, pB: npt.ArrayLike) -> npt.ArrayLike:
        """Perform forward transform from base to follower frame"""
        pF = pB
        for s in self.sequence:
            if s == 0:
                pF = pF - self.position
            else:
                pF = euler_rotation(self.rotation[s-1], s-1, pF) 

        return pF 
            
    def inverse_transform(self, pF: npt.ArrayLike) -> npt.ArrayLike:
        """Perform inverse transform from follower to base frame"""
        pB = pF
        for s in np.flip(self.sequence):
            if s == 0:
                pB = pB + self.position
            else:
                pB = euler_rotation(-self.rotation[s-1], s-1, pB)

        return pB 


class KinematicSystem(nx.DiGraph):
    """Kinematic reference frame graph network system"""
    node_attr_dict_factory = KinematicFrame
    edge_attr_dict_factory = KinematicTransform

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.cycle = []

    def get_cycles(self, source=None, subgraph=None):
        if subgraph is None: 
            goi = self
        else:
            goi = self.subgraph(subgraph)

        cycle = []
        for c in nx.minimum_cycle_basis(goi.to_undirected(as_view=True)):
            cycle.append( nx.find_cycle(
                goi.subgraph(c), source=source, orientation='ignore') )

        return cycle