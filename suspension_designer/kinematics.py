"""kinematics.py - Kinematic Frames, Transforms, and Systems"""
from __future__ import annotations

from collections import UserDict
from numpy.typing import ArrayLike

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

__all__ = ['poi_factory', 'default_poi_factory', 'euler_rotation', 
           'KinematicFrame', 'KinematicTransform', 'KinematicSystem']


def poi_factory(name: str, position: ArrayLike, style: str = 'k.' ) -> dict:
    """Point of Interest Factory"""
    return {name: {'position': np.array(position, dtype=np.double), 'style': style}}


def default_poi_factory() -> dict:
    """Default Points of Interest Factory"""
    out = {}
    for poi in (['O', [0,0,0]], ['E0', [1,0,0]], ['E1', [0,1,0]], ['E2', [0,0,1]]):
        out.update(poi_factory(*poi)) 

    return out 


def euler_rotation(angle: float, axis: int, point: ArrayLike) -> ArrayLike:
    """Applies a rotation transform about a primary frame axis"""
    ia  : int = np.mod(axis,3)
    iap1: int = np.mod(axis+1,3)
    iam1: int = np.mod(axis-1,3)

    R = np.zeros((3,3), dtype=np.double)
    R[ia  , ia  ] = 1
    R[iam1, iam1] =  np.cos(angle)
    R[iap1, iap1] =  np.cos(angle)
    R[iam1, iap1] =  np.sin(angle)
    R[iap1, iam1] = -np.sin(angle)

    return np.matmul(R, point)


class KinematicFrame(UserDict):
    """Kinematic reference frame in 3D space"""
    def __init__(self, name: str='', poi: dict=default_poi_factory(), style: str='k'):
        """Allocate KinematicFrame instance properties"""
        super().__init__({'name': name, 'poi': poi, 'style': style})

class KinematicTransform(UserDict):
    """Kinematic tranformation between reference frames"""
    def __init__(self, 
            position: ArrayLike=np.zeros(3, dtype=np.double),
            rotation: ArrayLike=np.zeros(3, dtype=np.double),
            sequence: ArrayLike=np.array([0,3,2,1], dtype=np.ushort),
            dof     : ArrayLike=np.zeros(6, dtype=np.bool_) ):
        """Allocate KinematicTransform instance properties"""
        super().__init__({
            'position': np.array(position, dtype=np.double), 
            'rotation': np.array(rotation, dtype=np.double),
            'sequence': np.array(sequence, dtype=np.ushort),
            'dof'     : np.array(dof, dtype=np.bool_),
        })

    def transform(self, point: ArrayLike, direction: str='f' ) -> ArrayLike:
        """Streamline invoking kinematic affine transformations"""
        if any([direction == d for d in ['f', 'forward']]):
            return self.forward_transform(np.array(point))
        elif any([direction == d for d in ['r', 'i', 'reverse', 'inverse']]):
            return self.inverse_transform(np.array(point))
        else:
            raise ValueError('Transformation direction argument not valid') 

    def forward_transform(self, pB: ArrayLike) -> ArrayLike:
        """Perform forward transform from base to follower frame"""
        pF = pB
        for s in self['sequence']:
            if s == 0:
                pF = pF - self['position']
            else:
                pF = euler_rotation(self['rotation'][s-1], s-1, pF) 

        return pF 
            
    def inverse_transform(self, pF: ArrayLike) -> ArrayLike:
        """Perform inverse transform from follower to base frame"""
        pB = pF
        for s in np.flip(self['sequence']):
            if s == 0:
                pB = pB + self['position']
            else:
                pB = euler_rotation(-self['rotation'][s-1], s-1, pB)

        return pB 


class KinematicSystem(nx.DiGraph):
    """Kinematic reference frame graph network system"""
    node_attr_dict_factory = KinematicFrame
    edge_attr_dict_factory = KinematicTransform

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.cycle = []

    def compute_weights(self):
        for e in self.edges():
            self.edges[e]['weight'] = 10+np.sum(self.edges[e]['dof'])

    def get_path(self, source: str, target: str) -> list[tuple[str,str,str]]:
        """Generates transform sequence between two coordinate frames"""
        nodes = nx.shortest_path(self.to_undirected(as_view=True), source, target)

        path = []
        for j in range(len(nodes)-1):
            if (nodes[j], nodes[j+1]) in self.edges:
                path.append((nodes[j], nodes[j+1], 'forward'))
            elif (nodes[j+1], nodes[j]) in self.edges:
                path.append((nodes[j+1], nodes[j], 'reverse'))
            else:
                KeyError("Edge ({},{}) is not present".format(nodes[j], nodes[j+1]))

        return path

    def get_loops(self, subgraph=None):
        """Generates a directed minimum loop basis for a subgraph"""
        if subgraph is None: 
            goi = self
        else:
            goi = self.subgraph(subgraph)

        mcb = nx.minimum_cycle_basis(goi.to_undirected(as_view=True), weight='weight')

        self.loop = []
        for c in mcb:
            self.loop.append(nx.find_cycle(goi.subgraph(c), orientation='ignore'))

    def coord(self, point: str | ArrayLike, frame: str, outFrame: str) -> ArrayLike:
        """Reports point of interest coordinates in requested frame"""
        # Fetch point
        if isinstance(point, str):
            point = self.nodes[frame]['poi'][point]['position']

        for (frameA, frameB, direction) in self.get_path(frame, outFrame):
            point = self.edges[frameA, frameB].transform(point, direction)

        return point

    def plot(self, ax: plt.axes=None, frame: str=None, size: int=1):
        """Plots KinematicSystem"""
        # Default parameters
        ax = plt.gca() if ax is None else ax
        frame = list(self.nodes)[0] if frame is None else frame

        # Plot Cartesian frames
        for node in self.nodes():
            O = self.coord('O', node, frame)
            for i in range(3):
                ax.quiver3D(*O, *(self.coord('E{}'.format(i), node, frame)-O), 
                    color='k', length=size, normalize=True)

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal')