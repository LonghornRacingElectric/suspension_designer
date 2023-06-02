"""kinematics.py - Kinematic Frames, Transforms, and Systems"""
from __future__ import annotations

from collections import UserDict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

__all__ = ['poi_factory', 'default_poi_factory', 'euler_rotation', 
           'KinematicFrame', 'KinematicTransform', 'KinematicSystem']

def poi_factory(name: str, position: np.ndarray, style: str = 'k.') -> dict:
    """Point of interest factory"""
    return {name: {'position': np.array(position, dtype=np.double), 'style': style}}

def default_poi_factory() -> dict:
    """Default points of interest factory"""
    out = {}
    for poi in (['O', [0,0,0]], ['E0', [1,0,0]], ['E1', [0,1,0]], ['E2', [0,0,1]]):
        out.update(poi_factory(*poi)) 

    return out 

def euler_rotation(angle: np.number, axis: int, point: np.ndarray) -> np.ndarray:
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
    def __init__(self, name: str='', poi: dict=default_poi_factory(), color: str='k'):
        """Allocate KinematicFrame instance properties"""
        super().__init__({'name': name, 'poi': poi, 'color': color})

class KinematicTransform(UserDict):
    """Kinematic tranformation between reference frames"""
    def __init__(self, 
            position: np.ndarray = np.zeros(3, dtype=np.double),
            rotation: np.ndarray = np.zeros(3, dtype=np.double),                # intrinsic
            sequence: np.ndarray = np.array([0,3,2,1], dtype=np.ushort),
            dof     : np.ndarray = np.zeros(6, dtype=np.bool_) ):
        """Initialize KinematicTransform"""
        super().__init__({
            'position': np.array(position, dtype=np.double), 
            'rotation': np.array(rotation, dtype=np.double),
            'sequence': np.array(sequence, dtype=np.ushort),
            'dof'     : np.array(dof, dtype=np.bool_)})

    def transform(self, point: np.ndarray, direction: str = 'f') -> np.ndarray:
        """Streamline invoking kinematic affine transformations"""
        if any([direction == d for d in ['f', 'forward']]):
            return self.forward_transform(np.array(point))
        elif any([direction == d for d in ['r', 'i', 'reverse', 'inverse']]):
            return self.inverse_transform(np.array(point))
        else:
            raise ValueError('Transformation direction argument not valid') 
    
    def forward_transform(self, pB: np.ndarray) -> np.ndarray:
        """Perform forward transform from base to follower frame"""
        pF = pB
        for s in self['sequence']:
            if s == 0:
                pF = pF - self['position']
            else:
                pF = euler_rotation(self['rotation'][s-1], s-1, pF) 

        return pF 
            
    def inverse_transform(self, pF: np.ndarray) -> np.ndarray:
        """Perform inverse transform from follower to base frame"""
        pB = pF
        for s in np.flip(self['sequence']):
            if s == 0:
                pB = pB + self['position']
            else:
                pB = euler_rotation(-self['rotation'][s-1], s-1, pB)

        return pB 

    def rotate(self, d: np.ndarray, direction: str = 'f') -> np.ndarray:
        """Streamline invoking kinematic rotations"""
        if any([direction == d for d in ['f', 'forward']]):
            return self.forward_rotation(np.array(d))
        elif any([direction == d for d in ['r', 'i', 'reverse', 'inverse']]):
            return self.inverse_rotation(np.array(d))
        else:
            raise ValueError('Rotation direction argument not valid') 
        
    def forward_rotation(self, dB: np.ndarray) -> np.ndarray:
        """Perform forward rotation from base to follower frame"""
        dF = dB
        for s in self['sequence']:
            if s == 0:
                continue

            dF = euler_rotation(self['rotation'][s-1], s-1, dF) 

        return dF 
            
    def inverse_rotation(self, dF: np.ndarray) -> np.ndarray:
        """Perform inverse rotation from follower to base frame"""
        dB = dF
        for s in np.flip(self['sequence']):
            if s == 0:
                continue

            dB = euler_rotation(-self['rotation'][s-1], s-1, dB)

        return dB 
    
class KinematicSystem(nx.DiGraph):
    """Kinematic reference frame graph network system"""
    node_attr_dict_factory = KinematicFrame
    edge_attr_dict_factory = KinematicTransform

    def __init__(self, incoming_graph_data = None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self._path: dict[tuple[str,str], list[tuple[str,str,str]]] = {}

    def compute_weights(self):
        for e in self.edges():
            self.edges[e]['weight'] = 10+np.sum(self.edges[e]['dof'])

    def get_path(self, source: str, target: str) -> list[tuple[str,str,str]]:
        """Generates transform sequence between two coordinate frames"""
        if (source, target) in self._path.keys():
            return self._path[(source, target)]
        elif (target, source) in self._path.keys():
            self._path[(source, target)] = []
            for e in reversed(self._path[(target, source)]):
                d = 'r' if e[2] in ['f', 'forward'] else 'f'
                self._path[(source, target)].append((e[0],e[1],d))

            return self._path[(source, target)]
        
        # Path not previously determined
        nodes = nx.shortest_path(self.to_undirected(as_view=True), source, target)

        self._path[(source,target)] = []
        for j in range(len(nodes)-1):
            if (nodes[j], nodes[j+1]) in self.edges:
                self._path[(source, target)].append((nodes[j], nodes[j+1], 'forward'))
            elif (nodes[j+1], nodes[j]) in self.edges:
                self._path[(source, target)].append((nodes[j+1], nodes[j], 'reverse'))
            else:
                KeyError("Edge ({},{}) is not present".format(nodes[j], nodes[j+1]))

        return self._path[(source,target)]

    def get_loops(self, subgraph=None):
        """Generates a directed minimum loop basis for a subgraph"""
        if subgraph is None: 
            goi = self
        else:
            goi = self.subgraph(subgraph)

        mcb = nx.minimum_cycle_basis(goi.to_undirected(as_view=True), weight='weight')              # type: ignore

        self.loop = []
        for c in mcb:
            self.loop.append(nx.find_cycle(goi.subgraph(c), orientation='ignore'))                  # type: ignore

    def coord(self, poi: str | np.ndarray, frame: str, out_frame: str | None = None) -> np.ndarray:
        """Reports point of interest coordinates in requested frame"""
        if isinstance(poi, str):
            point = self.nodes[frame]['poi'][poi]['position']
        else:
            point = poi

        if out_frame is None:
            return point
        
        for (frame_A, frame_B, direction) in self.get_path(frame, out_frame):
            point = self.edges[frame_A, frame_B].transform(point, direction)

        return point

    def direction(self, d: str | np.ndarray, frame: str, out_frame: str | None = None) -> np.ndarray:
        """Reports direction in requested frame"""
        if isinstance(d, str):
            d = self.nodes[frame]['poi'][d]['position']

        d = d / np.linalg.norm(d)

        if out_frame is None:
            return d
        
        for (frame_A, frame_B, direction) in self.get_path(frame, out_frame):
            d = self.edges[frame_A, frame_B].rotate(d, direction)

        return d

    def plot(self, ax: plt.Axes | None = None, frame: str | None = None, size: int = 1):
        """3D plots of KinematicSystem"""
        # Default parameters
        ax = plt.gca() if ax is None else ax
        frame = list(self.nodes)[0] if frame is None else frame

        # Plot Cartesian frames
        for node in self.nodes():
            O = self.coord('O', node, frame)
            for i in range(3):
                ax.quiver3D(*O, *(self.coord(f"E{i}", node, frame)-O),                              # type: ignore
                    color=self.nodes[node]['color'], length=size, normalize=True)

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')                                                                     # type: ignore
        ax.set_aspect('equal')

    def front_view_plot(self, ax: plt.Axes | None= None, frame: str | None = None):
        """Plot cardinal front view of KinematicSystem"""
        # Default parameters
        ax = plt.gca() if ax is None else ax
        frame = list(self.nodes)[0] if frame is None else frame
        
        # Plot Cartesian frames
        label_map = {0: 'X', 1: 'Y', 2: 'Z'}
        plt.sca(ax)

        i = [1,2]
        for node in self.nodes():
            O = self.coord('O', node, frame)
            for ii in i:
                E = self.coord(f"E{ii}", node, frame) - O
                ax.quiver(*O[i], *E[i], color=self.nodes[node]['color'])

        ax.set_title('Front View')
        ax.set_xlabel('Y [mm]')
        ax.set_ylabel('Z [mm]')                                                                
        ax.set_aspect('equal')

    def three_view_plot(self, fig: plt.Figure | None= None, frame: str | None = None):
        """Plot three cardinal views of KinematicSystem"""
        # Default parameters
        fig = plt.figure() if fig is None else fig
        frame = list(self.nodes)[0] if frame is None else frame

        axs = [fig.add_subplot(3,1,j+1) for j in range(3)]
        axs[1].invert_xaxis()
        
        # Plot Cartesian frames
        title_map = {0: 'Top', 1: 'Profile', 2: 'Front'}
        label_map = {0: 'X', 1: 'Y', 2: 'Z'}
        for j, ax in enumerate(axs):
            plt.sca(ax)

            i = [jj for jj in range(3) if jj != 2-j]
            for node in self.nodes():
                O = self.coord('O', node, frame)
                for ii in i:
                    E = self.coord(f"E{ii}", node, frame) - O
                    ax.quiver(*O[i], *E[i], color=self.nodes[node]['color'])

            ax.set_title(f"{title_map[j]} View")
            ax.set_xlabel(f"{label_map[i[0]]} [mm]")
            ax.set_ylabel(f"{label_map[i[1]]} [mm]")                                                                
            ax.set_aspect('equal')

        fig.tight_layout()