"""kinematics.py - Kinematic Frames, Transforms, and Systems"""
from __future__ import annotations

import numpy.typing as npt
from dataclasses import dataclass

import operator as op

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from suspension_designer.geometry import EulerRotation

__all__ = ['DatumPoint', 'datum_point_factory', 'default_datum_factory', 
           'KinematicFrame', 'KinematicTransform', 'KinematicSystem']

#%% Datums
@dataclass
class DatumPoint:
    """Reference point datum

    :param name: Datum name
    :type name: str

    :param position: Datum position
    :type position: numpy.ndarray

    :param style: Plotting style, defaults to 'k.'
    :type style: str, optional
    """
    name: str
    position: np.ndarray
    style: str = 'k.'

def datum_point_factory(name: str, position: npt.ArrayLike, style = 'k.') -> dict[str, DatumPoint]:
    """Datum point dictionary factory
    
    :param name: Datum name
    :type name: str

    :param position: Datum position
    :type position: numpy.typing.ArrayLike

    :param style: Plotting style, defaults to 'k.'
    :type style: str, optional

    :return: Datum point dictionary pair
    :rtype: dict[str, DatumPoint]
    """
    return {name: DatumPoint(name, np.array(position, dtype=np.double), style)}

def default_datum_factory() -> dict:
    """Default Cartesian datum dictionary factory
    
    :return: Default datum point dictionary
    :rtype: dict[str, DatumPoint]
    """
    default_name     = ['O']     + [f'E{i}'      for i in range(3)]
    default_position = [[0,0,0]] + [np.eye(3)[i] for i in range(3)]

    out = {}
    for name, position in zip(default_name, default_position):
        out.update(datum_point_factory(name, position)) 

    return out 

class KinematicFrame(dict):
    """Kinematic reference frame in 3D space
    
    :param name: Frame name, defaults to ''
    :type name: str, optional

    :param datum: Datums associated with frame, defaults to default_datum_factory()
    :type datum: dict[str, DatumPoint], optional

    :param color: Plotting color, defaults to 'k'
    :type color: str, optional
    """
    def __init__(self, 
                 name: str = '', 
                 datum: dict[str, DatumPoint] = default_datum_factory(), 
                 color: str = 'k'):
        """Initialize KinematicFrame"""
        self.name = name
        self.datum = datum
        self.color = color

class KinematicTransform(dict):
    """Kinematic tranformation between reference frames

    :param position: Frame origin position in base frame, defaults to :code:`numpy.zeros(3)`
    :type position: numpy.typing.ArrayLike, optional

    :param angle: Intrinsic Euler angles [X,Y,Z], defaults to :code:`numpy.zeros(3)`
    :type angle: numpy.typing.ArrayLike, optional

    :param sequence: Euler angle sequence, defaults to :code`'ZYX'`
    :type sequence: str, optional

    :param degrees: Flag to denote if angles are supplied in degrees, defaults to :code:`False`
    :type degrees: bool, optional
    """
    def __init__(self, 
            position: npt.ArrayLike | None = None,  # Longitudinal (X), Lateral (Y), Vertical (Z)
            angle   : npt.ArrayLike | None = None,  # Roll (X), Pitch (Y), Yaw (Z)
            sequence: str = 'ZYX', degrees: bool = True):
        """Intialize KinematicTransform"""
        position = position if position is not None else np.zeros(3)
        angle    = angle    if angle    is not None else np.zeros(3)

        self.position = position
        self.rotation = EulerRotation(angle, sequence, degrees)
    
    position: np.ndarray = property(op.attrgetter('_position'))

    @position.setter
    def position(self, value: npt.ArrayLike):
        """Sets position to array conversion with double datatype
        
        :param value: Relative position between frames
        :type value: numpy.ndarray
        """
        self._position = np.array(value, dtype=np.double)

    @property
    def sequence(self) -> str:
        """Returns rotation sequence from rotation attribute
        
        :return: Sequence string
        :rtype: str
        """
        return self.rotation.sequence
    
    @sequence.setter
    def sequence(self, value: str):
        """Sets rotation sequence through rotation attribute
        
        :param value: Rotation sequence
        :type value: str
        """
        self.rotation.sequence = value

    def transform(self, point: npt.ArrayLike, orientation: str = 'f') -> np.ndarray:
        """Streamline kinematic affine transformations

        :param point: Point position vector
        :type point: numpy.typing.ArrayLike

        :param orientation: Transformation orientation, defaults to 'f'
        :type orientation: str, optional

        :raises ValueError: If `orientation` is not recognized
        
        :return: Point position vector in new frame
        :rtype: numpy.ndarray
        """
        if orientation in ['f', 'forward']:
            return self.forward_transform(np.array(point))
        elif orientation in ['r', 'i', 'reverse', 'inverse']:
            return self.inverse_transform(np.array(point))
        else:
            raise ValueError('Transformation orientation argument not valid') 
    
    def forward_transform(self, point: np.ndarray) -> np.ndarray:
        """Perform forward transform from base to follower frame
        
        :param point: Point position vector in base frame
        :type point: numpy.ndarray

        :return: Point position vector in follower frame
        :rtype: numpy.ndarray
        """
        return self.rotation.apply(point - self.position)
            
    def inverse_transform(self, point: np.ndarray) -> np.ndarray:
        """Perform inverse transform from follower to base frame
        
        :param point: Point position vector in follower frame
        :type point: numpy.ndarray

        :return: Point position vector in base frame
        :rtype: numpy.ndarray
        """
        return self.rotation.apply(point, inverse=True) + self.position

    def rotate(self, direction: npt.ArrayLike, orientation: str = 'f') -> np.ndarray:
        """Streamline kinematic affine transformations
        
        :param point: Direction vector
        :type point: numpy.typing.ArrayLike

        :param orientation: Transformation orientation, defaults to 'f'
        :type orientation: str, optional

        :raises ValueError: If `orientation` is not recognized
        
        :return: Direction vector in new frame
        :rtype: numpy.ndarray
        """
        if orientation in ['f', 'forward']:
            return self.rotation.apply(np.array(direction))
        elif orientation in ['r', 'i', 'reverse', 'inverse']:
            return self.rotation.apply(np.array(direction), inverse=True)
        else:
            raise ValueError('Rotation orientation argument not valid') 
     
# %% Kinematic System
class KinematicSystem(nx.DiGraph):
    """Kinematic reference frame graph network system"""
    node_attr_dict_factory = KinematicFrame
    edge_attr_dict_factory = KinematicTransform

    def __init__(self, incoming_graph_data = None, **attr):
        """Initialize KinematicSystem"""
        super().__init__(incoming_graph_data, **attr)
        self._path: dict[tuple[str,str], list[tuple[str,str,str]]] = {}

    # Traversal
    def get_path(self, source: str, target: str) -> list[tuple[str,str,str]]:
        """Generates transform sequence between two coordinate frames

        :param source: Source node label
        :type source: str

        :param target: Target node label
        :type target: str

        :return: Shortest path between nodes as a list of tuples describing the 
            transformations: (base, follower, orientation)
        :rtype: list[tuple[str,str,str]]
        """
        if (source, target) in self._path.keys():
            # Path previously cached
            return self._path[(source, target)]
        elif (target, source) in self._path.keys():
            # Reverse path previously cached
            self._path[(source, target)] = []
            for e in reversed(self._path[(target, source)]):
                o = 'r' if e[2] in ['f', 'forward'] else 'f'
                self._path[(source, target)].append((e[0], e[1], o))
    
            return self._path[(source, target)]
       
        # Path not previously cached
        nodes = nx.shortest_path(self.to_undirected(as_view=True), source, target)
        self._path[(source,target)] = self.path(nodes)
    
        return self._path[(source,target)]
    
    def path(self, nodes: list[str], search: bool = False) -> list[tuple[str,str,str]]:
        """Generates transform sequence from list of nodes
        
        :param nodes: Sequence of node labels that define a path
        :type nodes: list[str]

        :param search: Allow node list to be non-adjacent, defaults to False
        :type search: bool, optional

        :raises KeyError: If `search` is `False` and node list is non-adjacent

        :return: Path from node list as a list of tuples describing the 
            transformations: (base, follower, orientation)
        :rtype: list[tuple[str,str,str]]
        """
        path = []
        for j in range(len(nodes)-1):
            if (nodes[j], nodes[j+1]) in self.edges:
                path.append((nodes[j], nodes[j+1], 'forward'))
            elif (nodes[j+1], nodes[j]) in self.edges:
                path.append((nodes[j+1], nodes[j], 'reverse'))
            else:
                if search:
                    path += self.get_path(nodes[j], nodes[j+1])
                else:
                    raise KeyError("Edge ({},{}) is not present".format(nodes[j], nodes[j+1]))

        return path

    def loops(self, subgraph: list[str] = None) -> list:                                            # TODO: check return type
        """Generates a directed minimum loop basis for a subgraph

        :param subgraph: List of node labels comprising a subgraph, defaults to None
        :type subgraph: list[str], optional

        :return: List of directed edges 
        :rtype: list[tuple[str, str]]
        """
        if subgraph is None: 
            goi = self
        else:
            goi = self.subgraph(subgraph)

        mcb = nx.minimum_cycle_basis(goi.to_undirected(as_view=True))              

        loop = []
        for c in mcb:
            loop.append(nx.find_cycle(goi.subgraph(c), orientation='ignore'))                       
        
        return loop

    def position(self, datum: str | np.ndarray, nodes: str | list[str], search: bool = False) -> np.ndarray: 
        """Reports point of interest position through path of frames

        :param datum: Datum label or coordinate 
        :type datum: str | numpy.ndarray

        :param nodes: Target node or path to target node
        :type nodes: str | list[str]

        :param search: Search for path between source and target frame, defaults to False
        :type search: bool, optional

        :return: Datum position coordinates in target frame
        :rtype: numpy.ndarray
        """
        if isinstance(datum, str):
            node = nodes if isinstance(nodes, str) else nodes[0]
            point = self.nodes[node].datum[datum].position
        else:
            point = datum

        if isinstance(nodes, str):
            return point
        
        if search: 
            path = self.get_path(nodes[0], nodes[-1])
        else:
            path = self.path(nodes)

        for (frame_A, frame_B, orientation) in path:
            point = self.edges[frame_A, frame_B].transform(point, orientation)

        return point

    def direction(self, datum: str | np.ndarray, nodes: str | list[str], search: bool = False) -> np.ndarray:    
        """Reports direction through path of frames
        
        :param datum: Datum label or coordinate 
        :type datum: str | numpy.ndarray

        :param nodes: Target node or path to target node
        :type nodes: str | list[str]

        :param search: Search for path between source and target frame, defaults to False
        :type search: bool, optional

        :return: Datum direction coordinates in target frame
        :rtype: numpy.ndarray
        """   
        if isinstance(datum, str):
            node = nodes if isinstance(nodes, str) else nodes[0]
            direction = self.nodes[node].datum[datum].position
        else:
            direction = datum

        direction = direction / np.linalg.norm(direction)

        if isinstance(nodes, str):
            return direction
        
        if search: 
            path = self.get_path(nodes[0], nodes[-1])
        else:
            path = self.path(nodes)

        for (frame_A, frame_B, orientation) in path:
            direction = self.edges[frame_A, frame_B].rotate(direction, orientation)

        return direction
    
    def plot(self, ax: plt.Axes | None = None, frame: str | None = None, size: int = 1):
        """3D plot of KinematicSystem

        :param ax: Plotting axes, defaults to current axes
        :type ax: matplotlib.pyplot.Axes | None, optional

        :param frame: Reference frame to plot in, defaults to first node in graph
        :type frame: str | None, optional

        :param size: Quiver size, defaults to 1
        :type size: int, optional
        """
        # Default parameters
        ax = plt.gca() if ax is None else ax
        frame = list(self.nodes)[0] if frame is None else frame

        # Plot Cartesian frames
        for node in self.nodes():
            O = self.position('O', [node, frame], search=True)
            for i in range(3):
                ax.quiver(*O, *(self.position(f"E{i}", [node, frame], search=True)-O),                          
                    color=self.nodes[node].color, length=size, normalize=True)

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')                                                                     
        ax.set_aspect('equal')

    def front_view_plot(self, 
                        ax: plt.Axes | None= None, 
                        frame: str | None = None):
        """Plot cardinal front view of KinematicSystem

        :param ax: Plotting axes, defaults to current axes
        :type ax: matplotlib.pyplot.Axes | None, optional

        :param frame: Reference frame to plot in, defaults to first node in graph
        :type frame: str | None, optional
        """
        # Default parameters
        ax = plt.gca() if ax is None else ax
        frame = list(self.nodes)[0] if frame is None else frame
        
        # Plot Cartesian frames
        plt.sca(ax)

        i = [1,2]
        for node in self.nodes():
            O = self.position('O', [node, frame], search=True)
            for ii in i:
                E = self.position(f"E{ii}", [node, frame], search=True) - O
                ax.quiver(*O[i], *E[i], color=self.nodes[node].color)

        ax.set_title('Front View')
        ax.set_xlabel('Y [mm]')
        ax.set_ylabel('Z [mm]')                                                                
        ax.set_aspect('equal')

    # def three_view_plot(self, fig: plt.Figure | None= None, frame: str | None = None):
    #     """Plot three cardinal views of KinematicSystem"""
    #     # Default parameters
    #     fig = plt.figure() if fig is None else fig
    #     frame = list(self.nodes)[0] if frame is None else frame
    #
    #     axs = [fig.add_subplot(3,1,j+1) for j in range(3)]
    #     axs[1].invert_xaxis()
    #   
    #     # Plot Cartesian frames
    #     title_map = {0: 'Top', 1: 'Profile', 2: 'Front'}
    #     label_map = {0: 'X', 1: 'Y', 2: 'Z'}
    #     for j, ax in enumerate(axs):
    #         plt.sca(ax)
    #
    #         i = [jj for jj in range(3) if jj != 2-j]
    #         for node in self.nodes():
    #             O = self.position('O', node, frame)
    #             for ii in i:
    #                 E = self.position(f"E{ii}", node, frame) - O
    #                 ax.quiver(*O[i], *E[i], color=self.nodes[node]['color'])
    #
    #         ax.set_title(f"{title_map[j]} View")
    #         ax.set_xlabel(f"{label_map[i[0]]} [mm]")
    #         ax.set_ylabel(f"{label_map[i[1]]} [mm]")                                                                
    #         ax.set_aspect('equal')
    #
    #     fig.tight_layout()