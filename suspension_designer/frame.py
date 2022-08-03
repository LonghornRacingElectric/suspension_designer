"""frame.py - Example Module"""

# %% Imports
from __future__ import annotations
import numpy.typing as npt

import numpy as np
import networkx as nx


# %% Core Functions
class KinematicFrame():
    """Kinematic reference frame in 3D space"""
    def __init__(
                self,
                name: str = None,
                base: KinematicFrame = None,
                origin: npt.ArrayLike = np.zeros(3),
                orient: npt.ArrayLike = np.zeros(3),
            ):
        """Allocate KinematicFrame instance information"""
        self.name = name
        self.base = base
        self.origin = origin
        self.orient = orient


class KinematicSystem(nx.DiGraph):
    """Kinematic reference frame graph network system"""
    node_attr_dict_factory = KinematicFrame()
    edge_attr_dict_factory = KinematicTransform()

    


def main():
    sys = KinematicSystem()

    a = 1


if __name__ == "__main__":
    main()