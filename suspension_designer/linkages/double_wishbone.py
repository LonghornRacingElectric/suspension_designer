"""double_wishbone.py - Double Wishbone Linkage"""

# %% Imports
import os, sys; 

sys.path.insert(0, os.path.realpath(''))
from suspension_designer.kinematics import KinematicSystem


# %% Code
class DoubleWishboneBuilder(KinematicSystem):
    def __init__(self, linkage: KinematicSystem = None, **kwargs):
        # Network Generation
        if linkage is None:
            super().__init__()
        else:
            self = linkage

        self.add_node(['E','I','T','W','B','X','LA','UA','TA','LB','UB','TB'])

        self.add_edges_from([
            ('E', 'I'), 
            ('I','T'), ('T','W'),
            ('I','B'), ('B','X'),
            ('X','LA'), ('X','UA'), ('X','TA'),
            ('LA','LB'), ('UA','UB'), ('TA','TB'),
            ('LB','W'), ('UB','W'), ('TB','W')])