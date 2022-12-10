"""double_wishbone.py - Double Wishbone Linkage"""

import os, sys; 

sys.path.insert(0, os.path.realpath(''))
from suspension_designer.kinematics import KinematicSystem


class DoubleWishbone(KinematicSystem):
    def __init__(self, options: dict, **kwargs):
        # Network Generation
        super().__init__()

        self.add_edges_from([
            ('E', 'I'), 
            ('I','T'), ('T','W'),
            ('I','B'), ('B','X'),
            ('X','LA'), ('X','UA'), ('X','TA'),
            ('LA','LB'), ('UA','UB'), ('TA','TB'),
            ('LB','W'), ('UB','W'), ('TB','W')])

        # Allocate Options

    def design(self):
        pass