"""double_wishbone.py - Double Wishbone Linkage"""
import numpy as np

from scipy.optimize import fsolve

import matplotlib.pyplot as plt

import os, sys;
sys.path.insert(0, os.path.realpath(''))

from suspension_designer.kinematics import poi_factory, KinematicSystem
from suspension_designer.geometry   import lerp, Line, Plane, alignment_sequence

__all__ = ['DoubleWishbone', 'DoubleWishboneBuilder']

class DoubleWishbone(KinematicSystem):
    def __init__(self, data: dict | None = None, **kwargs):
        # Initialize KinematicSystem
        super().__init__(**kwargs)

        self.add_edges_from([
            ('I','T'),   ('T','W'),
            ('I','B'),   ('B','X'),
            ('X','LA'),  ('X','UA'),  ('X','TA'),
            ('LA','LB'), ('UA','UB'), ('TA','TB'),      # Added later
            ('W','LB'),  ('W','UB'),  ('W','TB')])

        # Assign degrees of freedom
        self.edges[('I','T')]['dof'][:] = [1,1,0,1,0,1]
        self.edges[('T','W')]['dof'][:] = [0,0,0,0,1,0]

        self.edges[('X','LA')]['dof'][:] = [0,0,0,1,0,0]
        self.edges[('X','UA')]['dof'][:] = [0,0,0,1,0,0]
        self.edges[('X','TA')]['dof'][:] = [0,0,0,1,0,1]

        self.edges[('W','LB')]['dof'][:] = [0,0,0,1,1,1]
        self.edges[('W','UB')]['dof'][:] = [0,0,0,1,1,1]
        self.edges[('W','TB')]['dof'][:] = [0,0,0,1,1,1]

        self.compute_weights()

        # Style frames
        for frame in ['LA', 'UA', 'TA']:
            self.nodes[frame]['color'] = 'r'

        for frame in ['LB', 'UB', 'TB']:
            self.nodes[frame]['color'] = 'b'

        for frame in ['T', 'W']:
            self.nodes[frame]['color'] = 'g' 

        # Assign positional data
        if data is not None:
            raise NotImplementedError
        
    def plot(self, ax: plt.axes=None):
        """Plot DoubleWishbone system"""
        # Default parameters
        ax = plt.gca() if ax is None else ax

        # Invoke KinematicSystem plot method
        super().plot(ax, 'I', 50)

        # Plot kingpin axis
        ax.plot(*np.vstack([self.coord('O','LB','I'), self.coord('O','UB','I')]).T, 'k--')

        # Plot Centers
        ax.plot(*self.coord('RC','I','I'), 'kx')
        ax.plot(*self.coord('PC','I','I'), 'kx')

        ax.plot(*np.vstack([self.coord('O','T','I'), self.coord('RIC','I','I')]).T, 'k:')
        ax.plot(*np.vstack([self.coord('O','T','I'), self.coord('PIC','I','I')]).T, 'k:')

        # Configure axes
        ax.view_init(elev=20, azim=30)
        ax.set_zlim([0, 500])

class DoubleWishboneBuilder():
    """DoubleWishbone linkage builder"""
    def __init__(self, vehicle: dict, target: dict, bound: dict):
        # Allocate options
        self.vehicle = vehicle
        self.target  = target
        self.bound   = bound

        self.linkage = DoubleWishbone()
        self.init_linkage()

    def init_linkage(self):
        """Design independent construction steps"""
        self.compute_axle_position()
        self.compute_swing_arm_lenths()

        self.place_tire_frame()
        self.place_wheel_frame()
        self.place_body_frame()
        self.place_axle_frame()

    def compute_axle_position(self):
        """Compute longitudinal axle position"""
        if self.target['axle'] in ['front', 'f']:
            self.target['position'] = 1-self.vehicle['%_front']/100
        elif self.target['axle'] in ['rear', 'r']:
            self.target['position'] = self.vehicle['%_front']/100
        else:
            raise ValueError("Axle target unrecognized: {}".format(self.target['axle']))
        
    def compute_swing_arm_lenths(self):
        """Compute swing arm target lengths"""
        for (gain, swing_arm) in [['camber_gain','FVSA'], ['caster_gain','SVSA']]:
            if self.target[gain] == 0: 
                self.target[swing_arm] = np.inf
                continue

            self.target[swing_arm] = 1/np.tan(np.deg2rad(np.abs(self.target[gain]))) 

    def place_tire_frame(self):
        """Place tire frame in desired static position based on targets"""
        self.linkage.edges['I','T']['position'][0] = self.vehicle['wheelbase']*self.target['position']
        self.linkage.edges['I','T']['position'][1] = self.target['track']/2
        self.linkage.edges['I','T']['rotation'][0] = np.deg2rad(self.target['camber'])
        self.linkage.edges['I','T']['rotation'][2] = np.deg2rad(self.target['toe'])

    def place_wheel_frame(self):
        """Place wheel frame in desired static position based on targets"""
        self.linkage.edges['T','W']['position'][2] = self.vehicle['tire_radius']
        self.linkage.edges['T','W']['rotation'][1] = np.deg2rad(self.target['caster'])

    def place_body_frame(self):
        """Place body frame in desired static position based on targets"""
        self.linkage.edges['I','B']['position'][2] = self.vehicle['cg_height']
        self.linkage.edges['I','B']['rotation'][1] = -np.deg2rad(self.vehicle['rake'])

    def place_axle_frame(self):
        """Place axle frame in desired static position based on targets"""
        self.linkage.edges['B','X']['position'][0] = self.vehicle['wheelbase']*self.target['position']
        self.linkage.edges['B','X']['position'][2] = self.vehicle['ride'] - self.vehicle['cg_height']

    def design_linkage(self, design: np.ndarray = np.full((11,), 0.5)):
        """Design dependent construction steps"""
        self.sample_design(design)

        self.place_upper_A_arm_outboard_frame()         # Meet kingpin settings
        self.compute_centers()                          # Compute (instant) centers
        self.place_tie_rod_inboard_frame()              # Set up tie-rod 
        self.place_A_arm_inboard_frames()               # Set up A-arms

    def sample_design(self, design: np.ndarray = np.full((11,), 0.5)):
        """Sample design bounds
        
        :param design: Design space for double wishbone linkage
            [0, 1,2]: x,y,z coordinates of outboard lower A-arm
            [3,   4]: x,  z coordinates of outboard upper A-arm
            [5, 6,7]: x,y,z coordinates of outboard tie rod 
            [     8]:   y   coordinate  of inboard  lower A-arm
            [9,10  ]: x,y   coordinates of inboard  tie rod  
        :type design: numpy.ndarray
        """
        def lerp_bound(fB: str, fF: str, j: int, t: float):
            self.linkage.edges[fB, fF]['position'][j] = \
                lerp(self.bound[fF][j,0], self.bound[fF][j,1], t)
        
        for j, i in zip([0,1,2], [0,1,2]):
            lerp_bound('W', 'LB', j, design[i])
        
        for j,i in zip([0,2], [3,4]):
            lerp_bound('W', 'UB', j, design[i])

        for j,i in zip([0,1,2], [5,6,7]): 
            lerp_bound('W', 'TB', j, design[i])
        
        lerp_bound('X', 'LA', 1, design[8])

        for j,i in zip([0,1], [9,10]): 
            lerp_bound('X', 'TA', j, design[i])

    def place_upper_A_arm_outboard_frame(self):
        """Set upper A-arm outboard pickup lateral coordinate. All computation 
        in wheel frame"""
        T  = self.linkage.coord('O','T','W')
        LB = self.linkage.coord('O','LB','W')
        UB = self.linkage.coord('O','UB','W')
        
        if self.target['axle'] in ["front", "f"]:
            # Place via KPI target    
            self.linkage.edges['W','UB']['position'][1] = LB[1] - \
                np.tan(np.deg2rad(self.target['kpi']))*(UB[2] - LB[2])  
        elif self.target['axle'] in ["rear", "r"]:
            # Place via scrub target            
            self.linkage.edges['W','UB']['position'][1] = \
                Line(T[1:], LB[1:])(UB[2], 1)[0]
        else:
            raise ValueError("Axle target unrecognized: {}".format(self.target['axle']))

    def compute_centers(self):
        """Compute roll and pitch centers and corresponding instant centers.
        All computation in intermediate frame"""
        T = self.linkage.coord('O','T','I')
        W = self.linkage.coord('O','W','I')

        # Roll and pitch centers
        RC = np.array([T[0], 0, self.vehicle['cg_height']*(1-self.target['%_roll']/100)])
        self.linkage.nodes['I']['poi'].update(poi_factory('RC', RC, 'kx'))

        PC = np.array([0, T[1], self.vehicle['cg_height']*(1-self.target['%_pitch']/100)])
        self.linkage.nodes['I']['poi'].update(poi_factory('PC', PC, 'kx'))

        # Instant centers
        z_root = lambda p0, p1, pc, R, z: \
            ((p1[0] - p0[0])/(p1[1] - p0[1]))*(z - p0[1]) + np.sqrt(R**2 - (z-pc[1])**2)

        for [j, c, ic, arm] in [[1,'RC','RIC','FVSA'], [0,'PC','PIC','SVSA']]:
            self.linkage.nodes['I']['poi'].update(poi_factory(ic, T, 'ko'))

            if np.isinf(self.target[arm]):
                self.linkage.nodes['I']['poi'][ic]['position'][[j,2]] = [-1e9, 0]
                continue

            C = self.linkage.coord(c,'I')
            self.linkage.nodes['I']['poi'][ic]['position'][2] = fsolve( 
                lambda z: z_root(T[[j,2]], C[[j,2]], W[[j,2]], self.target[arm], z), 0)

            self.linkage.nodes['I']['poi'][ic]['position'][j] = \
                Line(T,C)(self.linkage.coord(ic,'I')[2], 2)[j]
            
    def place_tie_rod_inboard_frame(self):
        """Place tie rod inboard frame on plane defined by outboard pickup 
        and instant center and then align with rotation axis and outboard pickup"""
        RIC = self.linkage.coord('RIC', 'I') 
        PIC = self.linkage.coord('PIC', 'I')

        # Update tie rod inboard height
        pA = self.linkage.coord('O', 'TA', 'I')
        pB = self.linkage.coord('O', 'TB', 'I')

        pA = Plane(RIC, PIC, pB)(pA[[0,1]])
        self.linkage.edges['X','TA']['position'] = self.linkage.coord(pA, 'I', 'X')
        
        # Align ball joint
        axis = self.linkage.direction(pB - pA, 'I', 'TA')  
        self.linkage.edges['X','TA']['rotation'][2] = np.arctan(axis[0]/axis[1])
        
        axis = self.linkage.direction(pB - pA, 'I', 'TA') 
        self.linkage.edges['X','TA']['rotation'][0] = np.arctan(axis[2]/axis[1])

        # Set A-arm transform
        self.linkage.edges['TA','TB']['position'] = \
            self.linkage.coord(self.linkage.coord('O', 'TB', 'I'), 'I', 'TA')

    def place_A_arm_inboard_frames(self):
        """Place lower A-arm inboard frame on plane defined by outboard pickup 
        and instant center and then align with rotation axis and outboard pickup"""
        RIC = self.linkage.coord('RIC', 'I') 
        PIC = self.linkage.coord('PIC', 'I')

        
        for A,B in [['LA', 'LB'], ['UA', 'UB']]:
            # Place revolute joint
            pA = self.linkage.coord('O', A, 'I')
            pB = self.linkage.coord('O', B, 'I')
            
            arm_plane = Plane(RIC, PIC, pB) 

            if A[0] == 'L':
                p0, p1 = arm_plane((0,pA[1])), arm_plane((pA[[0,1]]))           #! TODO: Add swing

            elif A[0] == 'U':
                pLA = self.linkage.coord('O', 'LA', 'I')
                pTA = self.linkage.coord('O', 'TA', 'I')
                p3  = pLA + np.array([1,0,0])                                   #! TODO: Add swing

                inboard_plane = Plane(pLA, pTA, p3)
                pickup_line = arm_plane.intersection(inboard_plane)

                p0, p1 = pickup_line.p[0], pickup_line.p[1]
            
            axis = (p1-p0) / np.linalg.norm(p1-p0)
            self.linkage.edges['X',A]['position'] = \
                self.linkage.coord(p0 + np.dot(pB-p0, axis) * axis, 'I', 'X')
            
            # Align revolute joint
            self.linkage.edges['X',A]['rotation'] = alignment_sequence(
                self.linkage.direction(axis, 'I', A), 
                self.linkage.direction('E0', A),
                self.linkage.edges['X',A]['sequence']) 
            
            dB = self.linkage.coord(self.linkage.coord('O', B, 'I'), 'I', A)
            self.linkage.edges['X',A]['rotation'][0] = -np.arctan(dB[2] / dB[1])

            # Set A-arm transform
            self.linkage.edges[A,B]['position'] = \
                self.linkage.coord(self.linkage.coord('O', B, 'I'), 'I', A)