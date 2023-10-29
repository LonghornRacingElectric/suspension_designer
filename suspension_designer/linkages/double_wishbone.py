"""double_wishbone.py - Double Wishbone Linkage"""
from __future__ import annotations

from copy import deepcopy

import numpy.typing as npt
import numpy as np

from scipy.optimize import fsolve, minimize

import matplotlib.pyplot as plt

from suspension_designer.kinematics import datum_point_factory, KinematicSystem
from suspension_designer.geometry   import (
    lerp, Line, Plane, vector_alignment_angles, vector_alignment_rotation)

__all__ = ['DoubleWishbone', 'DoubleWishboneBuilder']

class DoubleWishbone(KinematicSystem):
    def __init__(self, data: dict | None = None, **kwargs):
        """Initialize KinematicSystem"""
        super().__init__(**kwargs)

        self.add_edges_from([
            ('I','T'),   ('T','W'),
            ('I','B'),   ('B','X'),
            ('X','LA'),  ('X','UA'),  ('X','TA'),
            ('LA','LB'), ('UA','UB'), ('TA','TB'),
            ('W','LB'),  ('W','UB'),  ('W','TB')])

        if data is None:
            for node in ['LA', 'UA', 'TA']:
                self.nodes[node].color = 'r'

            for node in ['LB', 'UB', 'TB']:
                self.nodes[node].color = 'b'

            return
            
        raise NotImplementedError                                               #! TODO: assign positional data  
        
    def set_static_config(self):
        """Create a deep copy of current state as a graph attribute"""
        self.static: DoubleWishbone = deepcopy(self)

    # Quantities of Interest
    def rack_displacement(self):
        """Computes lateral steering rack displacement"""
        return self.position('O', ['TA','X'])[1] \
             - self.static.position('O', ['TA','X'])[1]
    
    def jounce(self):
        """Computes jounce as the vertical displacement of the wheel in the axle frame"""
        return self.position('O', ['W','LB','LA','X'])[2] \
             - self.static.position('O', ['W','LB','LA','X'])[2]
    
    def track(self):
        """Returns track as the lateral position of the tire in the intermediate frame"""
        return self.edges['I','T'].position[1]
    
    def rake(self): 
        """Computes longitudinal displacement of the tire in the axle frame"""
        return self.position('O', ['W','LB','LA','X'])[0] \
             - self.static.position('O', ['W','LB','LA','X'])[0]
    
    def toe(self):
        """Returns toe anlge"""
        return self.edges['I','T'].rotation[2]
    
    def inclination(self):
        """Returns inclination angle between tire and intermediate frame"""
        return self.edges['I','T'].rotation[0]
    
    def camber(self):
        """Computes difference between roll and inclination angle. 
        WARNING: Typically inclination is desired, see SAE J670."""
        raise NotImplementedError
    
    def caster(self):
        raise NotImplementedError
    
    def contact_patch_lever(self):
        """Computes contact patch moment lever to kingpin axis in tire frame"""
        p_LB = self.position('O', ['LB','W','T'])
        p_UB = self.position('O', ['UB','W','T'])
        p_T  = self.position('O', 'T')

        L_KP = Line(p_LB, p_UB)

        return (L_KP(0,2) - p_T)[[0,1]] 
    
    def trail(self):
        """Returns mechanical trail"""
        return -self.contact_patch_lever()[0]
    
    def scrub(self):
        """Returns mechanical scrub"""
        return self.contact_patch_lever()[1]
    
    def tie_rod_misalignment(self) -> float:
        """Computes inboard misalignment angle of the tie rod relative to the 
        lateral steering rack. Important for bending loads onto the inboard
        steering system."""
        p_TB = self.position('O', ['TB','W','T','I','B','X','TA'])
        return np.arccos(p_TB[1] / np.linalg.norm(p_TB))
    
    def steering_leverage(self) -> np.ndarray:
        raise NotImplementedError
    
    # Alignment
    def _align_inboard_joint(self, key: str):
        """Aligns inboard joints such that the outboard ball joint falls
        in the direction of :math:`e_1 = (0,1,0)`.
        
        :param key: Linkage member key, one of 'L', 'U', or 'T' representing
            lower A-arm, upper A-arm, and tie rod
        :type key: str
        """
        f_A, f_B = f'{key}A', f'{key}B'

        if key == 'T':
            self.edges['X',f_A].rotation[[0,2]] = [0,0]
        else:
            self.edges['X',f_A].rotation[0] = 0

        d_B = self.position('O', [f_B,'W','T','I','B','X',f_A])
        if key == 'T':
            psi, phi = vector_alignment_angles(d_B, np.array([0,1,0]), 'ZX')

            self.edges['X',f_A].rotation[0] = phi * 180/np.pi
            self.edges['X',f_A].rotation[2] = psi * 180/np.pi
            return
        
        self.edges['X',f_A].rotation[0] = -np.arctan2(d_B[2], d_B[1]) * 180/np.pi

    def _align_outboard_joint(self, key: str):
        """Aligns outboard ball joints to match the normal of the ball joint
        
        :param key: Linkage member key, one of 'L', 'U', or 'T' representing
            lower A-arm, upper A-arm, and tie rod
        :type key: str
        """
        f_A, f_B = f'{key}A', f'{key}B'

        self.edges['W',f_B].rotation[:] = [0,0,0] 
        d_A = self.direction('E2', [f_A,'X','B','I','T','W',f_B])
        theta, phi = vector_alignment_angles(np.array([0,0,1]), d_A, 'YX')

        self.edges['W',f_B].rotation[0] = phi * 180/np.pi
        self.edges['W',f_B].rotation[1] = theta * 180/np.pi

    def _align_outboard_joints(self):
        """Aligns all of the outboard ball joints"""
        self._align_outboard_joint('L')
        self._align_outboard_joint('U')
        self._align_outboard_joint('T')
    
    def solve_axle_kinematics(self, jounce: float, rack_displacement: float):
        """Solves axle sweep configuration"""
        # Translate tire
        self.edges['I','T'].position[2] = jounce

        # Translate tie rod
        y_TA = self.static.edges['X','TA'].position[1] 
        self.edges['X','TA'].position[1] = y_TA + rack_displacement

        # Solve kinematic loops
        def __loop_residual(x: np.ndarray) -> float:
            # Set configuration
            self.edges['X','LA'].rotation[0] = x[0]
            self.edges['X','UA'].rotation[0] = x[1]
            self.edges['X','TA'].rotation[[0,2]] = x[[2,3]]
            self.edges['I','T'].position[[0,1]] = x[[4,5]]
            self.edges['I','T'].rotation[[0,2]] = x[[6,7]]
            self.edges['T','W'].rotation[1] = x[8]

            # Compute outboard pickup path positions
            p_LB_X = self.position('O', ['LB','LA','X','B','I'])
            p_LB_W = self.position('O', ['LB','W','T','I'])

            p_UB_X = self.position('O', ['UB','UA','X','B','I'])
            p_UB_W = self.position('O', ['UB','W','T','I'])

            p_TB_X = self.position('O', ['TB','TA','X','B','I'])
            p_TB_W = self.position('O', ['TB','W','T','I'])

            # Compute stacked loop residual
            p_X = np.concatenate([p_LB_X, p_UB_X, p_TB_X])
            p_W = np.concatenate([p_LB_W, p_UB_W, p_TB_W])

            return np.linalg.norm(p_X - p_W)
        
        sol = minimize(__loop_residual, x0=np.hstack([
            self.edges['X','LA'].rotation[0],
            self.edges['X','UA'].rotation[0],
            *[self.edges['X','TA'].rotation[j] for j in [0,2]],
            self.edges['I','T'].position[[0,1]],
            *[self.edges['I','T'].rotation[j] for j in [0,2]],
            self.edges['T','W'].rotation[1]]))
        
        if not sol['success']:
            if sol['message'] == 'Desired error not necessarily achieved due to precision loss.':
                # Compute outboard pickup path positions
                p_LB_X = self.position('O', ['LB','LA','X','B','I'])
                p_LB_W = self.position('O', ['LB','W','T','I'])

                p_UB_X = self.position('O', ['UB','UA','X','B','I'])
                p_UB_W = self.position('O', ['UB','W','T','I'])

                p_TB_X = self.position('O', ['TB','TA','X','B','I'])
                p_TB_W = self.position('O', ['TB','W','T','I'])

                # Compute stacked loop residual
                p_X = np.concatenate([p_LB_X, p_UB_X, p_TB_X])
                p_W = np.concatenate([p_LB_W, p_UB_W, p_TB_W])
            
                if np.max(np.abs(p_X - p_W)) < 1e-4:
                    return

            raise RuntimeError(sol['message'])
        
        # Align outboard joints
        self._align_outboard_joints()

    # Axle sweeps
    def jounce_sweep(self, limits: tuple[float, float] = (-30, 30), n: int = 7) \
            -> dict[float, DoubleWishbone]:
        """Sweeps jounce by manipulating inboard lower A-arm joint angle"""
        sweep: dict[float, DoubleWishbone] = {}
        for jounce in np.linspace(*limits, n):
            self.solve_axle_kinematics(jounce, 0)
            sweep[jounce] = deepcopy(self)

        return sweep

    def steer_sweep(self, limits: tuple[float, float] = (-30, 30), n: int = 7,
                    jounce: float = 0) -> dict[float, DoubleWishbone]:
        """Sweeps rack displacement to study steering behaviors"""
        # Solve neutral steer
        self.solve_axle_kinematics(jounce, 0)
        sweep = {0: deepcopy(self)}

        # Solve left steers
        n_half = int(np.ceil(n/2))
        for rack_displacement in np.linspace(0, limits[1], n_half)[1:]:
            self.solve_axle_kinematics(jounce, rack_displacement)
            sweep[rack_displacement] = deepcopy(self)

        # Solve right steers
        self = deepcopy(sweep[0])
        for rack_displacement in np.linspace(0, limits[0], n_half)[1:]:
            self.solve_axle_kinematics(jounce, rack_displacement)
            sweep[rack_displacement] = deepcopy(self)
            
        return sweep

    def motion_sweep(self,
                     jounce_limits: tuple[float, float] = (-30, 30),
                     rack_limits: tuple[float, float] = (-30, 30),
                     n: tuple(int, int) = (7,7)) \
                    -> dict[(float, float), DoubleWishbone]:
        sweep: dict[float, dict[float, DoubleWishbone]] = {}
        for jounce in np.linspace(*jounce_limits, n[0]):
            sweep[jounce] = self.steer_sweep(rack_limits, n[1], jounce)
    
        # Reformat sweep
        resweep = {}
        for jounce, steer_sweep in sweep.items():
            for rack_displacement, result in steer_sweep.items():
                resweep[(jounce, rack_displacement)] = result
        
        return resweep
    
    # Plotting
    def plot(self, ax: plt.axes = None):
        """Plot DoubleWishbone system"""
        # Default parameters
        ax = plt.gca() if ax is None else ax

        # Invoke KinematicSystem plot method
        super().plot(ax, 'I', 50)

        # Plot linkages
        p: dict[str, np.ndarray] = {}
        for n in ['LA', 'LB', 'UA', 'UB', 'TA', 'TB']:
            p[n] = self.position('O', [n, 'I'], search=True)

        ax.plot(*np.vstack([p['LA'], p['LB']]).T, 'k')
        ax.plot(*np.vstack([p['UA'], p['UB']]).T, 'k')
        ax.plot(*np.vstack([p['TA'], p['TB']]).T, 'k')

        # Plot kingpin axis
        ax.plot(*np.vstack([p['LB'], p['UB']]).T, 'k--')

        # Plot steer arm
        KP = Line(p['LB'], p['UB'])
        ax.plot(*np.vstack([p['TB'], KP.proj(p['TB'])]).T, 'k--')

        # Plot centers
        T = self.position('O', ['T','I'], search=True)

        ax.plot(*self.position('RC', 'I'), 'kx')
        ax.plot(*self.position('PC', 'I'), 'kx')

        ax.plot(*self.position('RIC', 'I'), 'ko')
        ax.plot(*self.position('PIC', 'I'), 'ko')

        ax.plot(*np.vstack([T, self.position('RIC','I')]).T, 'k:')
        ax.plot(*np.vstack([T, self.position('PIC','I')]).T, 'k:')

        # Configure axes
        ax.view_init(elev=20, azim=30)
        ax.set_zlim([0, 500])

    def front_view_plot(self, ax: plt.axes = None):
        """Plot cardinal front view of DoubleWishbone system"""
        # Default parameters
        ax = plt.gca() if ax is None else ax

        # Invoke KinematicSystem front_view_plot()
        super().front_view_plot(ax, 'I')

        # Plot ground
        ax.plot([-100, 1000], [0, 0], 'k--')
        # Compute locations
        p = {}
        for k in ['T', 'LA', 'LB', 'UA', 'UB', 'TA', 'TB']:
            p[k] = self.position('O', [k,'I'], search=True)[[1,2]]
            ax.plot(*p[k], 'ko')

        for k in ['RC', 'RIC']:
            p[k] = self.position(k,'I')[[1,2]]
            ax.plot(*p[k], 'kx')

        # Plot links
        ax.plot(*np.vstack([p['LA'], p['LB']]).T, 'k')
        ax.plot(*np.vstack([p['UA'], p['UB']]).T, 'k')
        ax.plot(*np.vstack([p['TA'], p['TB']]).T, 'k')

        # Plot kingpin axis
        KP = Line(p['LB'], p['UB'])
        
        ax.plot(*np.vstack([KP(0,1), p['UB']]).T, 'k-.')

        # Plot Centers
        ax.plot(*np.vstack([p['T'], p['RIC']]).T, 'k:')

class DoubleWishboneBuilder():
    """DoubleWishbone linkage builder"""
    def __init__(self, vehicle: dict, target: dict, bound: dict):
        # Allocate options
        self.vehicle = vehicle
        self.target  = target
        self.bound   = bound

        self.linkage = DoubleWishbone()
        self.init_linkage()

    # Linkage Initialization
    def init_linkage(self):
        """Design independent construction steps"""
        self._compute_axle_position()
        self._compute_swing_arm_lenths()

        self._place_tire_frame()
        self._place_wheel_frame()
        self._place_body_frame()
        self._place_axle_frame()

    def _compute_axle_position(self):
        """Compute longitudinal axle position"""
        if self.target['axle'] in ['front', 'f']:
            self.target['position'] = 1-self.vehicle['%_front']/100
        elif self.target['axle'] in ['rear', 'r']:
            self.target['position'] = self.vehicle['%_front']/100
        else:
            raise ValueError("Axle target unrecognized: {}".format(self.target['axle']))
        
    def _compute_swing_arm_lenths(self):
        """Compute swing arm target lengths"""
        for (gain, swing_arm) in [['camber_gain','FVSA'], ['caster_gain','SVSA']]:
            if self.target[gain] == 0: 
                self.target[swing_arm] = np.inf
                continue

            self.target[swing_arm] = 1/np.tan(np.deg2rad(np.abs(self.target[gain]))) 

    def _place_tire_frame(self):
        """Place tire frame in desired static position based on targets"""
        self.linkage.edges['I','T'].position[0] = self.vehicle['wheelbase']*self.target['position']
        self.linkage.edges['I','T'].position[1] = self.target['track']/2
        self.linkage.edges['I','T'].rotation[[0,2]] = [self.target['camber'], self.target['toe']]

    def _place_wheel_frame(self):
        """Place wheel frame in desired static position based on targets"""
        self.linkage.edges['T','W'].position[2] = self.vehicle['loaded_radius']
        self.linkage.edges['T','W'].rotation[1] = self.target['caster']

    def _place_body_frame(self):
        """Place body frame in desired static position based on targets"""
        self.linkage.edges['I','B'].position[2] = self.vehicle['cg_height']
        self.linkage.edges['I','B'].rotation[1] = -self.vehicle['rake']

    def _place_axle_frame(self):
        """Place axle frame in desired static position based on targets"""
        self.linkage.edges['B','X'].position[0] = self.vehicle['wheelbase']*self.target['position']
        self.linkage.edges['B','X'].position[2] = self.vehicle['ride'] - self.vehicle['cg_height']

    # Linkage Design
    def design_linkage(self, design: npt.ArrayLike | None = None):
        """Design dependent construction steps"""
        design = design if design is not None else np.full((11,), 0.5)

        self._sample_design(design)

        self._place_upper_A_arm_outboard_frame()        # meet kingpin settings
        self._compute_centers()                         # compute (instant) centers
        self._place_tie_rod_inboard_frame()             # configure tie-rod 
        self._place_A_arm_inboard_frames()              # configure A-arms

        self.linkage._align_outboard_joints()           # rotate outboard ball joints
        self.linkage.set_static_config()                # set design configuration

    def _sample_design(self, design: npt.ArrayLike):
        """Sample design bounds
        
        :param design: Design space for double wishbone linkage
            [0, 1,2]: x,y,z coordinates of outboard lower A-arm
            [3,   4]: x,  z coordinates of outboard upper A-arm
            [5, 6,7]: x,y,z coordinates of outboard tie rod 
            [   8  ]:   y   coordinate  of inboard lower A-arm
            [9,10  ]: x,y   coordinates of inboard tie rod  
        :type design: numpy.ndarray
        """
        def __lerp_bound(f_B: str, f_F: str, j: int, t: float):
            """Linearly interpolates a joint coordinate bound

            :param f_B: Base frame label
            :type f_B: str

            :param f_F: Follower frame label
            :type f_F: str

            :param j: Coordinate index
            :type j: int

            :param t: Interpolation position
            :type t: float
            """
            self.linkage.edges[f_B, f_F].position[j] = \
                lerp(self.bound[f_F][j,0], self.bound[f_F][j,1], t)
        
        for j,i in zip([0,1,2], [0,1,2]): __lerp_bound('W', 'LB', j, design[i])
        for j,i in zip([0,  2], [3,  4]): __lerp_bound('W', 'UB', j, design[i])
        for j,i in zip([0,1,2], [5,6,7]): __lerp_bound('W', 'TB', j, design[i])
        
        __lerp_bound('X', 'LA', 1, design[8])
        for j,i in zip([0,1], [9,10]): __lerp_bound('X', 'TA', j, design[i])

    def _place_upper_A_arm_outboard_frame(self):
        """Set upper A-arm outboard pickup lateral coordinate in the wheel frame"""
        p_T  = self.linkage.position('O', ['T' ,'W'])
        p_LB = self.linkage.position('O', ['LB','W'])
        p_UB = self.linkage.position('O', ['UB','W'])
        
        if self.target['axle'] in ["front", "f"]:       
            # Place via kingpin inclination (KPI) target   
            y_UB = p_LB[1] - np.tan(np.deg2rad(self.target['kpi']))*(p_UB[2] - p_LB[2])  
        elif self.target['axle'] in ["rear", "r"]:      
            # Place via scrub target 
            y_UB = Line(p_T[1:], p_LB[1:])(p_UB[2], 1)[0]                                               #! TODO: verify scrub
        else:
            raise ValueError("Axle target unrecognized: {}".format(self.target['axle']))

        self.linkage.edges['W','UB'].position[1] = y_UB

    def _compute_centers(self):                                                 
        """Compute roll and pitch centers and corresponding instant centers in
        the intermediate frame.
        """
        p_T = self.linkage.position('O', ['T','I'])
        p_W = self.linkage.position('O', ['W','T','I'])

        # Roll and pitch centers
        p_RC = np.array([p_T[0], 0, self.vehicle['cg_height']*(1-self.target['%_roll']/100)])
        self.linkage.nodes['I'].datum.update(datum_point_factory('RC', p_RC, 'kx'))

        p_PC = np.array([0, p_T[1], self.vehicle['cg_height']*(1-self.target['%_pitch']/100)])
        self.linkage.nodes['I'].datum.update(datum_point_factory('PC', p_PC, 'kx'))

        # Instant centers                                          
        def __instant_center(j: int, pC: np.ndarray, R: float) -> np.ndarray:
            root = lambda z: Line(p_T, pC)(z, 2)[j] + np.sqrt(R**2 - (z-p_W[2])**2)

            p_IC = np.empty((2,), dtype=np.double)
            p_IC[1] = fsolve(root, 0)
            p_IC[0] = Line(p_T, pC)(p_IC[1], 2)[j]

            return p_IC

        for [j, k_C, k_IC, k_SA] in [[1,'RC','RIC','FVSA'], [0,'PC','PIC','SVSA']]:
            self.linkage.nodes['I'].datum.update(datum_point_factory(k_IC, p_T, 'ko'))

            if np.isinf(self.target[k_SA]):
                self.linkage.nodes['I'].datum[k_IC].position[[j,2]] = [-1e9, 0]
            
            else:
                p_C = self.linkage.position(k_C, 'I')
                self.linkage.nodes['I'].datum[k_IC].position[[j,2]] = \
                    __instant_center(j, p_C, self.target[k_SA])
            
    def _place_tie_rod_inboard_frame(self):
        """Place tie rod inboard frame on plane defined by outboard pickup 
        and instant center and then align with rotation axis and outboard pickup"""
        p_RIC = self.linkage.position('RIC', 'I') 
        p_PIC = self.linkage.position('PIC', 'I')

        p_TA = self.linkage.position('O', ['TA','X','B','I'])
        p_TB = self.linkage.position('O', ['TB','W','T','I'])

        # Compute inboard tie rod height
        p_TA = Plane(p_RIC, p_PIC, p_TB)(p_TA[[0,1]])
        self.linkage.edges['X','TA'].position = self.linkage.position(p_TA, ['I','B','X'])
        
        # Align inboard ball joint
        self.linkage._align_inboard_joint('T')

        # Set link position
        self.linkage.edges['TA','TB'].position = \
            self.linkage.position('O', ['TB','W','T','I','B','X','TA'])

    def _place_A_arm_inboard_frames(self):
        """Places lower A-arm inboard frame. The current design procedure is 
        as follows:
        
        #. Let the instant center plane :math:`P_{IC}` be defined by the pitch
            and roll instant centers and the outboard pickup of the A-arm

        #. Define the pickup line, :math:`e_{A}` which is the line of potential
            locations for the inboard pickup:

            * For the lower A-arm, simply take a slice of the instant center
                plane at the desired lateral coordinate

            * For the upper A-arm, define an inboard pickup plane :math:`P_{A}` 
                by the inboard lower A-arm and tie rod pickups 
                (:math:`p_{LA}, p_{TA}`) and :math:`p_{LA} + e_x` to ensure the
                inboard pickup plane is perpendicular to the body :math:`YZ` plane

        #. Define the pickup axis, :math:`e_{A}'` as the projection of the 
            pickup line onto the :math:`XZ` plane:

            .. math::

                e_{A}' = \Pi_{XZ} e_{A} = \langle e_{A,x}, 0, e_{A,z} \\rangle

        #. Find the inboard pickup position, :math:`p_A \in P_{IC} \cap P_{A}`:

            .. math::

                (p_{B} - p_{A}) \cdot e_{A}' = 0

        #. Align the inboard pickup frame to align it's :math:`e_1` with :math:`e_A`

        NOTE: This does not allow for trailing arm designs.
        """
        p_RIC = self.linkage.position('RIC', 'I') 
        p_PIC = self.linkage.position('PIC', 'I')

        for f_A, f_B in [['LA', 'LB'], ['UA', 'UB']]:
            p_A = self.linkage.position('O', [f_A,'X','B','I'])
            p_B = self.linkage.position('O', [f_B,'W','T','I'])
            
            P_IC = Plane(p_RIC, p_PIC, p_B) 

            if f_A[0] == 'L':
                p_0, p_1 = P_IC((0,p_A[1])), P_IC((p_A[[0,1]]))                 
                L_A = Line(p_0, p_1)

            elif f_A[0] == 'U':
                p_LA = self.linkage.position('O', ['LA','X','B','I'])
                p_TA = self.linkage.position('O', ['TA','X','B','I'])
                p_3  = p_LA + np.array([10,0,0])                                

                P_A = Plane(p_LA, p_TA, p_3)
                L_A = P_IC.intersection(P_A)
            
            # Solve special projection problem (see docstring)
            e_A = np.copy(L_A.basis)
            e_A[1] = 0
            e_A /= np.linalg.norm(e_A)
            
            t = np.dot(p_B - L_A.point[0], e_A) / np.dot(L_A.basis, e_A) 
            p_A = L_A.point[0] + t*L_A.basis

            self.linkage.edges['X',f_A].position = self.linkage.position(p_A, ['I','B','X'])

            # Align revolute joint with axis
            d_A = self.linkage.direction(e_A, ['I','B','X',f_A])
            self.linkage.edges['X',f_A].rotation[1] = np.arctan2(d_A[2], d_A[0]) * 180/np.pi

            # Rotate revolute joint with pickup
            d_B = self.linkage.position('O', [f_B,'W','T','I','B','X',f_A])
            self.linkage.edges['X',f_A].rotation[0] = -np.arctan2(d_B[2], d_B[1]) * 180/np.pi

            # Set A-arm transform
            self.linkage.edges[f_A,f_B].position = \
                self.linkage.position('O', [f_B,'W','T','I','B','X',f_A])