"""double_wishbone.py - Double Wishbone Linkage"""
from copy import deepcopy

import numpy.typing as npt
import numpy as np

from scipy.optimize import fsolve, minimize
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

from suspension_designer.kinematics import datum_point_factory, KinematicSystem
from suspension_designer.geometry   import lerp, Line, Plane, \
                                           vector_alignment_rotation

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
        
    # Set static configuraiton
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
        LB = self.position('O', ['LB','W','T'])
        UB = self.position('O', ['UB','W','T'])
        T  = self.position('O', 'T')

        KP = Line(LB, UB)

        return (KP(0,2) - T)[[0,1]] 
    
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
        pTB = self.position('O', ['TB','W','T','I','B','X','TB'])
        return np.arccos(pTB[1] / np.linalg.norm(pTB))
    
    def steering_leverage(self):
        raise NotImplementedError
    
    # Alignment
    def align_inboard_joint(self, key: str):
        """Aligns inboard joints such that the outboard ball joint falls
        in the direction of :math:`e_1 = (0,1,0)`.
        
        :param key: Linkage member key, one of 'L', 'U', or 'T' representing
            lower A-arm, upper A-arm, and tie rod
        :type key: str
        """
        fA, fB = f'{key}A', f'{key}B'

        if key == 'T':
            dB = self.position('O', [fB,'W','T','I','B','X',fA])
            self.edges['X',fA].rotation[2] = np.arctan2(dB[0], dB[1]) * 180/np.pi

        dB = self.position('O', [fB,'W','T','I','B','X',fA])
        self.edges['X',fA].rotation[0] = -np.arctan2(dB[2], dB[1]) * 180/np.pi

    def align_outboard_joint(self, key: str):
        """Aligns outboard ball joints to match the normal of the 
        
        :param key: Linkage member key, one of 'L', 'U', or 'T' representing
            lower A-arm, upper A-arm, and tie rod
        :type key: str
        """
        fA, fB = f'{key}A', f'{key}B'

        dA = self.direction('E2', [fA,'X','B','I','T','W',fB])
        self.edges['W',fB].rotation[1] = -np.arctan2(dA[0], dA[2]) * 180/np.pi

        dA = self.direction('E2', [fA,'X','B','I','T','W',fB])
        self.edges['W',fB].rotation[0] = np.arctan2(dA[1], dA[2]) * 180/ np.pi

    def align_outboard_joints(self):
        self.align_outboard_joint('L')
        self.align_outboard_joint('U')
        self.align_outboard_joint('T')

    # Loop solvers
    def solve_a_arm_loop(self):
        """Positions upper A-arm to rectify the kinematic loop:
        X -> LA -> LB -> W -> UB -> UA -> X
        """
        # Compute static variables
        pLB = self.position('O', ['LB','LA'])
        lW  = np.linalg.norm(self.position('O', ['UB','W','LB']))
        
        # Loop length residual minimization
        def residual(x: float) -> float:
            self.edges['X','UA'].rotation[0] = x                                
            pUB = self.position('O', ['UB','UA','X','LA'])

            return (np.linalg.norm(pUB - pLB) - lW)**2

        sol = minimize(residual, x0=self.edges['X','UA'].rotation[0])
        if not sol['success']:
            raise RuntimeError(sol['message'])

        self.align_outboard_joint('L')
        self.align_outboard_joint('U')

    def solve_tie_rod_loop(self):
        """Positions hub to rectify the kinematic loop:
        X -> LA -> LB -> W -> TB -> TA -> X
        """
        # Kingpin alignment
        vKP = self.position('O', ['UB','UA','X','LA','LB'])
        dKP = vKP / np.linalg.norm(vKP)

        vKP_W = self.position('O', ['UB','W','LB'])
        dKP_W = vKP_W / np.linalg.norm(vKP_W)

        R0 = vector_alignment_rotation(dKP_W, dKP)
    
        # Compute static variables
        pTA = self.position('O', ['TA','X','LA','LB'])
        pTB = R0.apply(self.position('O', ['TB','W','LB']))

        lT = np.linalg.norm(self.position('O', ['TB','TA']))
        
        # Loop length residual minimization
        def residual(x: np.ndarray) -> float:
            R1 = Rotation.from_rotvec(x * dKP, degrees=True)
            p = R1.apply(pTB)

            return (np.linalg.norm(p - pTA) - lT)**2

        sol = minimize(residual, x0=0)
        if not sol['success']:
            raise RuntimeError(sol['message'])
        
        # Position linkage
        R1 = Rotation.from_rotvec(sol['x'] * dKP, degrees=True)
        R  = Rotation.from_matrix(R1.as_matrix() @ R0.as_matrix())

        vT = R.apply(pTB) - pTA

        self.edges['X','TA']['rotation'][:] = 0

        raise NotImplementedError
    
        dT = self.direction_path(vT, ['LB','LA','X','TA'])
        self.edges['X','TA']['rotation'][2] = np.arctan(dT[0]/dT[1])

        dT = self.direction_path(vT, ['LB','LA','X','TA'])
        self.edges['X','TA']['rotation'][0] = -np.arctan(dT[2]/dT[1])

        # Align wheel and tire
        R_wheel = Rotation.from_matrix(R_KP.as_matrix() @ R.as_matrix())
        a_wheel = R_wheel.as_euler('zyx')

        self.edges['I','T']['rotation'][2] = a_wheel[0]
        self.edges['I','T']['rotation'][0] = a_wheel[1]
        self.edges['T','W']['rotation'][1] = a_wheel[2]

        # Position wheel
        pLB = self.position('O', ['LB','LA','X','B','I'])
        pUB = self.position('O', ['UB','UA','X','B','I'])
        pTB = self.position('O', ['TB','TA','X','B','I'])

        vUB = pUB - pLB
        vTB = pTB - pLB

        dUB = self.direction_path(vUB, ['I','T','W'])
        dTB = self.direction_path(vTB, ['I','T','W'])

        pLB_w = self.position('O', ['LB','W'])
        pUB_w = self.position('O', ['UB','W'])
        pTB_w = self.position('O', ['TB','W'])

        vUB_w = pUB_w - pLB_w
        vTB_w = pTB_w - pLB_w

        dUB_w = vUB_w / np.linalg.norm(vUB_w)
        dTB_w = vTB_w / np.linalg.norm(vTB_w)

        _pause = True
 
    def place_wheel_frame(self):
        raise NotImplementedError
    
    def solve_axle_kinematics(self):
        """Solves axle sweep configuration"""
        self.solve_a_arm_loop()
        self.solve_tie_rod_loop()
        self.place_wheel_frame()

    # Axle sweeps
    def jounce_sweep(self, I: tuple[float, float] = (-30, 30), n: int = 11):
        """Sweeps jounce by manipulating inboard lower A-arm joint angle"""
        # Compute lower A-arm joint angles
        dz = np.linspace(I[0], I[0], n)
        
        a0 = self.graph['static'].edges['X','LA']['rotation'][0]
        l  = self.graph['static'].edges['LA','LB'].position[1]
        da  = np.arcsin((dz + l*np.sin(a0))/l)

        # Loop configurations
        for a in da:
            self.edges['X','LA']['rotation'][0] = a
            self.solve_axle_kinematics()

    def steer_sweep(self, I: tuple[float, float] = (-30, 30), n: int = 11):
        """Sweeps rack displacement to study steering behaviors"""
        raise NotImplementedError

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
        self.linkage.edges['I','T'].rotation[:] = [self.target['toe'], 0, self.target['camber']]

    def _place_wheel_frame(self):
        """Place wheel frame in desired static position based on targets"""
        self.linkage.edges['T','W'].position[2] = self.vehicle['loaded_radius']
        self.linkage.edges['T','W'].rotation[:] = [0, self.target['caster'], 0]

    def _place_body_frame(self):
        """Place body frame in desired static position based on targets"""
        self.linkage.edges['I','B'].position[2] = self.vehicle['cg_height']
        self.linkage.edges['I','B'].rotation[:] = [0, -self.vehicle['rake'], 0]

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
        self.linkage.align_outboard_joints()            # rotate outboard ball joints

        self.linkage.set_static_config()                # set design configuration

    def _sample_design(self, design: npt.ArrayLike):
        """Sample design bounds
        
        :param design: Design space for double wishbone linkage
            [0, 1,2]: x,y,z coordinates of outboard lower A-arm
            [3,   4]: x,  z coordinates of outboard upper A-arm
            [5, 6,7]: x,y,z coordinates of outboard tie rod 
            [     8]:   y   coordinate  of inboard  lower A-arm
            [9,10  ]: x,y   coordinates of inboard  tie rod  
        :type design: numpy.ndarray
        """
        def _lerp_bound(fB: str, fF: str, j: int, t: float):
            """Linearly interpolates a joint coordinate bound

            :param fB: Base frame label
            :type fB: str

            :param fF: Follower frame label
            :type fF: str

            :param j: Coordinate index
            :type j: int

            :param t: Interpolation position
            :type t: float
            """
            self.linkage.edges[fB, fF].position[j] = \
                lerp(self.bound[fF][j,0], self.bound[fF][j,1], t)
        
        for j,i in zip([0,1,2], [0,1,2]): _lerp_bound('W', 'LB', j, design[i])
        for j,i in zip([0,2]  , [3,4]  ): _lerp_bound('W', 'UB', j, design[i])
        for j,i in zip([0,1,2], [5,6,7]): _lerp_bound('W', 'TB', j, design[i])
        
        _lerp_bound('X', 'LA', 1, design[8])
        for j,i in zip([0,1], [9,10]): _lerp_bound('X', 'TA', j, design[i])

    def _place_upper_A_arm_outboard_frame(self):
        """Set upper A-arm outboard pickup lateral coordinate in the wheel frame"""
        pT  = self.linkage.position('O', ['T' ,'W'])
        pLB = self.linkage.position('O', ['LB','W'])
        pUB = self.linkage.position('O', ['UB','W'])
        
        if self.target['axle'] in ["front", "f"]:       
            # Place via kingpin inclination (KPI) target   
            yUB = pLB[1] - np.tan(np.deg2rad(self.target['kpi']))*(pUB[2] - pLB[2])  
        elif self.target['axle'] in ["rear", "r"]:      
            # Place via scrub target 
            yUB = Line(pT[1:], pLB[1:])(pUB[2], 1)[0]                                               #! TODO: verify scrub
        else:
            raise ValueError("Axle target unrecognized: {}".format(self.target['axle']))

        self.linkage.edges['W','UB'].position[1] = yUB

    def _compute_centers(self):                                                 
        """Compute roll and pitch centers and corresponding instant centers in
        the intermediate frame.
        """
        pT = self.linkage.position('O', ['T','I'])
        pW = self.linkage.position('O', ['W','T','I'])

        # Roll and pitch centers
        pRC = np.array([pT[0], 0, self.vehicle['cg_height']*(1-self.target['%_roll']/100)])
        self.linkage.nodes['I'].datum.update(datum_point_factory('RC', pRC, 'kx'))

        pPC = np.array([0, pT[1], self.vehicle['cg_height']*(1-self.target['%_pitch']/100)])
        self.linkage.nodes['I'].datum.update(datum_point_factory('PC', pPC, 'kx'))

        # Instant centers                                          
        def __instant_center(j: int, pC: np.ndarray, R: float) -> np.ndarray:
            root = lambda z: Line(pT, pC)(z, 2)[j] + np.sqrt(R**2 - (z-pW[2])**2)

            pIC = np.empty((2,), dtype=np.double)
            pIC[1] = fsolve(root, 0)
            pIC[0] = Line(pT, pC)(pIC[1], 2)[j]

            return pIC

        for [j, kC, kIC, kSA] in [[1,'RC','RIC','FVSA'], [0,'PC','PIC','SVSA']]:
            self.linkage.nodes['I'].datum.update(datum_point_factory(kIC, pT, 'ko'))

            if np.isinf(self.target[kSA]):
                self.linkage.nodes['I'].datum[kIC].position[[j,2]] = [-1e9, 0]
            
            else:
                pC = self.linkage.position(kC, 'I')
                self.linkage.nodes['I'].datum[kIC].position[[j,2]] = \
                    __instant_center(j, pC, self.target[kSA])
            
    def _place_tie_rod_inboard_frame(self):
        """Place tie rod inboard frame on plane defined by outboard pickup 
        and instant center and then align with rotation axis and outboard pickup"""
        pRIC = self.linkage.position('RIC', 'I') 
        pPIC = self.linkage.position('PIC', 'I')

        pTA = self.linkage.position('O', ['TA','X','B','I'])
        pTB = self.linkage.position('O', ['TB','W','T','I'])

        # Compute inboard tie rod height
        pTA = Plane(pRIC, pPIC, pTB)(pTA[[0,1]])
        self.linkage.edges['X','TA'].position = self.linkage.position(pTA, ['I','B','X'])
        
        # Align inboard ball joint
        self.linkage.align_inboard_joint('T')

        # Set link position
        self.linkage.edges['TA','TB'].position = \
            self.linkage.position('O', ['TB','W','T','I','B','X','TA'])

    def _place_A_arm_inboard_frames(self):
        """Place lower A-arm inboard frame on plane defined by outboard pickup 
        and instant center and then align with rotation axis and outboard pickup"""
        pRIC = self.linkage.position('RIC', 'I') 
        pPIC = self.linkage.position('PIC', 'I')
        
        for fA, fB in [['LA', 'LB'], ['UA', 'UB']]:
            # Place revolute joint
            pA = self.linkage.position('O', [fA,'X','B','I'])
            pB = self.linkage.position('O', [fB,'W','T','I'])
            
            arm_plane = Plane(pRIC, pPIC, pB) 

            if fA[0] == 'L':
                p0, p1 = arm_plane((0,pA[1])), arm_plane((pA[[0,1]]))           #! TODO: Add swing
                pickup_line = Line(p0, p1)

            elif fA[0] == 'U':
                pLA = self.linkage.position('O', ['LA','X','B','I'])
                pTA = self.linkage.position('O', ['TA','X','B','I'])
                p3  = pLA + np.array([1,0,0])                                   #! TODO: Add swing

                inboard_plane = Plane(pLA, pTA, p3)
                pickup_line = arm_plane.intersection(inboard_plane)
            
            self.linkage.edges['X',fA].position = \
                self.linkage.position(pickup_line.proj(pB), ['I','B','X'])
            
            # Align revolute joint
            dA = self.linkage.direction(pickup_line.basis, ['I','B','X',fA])
            self.linkage.edges['X',fA].rotation[2] = np.arctan2(dA[1], dA[0]) * 180/np.pi
            
            dA = self.linkage.direction(pickup_line.basis, ['I','B','X',fA])
            self.linkage.edges['X',fA].rotation[1] = np.arctan2(dA[2], dA[0]) * 180/np.pi

            self.linkage.align_inboard_joint(fA[0])
            
            # Set A-arm transform
            self.linkage.edges[fA,fB].position = \
                self.linkage.position('O', [fB,'W','T','I','B','X',fA])