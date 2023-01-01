"""double_wishbone.py - Double Wishbone Linkage"""
from typing import Callable
from numpy.typing import ArrayLike

import numpy as np;
from scipy.optimize import fsolve;

import matplotlib.pyplot as plt

import os, sys; 
sys.path.insert(0, os.path.realpath(''))

from suspension_designer.kinematics import poi_factory, KinematicSystem

class DoubleWishbone(KinematicSystem):
    def __init__(self, options: dict, **kwargs):
        # Network Generation
        super().__init__(**kwargs)

        self.add_edges_from([
            ('I','T'), ('T','W'),
            ('I','B'), ('B','X'),
            ('X','LA'), ('X','UA'), ('X','TA'),
            ('LA','LB'), ('UA','UB'), ('TA','TB'),
            ('W','LB'), ('W','UB'), ('W','TB')])

        # Degrees of Freedom
        self.edges[('I','T')]['dof'][:] = [1,1,0,1,0,1]
        self.edges[('T','W')]['dof'][:] = [0,0,0,0,1,0]

        self.edges[('X','LA')]['dof'][:] = [0,0,0,1,0,0]
        self.edges[('X','UA')]['dof'][:] = [0,0,0,1,0,0]
        self.edges[('X','TA')]['dof'][:] = [0,0,0,1,0,1]

        self.edges[('W','LB')]['dof'][:] = [0,0,0,1,1,1]
        self.edges[('W','UB')]['dof'][:] = [0,0,0,1,1,1]
        self.edges[('W','TB')]['dof'][:] = [0,0,0,1,1,1]

        self.compute_weights()
        
        # Allocate Options
        self.vehicle = options['vehicle']
        self.target  = options['target']
        self.bound   = options['bound']

        # Compute Additional Targets
        self.target['FVSA'] = 1/np.tan(np.deg2rad(np.abs(self.target['camber_gain'])))
        self.target['SVSA'] = 1/np.tan(np.deg2rad(np.abs(self.target['caster_gain'])))

    def design(self, design: ArrayLike=np.random.rand(9)):
        """Generate design meeting targets and bounds"""
        ### Sample Design ###
        lerp: Callable[[float,float,float],float] = lambda x0, x1, t: x0*(1-t) + x1*t
        lerpBound: Callable[[str,int,float],float] = \
            lambda p, i, t: lerp(self.bound[p][i][0], self.bound[p][i][1], t)

        self.edges['W','LB']['position'][0] = lerpBound('LB', 0, design[0])
        self.edges['W','LB']['position'][1] = lerpBound('LB', 1, design[1])
        self.edges['W','LB']['position'][2] = lerpBound('LB', 2, design[2])

        self.edges['W','UB']['position'][0] = lerpBound('UB', 0, design[3])
        self.edges['W','UB']['position'][2] = lerpBound('UB', 2, design[4])

        ### Apply Vechicle Options and Design Targets ###
        match self.target['axle']:
            case "front" | "f":
                axleSkew = 1-self.vehicle['%f']/100
            case "rear" | "r":
                axleSkew = self.vehicle['%f']/100
            case _:
                return ValueError("Axle target unrecognized: {}".format(self.target['axle']))

        self.edges['I','T']['position'][0] = self.vehicle['wheelbase']*axleSkew
        self.edges['I','T']['position'][1] = self.target['track']/2
        self.edges['I','T']['rotation'][0] = np.deg2rad(self.target['camber'])
        self.edges['I','T']['rotation'][2] = np.deg2rad(self.target['toe'])

        self.edges['T','W']['position'][2] = self.vehicle['tire_radius']
        self.edges['T','W']['rotation'][0] = np.deg2rad(self.target['caster'])

        self.edges['I','B']['position'][2] = self.vehicle['cg_height']
        self.edges['I','B']['rotation'][1] = self.vehicle['rake']*(np.pi/180)

        self.edges['B','X']['position'][0] = self.vehicle['wheelbase']*axleSkew
        self.edges['B','X']['position'][2] = self.vehicle['ride'] - self.vehicle['cg_height']

        ### Outboard A-Arm Design ###
        self.edges['W','UB']['position'][1] = self.edges['W','LB']['position'][1] \
            - np.tan(np.deg2rad(self.target['kpi']))*(self.edges['W','UB']['position'][2] - self.edges['W','LB']['position'][2])

        ### Roll and Pitch Centers ###
        self.nodes['I']['poi'].update(
            poi_factory('RC', np.array([self.coord('O','T','I')[0], 0, self.vehicle['cg_height']*(1-self.target['%roll']/100)]), 'kx')
        )

        self.nodes['I']['poi'].update(
            poi_factory('PC', np.array([0, self.coord('O','T','I')[1], self.vehicle['cg_height']*(1-self.target['%pitch']/100)]), 'kx')
        )

        ### Instant Centers ###
        zRoot = lambda p0, p1, pc, R, z: ((p1[0] - p0[0])/(p1[1] - p0[1]))*(z - p0[1]) + np.sqrt(R**2 - (z-pc[1])**2)
        jLine = lambda p0, p1, z: p0[0] + ((p1[0] - p0[0])/(p1[1] - p0[1]))*(z - p0[1])

        for (j,m,v) in [[1,'R','F'], [0,'P','S']]:
            self.nodes['I']['poi'].update(poi_factory(m+'IC', self.coord('O','T','I'), 'ko'))

            if np.isinf(self.target[v+'VSA']):
                self.nodes['I']['poi'][m+'IC']['position'][[j,2]] = [-np.inf, 0]
                continue

            self.nodes['I']['poi'][m+'IC']['position'][2] = \
                fsolve(lambda z: zRoot(self.coord('O','T','I')[[j,2]], self.coord(m+'C','I','I')[[j,2]], 
                                       self.coord('O','W','I')[[j,2]], self.target[v+'VSA'], z), 0)

            self.nodes['I']['poi'][m+'IC']['position'][j] = \
                jLine(self.coord('O','T','I')[[j,2]], self.coord(m+'C','I','I')[[j,2]], self.nodes['I']['poi'][m+'IC']['position'][2])
        
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        self.plot()
        

        _pause = 1

    def plot(self, ax: plt.axes=None):
        """Plot DoubleWishbone system"""
        # Default parameters
        ax = plt.gca() if ax is None else ax

        # Invoke KinematicSystem plot method
        super().plot(ax, 'I', 50)

        # Plot kingpin axis
        KPA = np.vstack((self.coord('O', 'LB', 'I'), self.coord('O', 'UB', 'I')))
        ax.plot(*KPA.T, 'k--')

        # Plot Centers
        ax.plot(*self.coord('RC','I','I'), 'kx')
        ax.plot(*self.coord('PC','I','I'), 'kx')

        ax.plot(*np.vstack([self.coord('O','T','I'), self.coord('RIC','I','I')]).T, 'k:')
        ax.plot(*np.vstack([self.coord('O','T','I'), self.coord('PIC','I','I')]).T, 'k:')

        # Configure axes
        ax.view_init(elev=20, azim=30)
        ax.set_zlim([0, 500])