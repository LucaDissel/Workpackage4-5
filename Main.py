#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.linalg as LA
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from functools import lru_cache

from typing import Any, Iterable, Optional

# In[ ]:
from NACA_63210 import airfoil_surface
from Lift_diagrams import chord, V_distribution, M_distribution, T_distribution, dynamic_p, q_crit, CL_crit, V, M, y

# In[ ]:
b2 = 16.07 / 2
root_c = 2.46
tip_c = 1.11
E = 68.9 * 10**9
G = 26 * 10**9
v = 0.33
UTS = 290 * 10**6
YieldStregth = 240 * 10**6
rho_al = 2700

# In[ ]:
def interp_on_domain(range_object):
    def wrapper(func):
        @lru_cache(maxsize=None)
        def make_interp1d(*args, **kwargs):
            return interp1d(range_object,
                            func(range_object, *args, **kwargs),
                            kind='cubic',
                            fill_value='extrapolate')

        def return_func(y, *args, **kwargs):
            interpreted_func = make_interp1d(*args, **kwargs)
            return interpreted_func(y)

        return return_func

    return wrapper


# In[ ]:


class t:
    def __init__(self, start_thickness, end_thicnkess=None, length=b2) -> None:
        if start_thickness < 0 or end_thicnkess < 0:
            raise ValueError('Cannot have negative thicknesses')
        self.t1 = start_thickness * 1e-3
        if end_thicnkess is not None:
            self.t2 = end_thicnkess * 1e-3
        else:
            self.t2 = end_thicnkess
        self.length = length

    def __call__(self, y):
        """Return thickness in m at y"""
        return self.t1 - (self.t1 -
                          self.t2) / self.length * y * (self.length >= y)


# In[ ]:
class Panel:
    def __init__(self, point1, point2, thickness: t, span=b2):
        """point1 and point2 are coordinates relative to the chord, thickness in mm"""
        # make sure point2 has the highest z coordinate
        if point1[1] > point2[1]:  
            point1, point2 = point2, point1
        self.p1 = point1
        self.p2 = point2
        self.vector = point2 - point1
        self.centre = (point1 + self.vector / 2)
        self.h = LA.norm(self.vector)
        self.t = thickness
        # z is the vertical distance from the x axis relative to chord
        self.z = (point1 + self.vector / 2)[1]
        self.span = span

    def l(self, y):
        return self.h * chord(y) * (y <= self.span)

    def A(self, y):
        return self.h * chord(y) * self.t(y) * (y <= self.span)

    def Q_x(self, y):
        """Returns first moment of area about z=0"""
        return self.centre[1] * chord(y) * self.A(y) * (y <= self.span)

    def Q_z(self, y):
        """Returns first moment of area about x=0"""
        return self.centre[0] * chord(y) * self.A(y) * (y <= self.span)

    def I_xc(self, y):
        """Returns second moment of area about own centroid"""
        cos_a = self.vector[0] / self.h
        sin_a = self.vector[1] / self.h

        return (self.t(y) * (self.h * chord(y)) *
                (self.t(y)**2 * cos_a**2 +
                 (self.h * chord(y))**2 * sin_a**2) / 12 * (y <= self.span))

    def I_xx(self, y, z_centroid):
        """Returns second moment of area about centroid"""
        d = self.centre[1] * chord(y) - z_centroid
        return self.I_xc(y) + self.A(y) * d**2 * (y <= self.span)

    def z_at_x(self, x):
        if self.vector[0] == 0:
            raise ValueError(
                "Horizontal plate does not have a heigh as a function of x")
        return (x - self.p1[0]) / self.vector[0] * self.vector[1] + self.p1[1]


# In[ ]:


class Stringer:
    def __init__(self, area, x, length, upper: bool = True):
        """area is in square mm, point is the coordinate realtive to the chord and length is the spanwise length of the stringer"""
        self.upper = upper
        self.area = area * 1e-6
        self.point = np.array([x, 0])
        self.length = length

    @property
    def x(self):
        return self.point[0]

    @property
    def z(self):
        return self.point[1]

    @z.setter
    def z(self, new_z):
        self.point[1] = new_z
    
    def A(self, y):
        return self.area * (self.length >= y)

    def Q_x(self, y):
        """Returns first moment of area about z=0"""
        return self.A(y) * self.point[1] * chord(y) * (self.length >= y)

    def Q_z(self, y):
        """Returns first moment of area about x=0"""
        return self.A(y) * self.point[0] * chord(y) * (self.length >= y)

    @property
    def I_xc(self):
        return 0

    def I_xx(self, y, z_centroid):
        """Returns second moment of area about z_centroid"""
        return self.I_xc + self.A(y) * (self.point[1] * chord(y) -
                                     z_centroid)**2 * (self.length >= y)


# In[ ]:


class L_stringer(Stringer):
    def calculate_I_xx(self, l1, l2, t):
        pass

    def __init__(self, x, length, l1, l2, t, upper=True):
        area = (l1 + l2) * t
        self.l1 = l1
        self.l2 = l2
        self.t = t
        self._Ixc = self.calculate_I_xx(l1, l2, t)
        super().__init__(area, x, length, upper=upper)

    @property
    def I_xc(self):
        return self._Ixc


# In[ ]:


class WingBox:
    Kc_lim = 4
    Ks_lim = 5.2
    Kv = 1.5 * 1.5

    def __init__(self,
                 front_spar_x,
                 front_spar_t,
                 rear_spar_x,
                 rear_spar_t,
                 upper_panel_t,
                 lower_panel_t,
                 stringers: Optional[Iterable[Stringer]] = None,
                 middle_spar_x=None,
                 middle_spar_t=None,
                 middle_spar_span=b2,
                 ribs=[0, b2]):

        if middle_spar_x is None:
            self.single_cell = True
        else:
            self.single_cell = False
        # Define points of the corners of the wingbox
        front_upper_z, front_lower_z = airfoil_surface(front_spar_x)
        front_upper = np.array([front_spar_x, front_upper_z])
        front_lower = np.array([front_spar_x, front_lower_z])
        rear_upper_z, rear_lower_z = airfoil_surface(rear_spar_x)
        rear_upper = np.array([rear_spar_x, rear_upper_z])
        rear_lower = np.array([rear_spar_x, rear_lower_z])

        self.points = {
            'front_upper': front_upper,
            'front_lower': front_lower,
            'rear_upper': rear_upper,
            'rear_lower': rear_lower
        }

        if not self.single_cell:
            # Add extra points if there is a middle spar
            middle_upper_z, middle_lower_z = airfoil_surface(middle_spar_x)
            middle_upper = np.array([middle_spar_x, middle_upper_z])
            middle_lower = np.array([middle_spar_x, middle_lower_z])
            self.points['middle_upper'] = middle_upper
            self.points['middle_lower'] = middle_lower

        self.panels = {}
        front_spar = Panel(front_lower, front_upper, front_spar_t)
        rear_spar = Panel(rear_lower, rear_upper, rear_spar_t)

        if self.single_cell:
            upper_panel = Panel(front_upper, rear_upper, upper_panel_t)
            lower_panel = Panel(front_lower, rear_lower, lower_panel_t)
            self.panels = {
                'front_spar': front_spar,
                'rear_spar': rear_spar,
                'upper_panel': upper_panel,
                'lower_panel': lower_panel
            }
        else:
            upper_panel_1 = Panel(front_upper, middle_upper, upper_panel_t)
            upper_panel_2 = Panel(middle_upper, rear_upper, upper_panel_t)
            lower_panel_1 = Panel(front_lower, middle_lower, lower_panel_t)
            lower_panel_2 = Panel(middle_lower, rear_lower, lower_panel_t)
            middle_spar = Panel(middle_lower, middle_upper, middle_spar_t,
                                middle_spar_span)
            self.panels = {
                'front_spar': front_spar,
                'rear_spar': rear_spar,
                'upper_panel_1': upper_panel_1,
                'upper_panel_2': upper_panel_2,
                'lower_panel_1': lower_panel_1,
                'lower_panel_2': lower_panel_2,
                'middle_spar': middle_spar
            }
            # Add extra lists so the two cells can be accesed seperately
            self.left_panels = {
                'front_spar': front_spar,
                'upper_panel_1': upper_panel_1,
                'lower_panel_1': lower_panel_1,
                'middle_spar': middle_spar
            }
            self.right_panels = {
                'rear_spar': rear_spar,
                'upper_panel_2': upper_panel_2,
                'lower_panel_2': lower_panel_2,
                'middle_spar': middle_spar
            }

        if stringers is not None:
            for stringer in stringers:
                panel = self.find_panel(stringer.x, stringer.upper)
                z = panel.z_at_x(stringer.x)
                stringer.z = z

        self.stringers = stringers
        self.ribs = ribs
        self.T = None

    def find_panel(self, x, upper: bool):
        if upper:
            key = 'upper'
        else:
            key = 'lower'
        if self.single_cell:
            return self.panels[key + '_panel']
        else:
            middle_x = self.panels['middle_spar'].p1[0]
            if x == middle_x:
                raise ValueError('This is the location of the middle spar')
            if x < middle_x:
                return self.panels[key + '_panel_1']
            else:
                return self.panels[key + '_panel_2']

    ################################################################################
    # Section Properties
    def A(self, y):
        """Returns the area of the matrial of a cross sectio at a given location of span"""
        A = 0
        for panel in self.panels.values():
            A += panel.A(y)

        if self.stringers is not None:
            for stringer in self.stringers:
                A += stringer.A(y)
        return A

    def enclosed_A(self, y, x1, x2):
        assert x2 > x1
        z1 = airfoil_surface(x1)
        z2 = airfoil_surface(x2)
        return (x2 - x1) / 2 * (z1[0] - z1[1] + z2[0] - z2[1]) * chord(y)**2

    def z_centroid(self, y):
        """Returns the z-coordinate of the centroid at a given location of span"""
        A = self.A(y)
        Az = 0
        for panel in self.panels.values():
            Az += panel.Q_x(y)
        if self.stringers is not None:
            for stringer in self.stringers:
                Az += stringer.Q_x(y)
        return Az / A

    def x_centroid(self, y):
        """Returns the z-coordinate of the centroid at a given location of span"""
        A = self.A(y)
        Ax = 0

        for panel in self.panels.values():
            Ax += panel.Q_z(y)
        if self.stringers is not None:
            for stringer in self.stringers:
                Ax += stringer.Q_z(y)
        return Ax / A

    def stringer_discontinuities(self):
        lst = []
        for stringer in self.stringers:
            if stringer.length != b2:
                lst.append(stringer.length)
            return lst

    def w(self, y):
        """Returns dW/dy, the weight distribution of the wing"""
        return self.A(y) * rho_al

    @property
    def W(self):
        """Returns the total weight of two wingboxes"""
        return quad(self.w, 0, b2)[0] * 2

    def I_xx(self, y):
        """Returns the moment of inertia about the centroid of the wingbox"""
        c = chord(y)
        z_centroid = self.z_centroid(y)
        I_xx = 0

        for panel in self.panels.values():
            I_xx += panel.I_xx(y, z_centroid)

        if self.stringers is not None:
            for stringer in self.stringers:
                I_xx += stringer.I_xx(y, z_centroid) * (stringer.length >= y)

        return I_xx

    def solve_shearflow(self, y, T):
        if not self.single_cell:
            # a * q1 + b * q2 - dthetadx = 0
            # c * q1 + d * q2 - dthetadx = 0
            # e * q1 + f * q2 - 0 = T
            # q1 is shear flow in left box, q2 shear flow in the right box
            points = self.points
            panels = self.panels
            x1 = points["front_upper"][0]
            x2 = points["middle_upper"][0]
            x3 = points["rear_upper"][0]
            A_1 = self.enclosed_A(y, x1, x2)
            A_2 = self.enclosed_A(y, x2, x3)
            a = 0
            for panel in self.left_panels.values():
                a += panel.l(y) / panel.t(y)
            a *= 1 / (2 * A_1 * G)
            b = -panels['middle_spar'].l(y) / panels['middle_spar'].t(y) / (
                2 * A_1 * G)

            d = 0
            for panel in self.right_panels.values():
                d += panel.l(y) / panel.t(y)
            d *= 1 / (2 * A_2 * G)
            c = -panels['middle_spar'].l(y) / panels['middle_spar'].t(y) / (
                2 * A_2 * G)

            e = 2 * A_1
            f = 2 * A_2
            system = np.array([[a, b, np.broadcast_to(-1, np.shape(a))],
                               [c, d, np.broadcast_to(-1, np.shape(a))],
                               [e, f, np.broadcast_to(0, np.shape(a))]])

            if system.ndim > 2:
                system = np.moveaxis(system, 2, 0)
                N, M, M = np.shape(system)
                if np.shape(T) == (N,):
                    T_column = np.reshape(T, (N, 1))
                    zeros = np.zeros((N, 2))
                    righthandside = np.hstack((zeros, T_column))
                elif np.shape(T) in ((), (1,)): 
                    righthandside = np.broadcast_to(np.array([0, 0, T]), (N, M))
                else:
                    raise ValueError('T must either have dimension 1 or have the same dimension as y')
                q1, q2, dthetadx = np.swapaxes(LA.solve(system, righthandside),
                                               0, 1)
            else:
                q1, q2, dthetadx = LA.solve(system, np.array([0, 0, T]))
            span_m = self.panels['middle_spar'].span

        if self.single_cell or (span_m != b2):
            x1 = self.points["front_upper"][0]
            x2 = self.points["rear_upper"][0]
            if self.single_cell:
                A = self.enclosed_A(y, x1, x2)
            else:
                A = A_1 + A_2
            q = T / (2 * A)
            integral = 0
            for panel in self.panels.values():
                integral += panel.l(y) / panel.t(y)
            dthetadx_single_cell = q / (2 * A * G) * integral

            if not self.single_cell:
                q1 = np.where(y <= span_m, q1, q)
                q2 = np.where(y <= span_m, q2, q)
                dthetadx = np.where(y <= span_m, dthetadx, dthetadx_single_cell)
            else:
                return q, q, dthetadx_single_cell

        return q1, q2, dthetadx

    def J_single_cell(self, y):
        # Calculate area inside box
        x1, x2 = self.points["front_upper"][0], self.points["rear_upper"][0]
        A = self.enclosed_A(y, x1, x2)
        integral = 0
        for panel in self.panels.values():
            integral += panel.l(y) / panel.t(y)
        return 4 * A**2 / integral

    def J_multi_cell(self, y):
        T = 1
        q1, q2, dthetadx = self.solve_shearflow(y, T)
        J = T / (G * dthetadx)
        return J

    def J(self, y):
        """Returns the torsional stiffness of the wingbox"""
        if self.single_cell:
            return self.J_single_cell(y)
        else:
            return self.J_multi_cell(y)

    #############################################################################
    # Stress and Shear
    def sigma(self, y, z, M=M):
        """Returns the stress in Pa at a given y of span and z with respect to chord"""
        z = z * chord(y)
        z_c = self.z_centroid(y)
        return M(y) * (z - z_c) / self.I_xx(y)

    def tau_max(self, y, T, V=V):

        spar_area = self.panels['front_spar'].A(y) + self.panels['rear_spar'].A(y)
        if self.single_cell:
            spar_area += self.panels['middle_spar'].A(y)
        tau_avg = V(y) / spar_area
        tau_max = tau_avg * self.Kv 
        q1, q2, _ = self.solve_shearflow(y, T(y))
        front_tau_max = -q1 / self.panels['front_spar'].t(y) + tau_max
        rear_tau_max = q2 / self.panels['rear_spar'].t(y) + tau_max
        middle_tau_max = None
        if not self.single_cell:
            middle_tau_max = (q1 - q2) / self.panels['middle_spar'].t(y) + tau_max
        return front_tau_max, middle_tau_max, rear_tau_max
        

    #################################################################################
    # Analysis
    def check_skin_buckling(self):
        # if there are n ribs, there should be n sections that need to be checked (assuming no rib at tip)
        # loop over each section, each section has rib1, and rib2
        #
        # the stringers that are considered to be on the top panel are the ones with positive z coordinate
        # sigma_cr is evaluated at rib2, sigma_max is evaluated at both rib1 and rib2 and the highest is selected
        #
        # for stringer in self.stringers:
        #
        # sigm
        # take t at rib2
        # a is the distance between rib1 and rib2, don't think a is used in calculation if Kc is taken constant

        # loop over sections between ribs
        rib1 = self.ribs[0]
        loadfactor_is_positive = True  # for positive loadfactor
        if M(0.1) > 0:
            loadfactor_is_positive = False  # for negative loadfactor

        for rib2 in self.ribs[1:]:
            y1, y2 = rib1, rib2
            chord1, chord2 = chord(y1), chord(y2)

            section_points = [
                self.panels['front_spar'].p2, self.panels['rear_spar'].p2
            ]
            if not self.single_cell:
                section_points.append(self.panels['middle_spar'].p2)
            if self.stringers is not None:
                for stringer in self.stringers:
                    if (stringer.upper is loadfactor_is_positive) and (stringer.length >= y2):
                            section_points.append(stringer.point)
            section_points.sort(key=lambda elem: elem[0])

            # loop over sections between stringers
            print(f'Ribs at {rib1} and {rib2} m span')
            point1 = section_points[0]
            for point2 in section_points[1:]:
                b = LA.norm(point2 - point1) * chord2
                x = (point1[0] + point2[0]) / 2
                t2 = self.find_panel(x, True).t(y2)
                sigma_crit = np.pi**2 * self.Kc_lim * E / (
                    12 *(1 - v**2)) * (t2 / b)**2

                if loadfactor_is_positive:  # for positive loadfactor
                    z_max_or_min = min(point1[1], point2[1])
                else:  # for negative loadfactor
                    z_max_or_min = max(point1[1], point2[1])
                sigma1_max = self.sigma(y1, z_max_or_min)
                sigma2_max = self.sigma(y2, z_max_or_min)
                sigma_max = max(sigma1_max, sigma2_max)

                print(f'section between {point1[0]} and {point2[0]}')
                print(f"Sigma crit = {sigma_crit:2.2}")
                print(f"Sigma 1 max = {sigma1_max:2.2}")
                print(f"Sigma 2 max = {sigma2_max:2.2}")
                point1 = point2
            rib1 = rib2
    
    def check_shear_buckling(self):
        rib1 = self.ribs[0]
        for rib2 in self.ribs[1:]:
            y1, y2 = rib1, rib2
            chord1, chord2 = chord(y1), chord(y2)

    def plot(self):
        c = chord(0)
        x = np.linspace(0, 1, 100)
        plt.plot(x, airfoil_surface(x)[0], color='grey', linewidth=1)
        plt.plot(x, airfoil_surface(x)[1], color='grey', linewidth=1)
        plt.scatter(self.x_centroid(0) / c, self.z_centroid(0) / c)
        for panel in self.panels.values():
            if panel.vector[0] != 0:
                x = np.linspace(panel.p1[0], panel.p2[0], 100)
                plt.plot(x, panel.z_at_x(x), color='black', linewidth=3)
            else:
                x = panel.p1[0]
                z = np.linspace(panel.p1[1], panel.p2[1], 100)
                plt.plot(np.full(100, x), z, color='black', linewidth=3)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.gcf().set_size_inches(15, 15)
            plt.gcf().set_dpi(200)

    def plot_max_stress(self, M=M):
        c = chord(0)
        x = np.linspace(0, 1, 100)
        panels = [self.panels['front_spar'], self.panels['rear_spar']]
        if not self.single_cell:
            panels.append(self.panels['middle_spar'])
        z_max = max([panel.p2[1] for panel in panels])
        z_min = min([panel.p1[1] for panel in panels])
        stress_lower = self.sigma(y, z_min, M=M)
        stress_upper = self.sigma(y, z_max, M=M)
        plt.plot(y, stress_lower, label='$\sigma$ lower panel')
        plt.plot(y, stress_upper, label='$\sigma$ upper panel')
        plt.legend()

    def plot_max_shear(self, CL=CL_crit, q=q_crit, T=None, V=V):
        if T is None:
            T = T_distribution(CL, q, self)
        front_tau_max, middle_tau_max, rear_tau_max = self.tau_max(y, T, V)
        plt.plot(y, front_tau_max, label='$\\tau$ front spar')
        if not self.single_cell:
            plt.plot(y, middle_tau_max, label='$\\tau$ middle spar')
        plt.plot(y, rear_tau_max, label='$\\tau$ rear spar')
        plt.legend()


#%%
@interp_on_domain(y)
def d2zdx2(y, wingbox: WingBox):
    """calculate second derivative of the displacement"""
    return -(M(y) / E / wingbox.I_xx(y))


@interp_on_domain(y)
def dzdx(y, wingbox: WingBox):
    """calculate first derivative through integration"""
    quad_vec = np.vectorize(quad)
    return quad_vec(d2zdx2, 0, y, args=(wingbox))[0]


def z(y, wingbox: WingBox):
    """calculate displacement through integration"""
    quad_vec = np.vectorize(quad)
    return quad_vec(dzdx, 0, y, args=(wingbox))[0]


@interp_on_domain(y)
def dthetadx(y, wingbox: WingBox, CL=CL_crit, q=q_crit):
    """calculate first derivative of the rotation"""
    T = T_distribution(CL, q, wingbox)
    return T(y) / wingbox.J(y) / G


def theta(y, wingbox: WingBox, CL=CL_crit, q=q_crit):
    """calculate rotation through integration"""
    # quad_vec = np.vectorize(quad)
    return [quad(dthetadx, 0, i, args=(wingbox, CL, q))[0] * 180 / np.pi for i in y]


# Luca typed this
class C_stringer(Stringer):
    
    def __init__(self, x, length, h, w, t, upper=True):
        self.height = h * 1e-3
        self.width = w * 1e-3
        self.thick = t * 1e-3
        area = 2*w*t + h*t - 3*t**2
        super().__init__(area, x, length, upper=upper)
    
    @property
    def zbar(self):
        Q = self.height * self.thick * (self.height/2) + (self.width-self.thick) * self.thick * (self.height - self.thick/2)
        A = 2*self.height*self.thick + self.height*self.thick - 3*self.thick**2
        return Q / A

    @property
    def I_xc(self):
        Ibar = (1/12)*(2*(self.width - self.thick) * self.thick**3 + self.height**3 * self.thick) 
        steiner1 = ((self.width - self.thick)*self.thick)*((self.zbar)**2 + (self.height - self.thick/2 - self.zbar)**2)
        steiner2 = (self.height*self.thick)*(self.height/2 - self.zbar)**2
        return Ibar + steiner1 + steiner2
    
    @property
    def I_over_A(self):
        """Returns inertia over area. Used for column buckling"""
        return self.I_xc / (2*self.height*self.thick + self.height*self.thick - 3*self.thick**2)