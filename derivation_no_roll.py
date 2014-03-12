#!/usr/bin/env python
from sympy.physics.mechanics import (dynamicsymbols, inertia, KanesMethod,
    Point, ReferenceFrame, RigidBody, mprint)
from sympy import symbols

t, m, g = symbols('t m g')                      # Time, total mass, gravity

q = dynamicsymbols('q:2')                       # Coordinates
qd = [qi.diff(t) for qi in q]                   # Coordinate time derivatives
u = dynamicsymbols('u:2')                       # Speeds
ud = [ui.diff(t) for ui in u]                   # Speed time derivatives

kindiffs = [qd[i] - u[i] for i in range(2)]     # Kinematic differential eqns

A = ReferenceFrame('A')                         # Inertial frame

B = A.orientnew('B', 'Axis', [q[0], A.y])       # Enclosure frame
B.set_ang_vel(A, u[0]*A.y)

C = B.orientnew('C', 'Axis', [q[1], B.z])       # Flywheel frame
C.set_ang_vel(B, u[1]*B.z)

Ixx, Iyy, Izz = symbols('Ixx Iyy Izz')          # Enclosure inertia scalars
I_enclosure = inertia(B, Ixx, Iyy, Izz)         # Inertia tensor of enclosure

IFxx, IFyy, IFzz = symbols('IFxx IFyy IFzz')    # Flywheel inertia scalars
I_flywheel = inertia(C, IFxx, IFyy, IFzz)       # Inertia tensor  of flywheel

O = Point('O')                                  # Mass center, assumed to be
O.set_vel(A, 0)                                 # on gimbal axis so it has zero
O.set_acc(A, 0)                                 # velocity and acceleration

# Define rigid body objects
Enclosure = RigidBody('Enclosure', O, B, 0, (I_enclosure, O))
Flywheel = RigidBody('Flywheel', O, C, 0, (I_flywheel, O))
body_list = [Enclosure, Flywheel]

# List of forces and torques
tau = symbols('tau')                            # Gimbal torque scalar
torque_force_list = [(O, m*g*A.z),              # Gravitational force
                     (B, tau*A.y)]              # Gimbal torque vector


# Form equations of motion
KM = KanesMethod(A, q, u, kindiffs)
fr, frstar = KM.kanes_equations(torque_force_list, body_list)
frstar.simplify()
print("Pre simplification")
mprint(fr)
mprint(frstar)

# Simplifying assumptions
simplifications = {IFxx: symbols('I'),          # Flywheel transverse inertias
                   IFyy: symbols('I')}          # assumed to be equal
frstar = frstar.subs(simplifications)
frstar.simplify()
print("Post simplification")
mprint(fr)
mprint(frstar)


