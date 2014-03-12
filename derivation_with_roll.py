#!/usr/bin/env python
from sympy.physics.mechanics import (dynamicsymbols, inertia, KanesMethod,
    Point, ReferenceFrame, RigidBody, mlatex, mprint)
from sympy import symbols, solve, Matrix

t, m, g = symbols('t m g')                      # Time, total mass, gravity

q = dynamicsymbols('q:3')                       # Coordinates
qd = [qi.diff(t) for qi in q]                   # Coordinate time derivatives
u = dynamicsymbols('u:3')                       # Speeds
ud = [ui.diff(t) for ui in u]                   # Speed time derivatives

kindiffs = [qd[i] - u[i] for i in range(3)]     # Kinematic differential eqns

A = ReferenceFrame('A')                         # Inertial frame

B = A.orientnew('B', 'Axis', [q[0], A.x])       # Roll frame
B.set_ang_vel(A, u[0]*A.x)

C = B.orientnew('C', 'Axis', [q[1], B.y])       # Enclosure frame
C.set_ang_vel(B, u[1]*B.y)

D = C.orientnew('D', 'Axis', [q[2], C.z])       # Flywheel frame
D.set_ang_vel(C, u[2]*C.z)

Ixx, Iyy, Izz = symbols('Ixx Iyy Izz')          # Enclosure inertia scalars
I_enclosure = inertia(C, Ixx, Iyy, Izz)         # Inertia tensor of enclosure

IFxx, IFyy, IFzz = symbols('IFxx IFyy IFzz')    # Flywheel inertia scalars
I_flywheel = inertia(D, IFxx, IFyy, IFzz)       # Inertia tensor  of flywheel

O = Point('O')                                  # Mass center, assumed to be
O.set_vel(A, 0)                                 # on gimbal axis so it has zero
O.set_acc(A, 0)                                 # velocity and acceleration

# Define rigid body objects
Enclosure = RigidBody('Enclosure', O, C, 0, (I_enclosure, O))
Flywheel = RigidBody('Flywheel', O, D, 0, (I_flywheel, O))
body_list = [Enclosure, Flywheel]

# List of forces and torques
tau = symbols('tau')                            # Gimbal torque scalar
torque_force_list = [(O, m*g*A.z),              # Gravitational force
                     (B, -tau*B.y),             # Gimbal reaction torque
                     (C,  tau*B.y)]             # Gimbal torque


# Form equations of motion
KM = KanesMethod(A, q, u, kindiffs)
fr, frstar = KM.kanes_equations(torque_force_list, body_list)

# Simplifying assumptions
simplifications = {IFxx: symbols('I'),          # Flywheel transverse inertias
                   IFyy: symbols('I'),          # assumed to be equal
                   IFzz: symbols('J'),
                   Ixx: 0,
                   Iyy: 0,
                   Izz: 0}
frstar = frstar.subs(simplifications)

MM = -frstar.jacobian(ud)
MM.simplify()
forcing = fr + frstar.subs({ud[0]: 0, ud[1]:0, ud[2]:0}).expand()
forcing.simplify()
print("Mass matrix:")
mprint(MM)
print("forcing vector:")
mprint(forcing)

eqns = MM * Matrix(ud) - forcing
udots = solve(eqns, ud)
for udi in ud:
    mprint(udi)
    print(":")
    mprint(udots[udi])
