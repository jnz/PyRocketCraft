import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_rocket_ode_model() -> AcadosModel:

    model_name = 'copterpos_ode'

    # constants
    # ---------
    gravity = 9.81
    mass_kg = 91.0 # 549.1

    # moment of inertia
    J_11 = 372.6  # Forward
    J_22 = 372.6  # Left
    J_33 = 1.55     # Up

    THRUST_MAX_N = 1800.0 # 8500.0
    THRUST_TAU = 2.5
    THRUST_VECTOR_TAU = 0.3
    THRUST_MAX_ANGLE = np.deg2rad(10.0)
    ATT_MAX_THRUST = 50.0

    # position of booster relative to CoG
    booster_pos_forward = 0.0
    booster_pos_left    = 0.0
    booster_pos_up      = -2.0

    att_x_booster_pos_forward = 0.0
    att_x_booster_pos_left    = 0.0
    att_x_booster_pos_up      = 2.0

    # set up states & controls
    # x
    q_0         = SX.sym('q_0') # qw (quaternion from body to navigation frame)
    q_1         = SX.sym('q_1') # qx
    q_2         = SX.sym('q_2') # qy
    q_3         = SX.sym('q_3') # qz
    q           = vertcat(q_0, q_1, q_2, q_3)

    omega_x     = SX.sym('omega_x') # rotation rates body vs. navigation frame in body frame
    omega_y     = SX.sym('omega_y')
    omega_z     = SX.sym('omega_z')
    omega       = vertcat(omega_x, omega_y, omega_z)

    # xdot
    q0_dot      = SX.sym('q0_dot')
    q1_dot      = SX.sym('q1_dot')
    q2_dot      = SX.sym('q2_dot')
    q3_dot      = SX.sym('q3_dot')
    q_dot       = vertcat(q0_dot, q1_dot, q2_dot, q3_dot)

    omega_x_dot = SX.sym('omega_x_dot')
    omega_y_dot = SX.sym('omega_y_dot')
    omega_z_dot = SX.sym('omega_z_dot')
    omega_dot   = vertcat(omega_x_dot, omega_y_dot, omega_z_dot)

    pos     = SX.sym('pos', 3, 1) # position in meter
    pos_dot = SX.sym('pos_dot', 3, 1)
    vel     = SX.sym('vel', 3, 1) # velocity in m/s
    vel_dot = SX.sym('vel_dot', 3, 1)

    thrust      = SX.sym('thrust')
    thrust_dot  = SX.sym('thrust_dot')
    t_alpha     = SX.sym('t_alpha') # thrust vector angle in forward/backward (X) direction
    t_beta      = SX.sym('t_beta')  # thrust vector angle in right/left (Y) direction
    t_alpha_dot = SX.sym('t_alpha_dot')
    t_beta_dot  = SX.sym('t_beta_dot')

    #              0  4      7    10   13
    x    = vertcat(q, omega, pos, vel, thrust, t_alpha, t_beta)
    xdot = vertcat(q_dot, omega_dot, pos_dot, vel_dot, thrust_dot, t_alpha_dot, t_beta_dot)

    nx = x.size()[0]
    weight_diag = np.ones((nx,)) * 1e-6  # default weight
    weight_diag[0] = 6.0
    weight_diag[1] = 6.0
    weight_diag[2] = 6.0
    weight_diag[3] = 6.0

    weight_diag[4] = 80.1
    weight_diag[5] = 80.1
    weight_diag[6] = 0.5

    weight_diag[7] = 1.0    # pos East
    weight_diag[8] = 1.0    # pos North
    weight_diag[9] = 3.0    # altitude
    weight_diag[10] = 5.1   # East velocity
    weight_diag[11] = 5.1   # North velocity
    weight_diag[12] = 25.0  # vertical velocity

    # Control input u
    u = SX.sym('u', 5, 1) # thrust (0 to 1), alpha (-1 to 1), beta (-1 to 1), att_x_thrust, att_y_thrust

    # System Dynamics
    # ---------------
    R_b_to_n = SX.sym('R_b_to_n', 3, 3)
    R_b_to_n[0, 0] = 1.0 - 2.0*q_2*q_2 - 2.0*q_3*q_3
    R_b_to_n[0, 1] = 2.0*q_1*q_2 - 2.0*q_3*q_0
    R_b_to_n[0, 2] = 2.0*q_1*q_3 + 2.0*q_2*q_0
    R_b_to_n[1, 0] = 2.0*q_1*q_2 + 2.0*q_3*q_0
    R_b_to_n[1, 1] = 1.0 - 2.0*q_1*q_1 - 2.0*q_3*q_3
    R_b_to_n[1, 2] = 2.0*q_2*q_3 - 2.0*q_1*q_0
    R_b_to_n[2, 0] = 2.0*q_1*q_3 - 2.0*q_2*q_0
    R_b_to_n[2, 1] = 2.0*q_2*q_3 + 2.0*q_1*q_0
    R_b_to_n[2, 2] = 1.0 - 2.0*q_1*q_1 - 2.0*q_2*q_2

    tau_v = SX.sym('tau_v', 3, 1)  # torque from vectored thrust
    booster_pos_cross = np.array([ [ 0,               -booster_pos_up,       booster_pos_left],
                                   [ booster_pos_up,   0,                   -booster_pos_forward],
                                   [-booster_pos_left, booster_pos_forward,  0 ] ])

    att_x_booster_pos_cross = np.array([ [  0,                     -att_x_booster_pos_up,       att_x_booster_pos_left ],
                                         [  att_x_booster_pos_up,   0,                         -att_x_booster_pos_forward ],
                                         [ -att_x_booster_pos_left, att_x_booster_pos_forward,  0 ] ])

    thrust_body = SX.sym('thrust_body', 3, 1)
    thrust_body[0] = thrust*t_alpha
    thrust_body[1] = thrust*t_beta
    thrust_body[2] = thrust

    att_x_thrust = SX.sym('att_x_thrust', 3, 1)
    att_x_thrust[0] = u[3] * ATT_MAX_THRUST
    att_x_thrust[1] = u[4] * ATT_MAX_THRUST
    att_x_thrust[2] = 0.0
    tau_v = booster_pos_cross @ thrust_body + att_x_booster_pos_cross @ att_x_thrust

    thrust_dot_eq  = (THRUST_MAX_N*u[0] - thrust)/THRUST_TAU
    t_alpha_dot_eq = (THRUST_MAX_ANGLE*u[1] - t_alpha)/THRUST_VECTOR_TAU
    t_beta_dot_eq  = (THRUST_MAX_ANGLE*u[2] - t_beta)/THRUST_VECTOR_TAU

    gravity_n = SX.sym('gravity_n', 3, 1)
    gravity_n[0] = 0.0
    gravity_n[1] = 0.0
    gravity_n[2] = -gravity

    vel_dot = R_b_to_n@(thrust_body/mass_kg) + gravity_n

    f_expl = vertcat( -(omega_x*q_1)/2.0 - (omega_y*q_2)/2.0 - (omega_z*q_3)/2.0,
                       (omega_x*q_0)/2.0 - (omega_y*q_3)/2.0 + (omega_z*q_2)/2.0,
                       (omega_y*q_0)/2.0 + (omega_x*q_3)/2.0 - (omega_z*q_1)/2.0,
                       (omega_y*q_1)/2.0 - (omega_x*q_2)/2.0 + (omega_z*q_0)/2.0,
                       (tau_v[0] + J_22*omega_y*omega_z - J_33*omega_y*omega_z)/J_11,
                       (tau_v[1] - J_11*omega_x*omega_z + J_33*omega_x*omega_z)/J_22,
                       (tau_v[2] + J_11*omega_x*omega_y - J_22*omega_x*omega_y)/J_33,
                       vel,
                       vel_dot,
                       thrust_dot_eq,
                       t_alpha_dot_eq,
                       t_beta_dot_eq,
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.weight_diag = weight_diag

    return model

