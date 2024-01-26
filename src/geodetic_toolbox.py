# (c) Jan Zwiener (jan@zwiener.org)
#
# Geodetic Toolbox
# ----------------
#
# A collection of math helper functions.

import numpy as np


def quat_from_rpy(r, p, y):
    """
    Convert Euler angle (roll, pitch, and yaw) to a quaternion.

    :param r: roll [rad]
    :param p: pitch [rad]
    :param y: yaw [rad]
    :return: Unit quaternion, describing the rotation. np.array with real part
             at q[0] (qw, qx, qy, qz).
    """
    assert (np.abs(r) <= 2 * np.pi), "Invalid arguments"
    assert (np.abs(y) <= 2 * np.pi), "Invalid arguments"
    assert (np.abs(p) <= 0.5 * np.pi), "Invalid arguments"
    sr2 = np.sin(r * 0.5)
    cr2 = np.cos(r * 0.5)
    sp2 = np.sin(p * 0.5)
    cp2 = np.cos(p * 0.5)
    sy2 = np.sin(y * 0.5)
    cy2 = np.cos(y * 0.5)
    qreal = cy2 * cp2 * cr2 + sy2 * sp2 * sr2
    q1 = cy2 * cp2 * sr2 - sy2 * sp2 * cr2
    q2 = cy2 * sp2 * cr2 + sy2 * cp2 * sr2
    q3 = sy2 * cp2 * cr2 - cy2 * sp2 * sr2
    q = np.array([qreal, q1, q2, q3])

    return q

def quat_to_matrix(q):
    """
    This function creates a 3x3 rotation matrix from an input quaternion.

    :param q: Input quaternion (qw, qx, qy, qz) that describes the rotation
    :return: Rotation matrix 3x3 (np.array)
    """
    assert len(q) == 4, "Invalid arguments"

    a, b, c, d = q
    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d

    R = np.array([
        [a2 + b2 - c2 - d2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a2 - b2 + c2 - d2, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a2 - b2 - c2 + d2]
    ])

    return R

def quat_to_rpy(q):
    """
    Extract Euler angles (roll, pitch and yaw) from a quaternion.
    :param q: Input quaternion
    :return: 3x1 vector with angles in radians (roll, pitch, yaw)
    """
    return extract_rpy_from_R_b_to_n(quat_to_matrix(q))

def extract_rpy_from_R_b_to_n(R_b_to_n):
    """
    This function extracts the three angles roll, pitch, and yaw from
    an R_b_to_n matrix (rotation from body to navigation-frame).

    :param R_b_to_n: 3x3 matrix describing a body to n-frame transformation
    :return: 3x1 vector with angles in radians (roll, pitch, yaw)
    """
    assert R_b_to_n.shape == (3, 3), "Invalid arguments"

    roll = np.arctan2(R_b_to_n[2, 1], R_b_to_n[2, 2])
    pitch = np.arcsin(-R_b_to_n[2, 0])
    yaw = np.arctan2(R_b_to_n[1, 0], R_b_to_n[0, 0])
    return np.array([roll, pitch, yaw])

def quat_norm(q):
    """
    Normalize quaternion (make sure the length is 1.0).
    :param q Input quaternion
    :return Normalized quaternion with length == 1.0
    """
    if len(q) != 4:
        raise ValueError("Invalid arguments")

    abssquared = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    if abssquared < 10.0 * np.finfo(float).eps:
        raise ValueError("Quaternion length close to zero")

    qnorm = q / np.sqrt(abssquared)
    return qnorm

def quat_multiply(q1, q2):
    """
    Multiply two quaternions
    :param q1 Quaternion 1
    :param q2 Quaternion 2
    :return Result of q1 * q2
    """
    assert len(q1) == len(q2) == 4, "Invalid arguments"
    a, b, c, d = q1
    q_matrix = np.array([
        [ a, -b, -c, -d],
        [ b,  a, -d,  c],
        [ c,  d,  a, -b],
        [ d, -c,  b,  a]
    ])
    return q_matrix @ q2

def attitude_step(dt_sec, q, omega, torque_b, J, Jinv):
    """
    Rigid body attitude simulation step. Update the aircraft attitude based on
    torque coming from motors, wind and other torques acting on the body.

    Update attitude based on Euler's rigid body dynamics equations:

        J*omega_dot + omega x J* omega = torque

    Args:
        dt_sec: Simulation timestep in seconds (>= 0)
        q: 4x1 quaternion (from "body" to "n-frame"/ref. nav. frame). Hamilton
           convention, unit length quaternion q.
           q = q(1) + q(2)*i + q(3)*j + q(4)*k, with i*i=j*j=k*k=i*j*k=-1
        omega: Rotation rate (rad/s) of body wrt. ref. nav-frame (in body frame coord. system)
           (basically the 3x1 gyroscope measurement in rad/s)
        torque_b: 3 x 1 torque in Nm acting on body
        J: 3 x 3 inertia matrix of object in body frame (kg*m*m)
        Jinv: 3 x 3 inverse of J

    Returns:
        qnext: 4x1 Attitude quaternion after this simulation step.
        omeganext: 3x1 rotation rate (rad/s) of body wrt. ref. nav-frame (in
                   body frame coord. system) after this simulation step.
    """
    omega_cross = np.array([ [ 0,        -omega[2],  omega[1]],
                             [ omega[2],  0,        -omega[0]],
                             [-omega[1],  omega[0],  0 ] ])
    omega_dot = Jinv@(torque_b - omega_cross@J@omega)
    omeganext = omega + omega_dot*dt_sec

    delta = omega*dt_sec
    delta_abs = np.sqrt( delta @ delta )
    if delta_abs > 1e-6:
        img_part = delta / delta_abs * np.sin(delta_abs*0.5)
        qr = np.block([ np.cos(delta_abs*0.5), img_part ])
        qnext = quat_multiply(q, qr)
        qnext = quat_norm(qnext)
    else:
        qnext = q

    return qnext, omeganext

def quat_invert(q):
    """
    Return the inverse of an rotation quaternion.

    :param q: 4x1 orientation (unit-length) quaternion
    :return: Unit quaternion, describing the inverse rotation.
             np.array with real part at q[0] (qw, qx, qy, qz).
    """
    qinv = np.array([ q[0], -q[1], -q[2], -q[3] ])
    return qinv

