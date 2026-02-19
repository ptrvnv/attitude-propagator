# attitude_utils.py

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def build_Omega(w):
    """
    Objective
    ---------
    Build the Omega matrix needed for quaternion kinematic propagation.
    
    Description
    -----------
    Given the angular velocity vector w = [wx, wy, wz] in the body frame,
    the function constructs the 4x4 Omega matrix needed to calculate the 
    quaternion derivative, where q = [qx, qy, qz, qs] with qs being the scalar part of the quaternion.

    Input
    -----
    w: ndarray, shape (3,)
        Angular velocity vector [wx, wy, wz] in [rad/s].

    Output
    ------
    Omega: ndarray, shape (4,4)
        The Omega matrix defining the quaternion kinematic 
        propagation of a quaternion q = [qx, qy, qz, qs] with qs being its scalar part.
    """
    w_cross = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    Omega = np.block([
        [-w_cross, w.reshape(3,1)],
        [-w.reshape(1,3), np.array([[0]])]
    ])
    return Omega

def propagate_attitude(t, x, M_rot, J, J_inv, tau_d):
    """
    Objective
    ---------
    Compute the time derivative of the satellite's attitude and angular velocity.
    
    Description
    -----------
    The standard attitude kinematics and dynamics equations of a rigid body are defined.
    The state vector consists of the attitude expressed as a quaternion (scalar-last convention) 
    in the body frame wrt the inertial frame and the angular velocity between the body frame and 
    the inertial frame (e.g., ECI). The function computes the quaternion (scalar-last convention) 
    and angular velocity derivatives q_dot and w_dot.      

    Input
    -----
    t: float
        Current simulation time required by ODE solvers [s].
    x: ndarray, shape (7,)
        Current state vector:
        - q : quaternion of the attitude in the body frame wrt the inertial frame
        - w : angular velocity of the body in the body frame [rad/s].
    J: ndarray, shape (3,3)
        Satellite inertia matrix [kgm^2].
    J_inv: ndarray, shape (3,3)
        Inverse of the satellite inertia matrix [kgm^2].
    tau_d: ndarray, shape (3,)
        Optional disturbance torque applied to the satellite [Nm].

    Output
    ------
    dxdt: ndarray, shape (7,)
        Time derivative of the state vector [q_dot, w_dot].
    """
    q = x[0:4]
    w = x[4:7]
    q_dot = 0.5 * build_Omega(w).dot(q)
    w_dot = J_inv.dot(tau_d - np.cross(w, J.dot(w)))
    return np.hstack((q_dot, w_dot))

def convert_to_quat(att, input_choice):
    """
    Objective
    ---------
    Convert an input attitude representation into a quaternion.

    Description
    -----------
    Depending on the selected input format, the function converts the given
    attitude description into its equivalent quaternion representation 
    (scalar-last convention). The supported input types are Quaternion, 
    Euler angles and Modified Rodrigues Parameters.

    Input
    -----
    att: ndarray
        Attitude representation, format depending on `input_choice`:
        - (4,) if quaternion
        - (3,) if Euler angles [deg]
        - (3,) if MRP.
    input_choice: int
        Integer flag selecting the input type:
        - 1: Quaternion
        - 2: Euler angles [deg]
        - 3: MRP.

    Output
    ------
    q: ndarray, shape (4,)
        Quaternion corresponding to the input attitude.
        
    Notes
    -----
    The function raises a ValueError if "input_choice" is not recognized.
    """
    if input_choice == 1:
        return att
    elif input_choice == 2:
        return R.from_euler('zyx', att, degrees=True).as_quat()
    elif input_choice == 3:
        return R.from_mrp(att).as_quat()
    else:
        raise ValueError(f"Unknown input format: {input_choice}. Must be 1 (Quaternion), 2 (Euler), or 3 (MRP).")

def convert_output_attitude(t, rot_data, output_choice):
    """
    Objective
    ---------
    Convert the propagated attitude data in the desired output format.

    Description
    -----------
    Given the attitude data obtained from the attitude propagation, 
    the function converts it into a Quaternion, Euler angles or 
    Modified Rodrigues Parameters.

    Input
    -----
    t: ndarray, shape (N,)
        Time vector corresponding to the propagated attitude states [s].
    rot_data: scipy.spatial.transform.Rotation
        Rotation object containing the attitude data.
    output_choice: int
        Format selection for the output attitude:
        - 1: Quaternion
        - 2: Euler angles [deg]
        - 3: MRP.

    Output
    ------
    att_data: ndarray, shape (N, n)
        Converted attitude data:
        - (N,4) for quaternions
        - (N,3) for Euler angles [deg]
        - (N,3) for MRPs.

    Notes
    -----
    The function raises a ValueError if "output_choice" is not recognized.
    """
    if output_choice == 1:
        q_data = rot_data.as_quat()
        return q_data
    elif output_choice == 2:
        euler_data = rot_data.as_euler('zyx', degrees=True)
        return euler_data
    elif output_choice == 3:
        MRP_data = rot_data.as_mrp()
        return MRP_data
    else:
        raise ValueError(f"Unknown output format: {output_choice}. Must be 1 (Quaternion), 2 (Euler), or 3 (MRP).")

def run_attitude_propagation(input_format=None, att0=None, w0=None, J=None, tau_d=None, t_span=None, dt=0.1, output_format=None):
    """
    Objective
    ---------
    Perform attitude propagation of a rigid satellite and visualize the results.

    Description
    -----------
    The function integrates the attitude kinematics and dynamics equations of a rigid 
    satellite over time. The propagation is performed in quaternion form, using the 
    specified inertia matrix and a disturbance torque.
    The function handles different input attitude formats (Quaternion, Euler angles, MRP),
    converts them to a quaternion representation (scalar-last convention) and computes the attitude 
    of the satellite in the body frame wrt the inertial frame. 
    The resulting attitude is then converted to the desired output format (Quaternion, Euler angles or MRP) 
    and plotted over the simulation time. Angular velocity components are also propagated and visualized.

    Input
    -----
    input_format: int, optional
        Specifies the format of the initial attitude input:
        - 1: Quaternion
        - 2: Euler angles [deg]
        - 3: Modified Rodrigues Parameters.
    att0: ndarray, shape (4,) or (3,), optional
        Initial attitude of the satellite in the body frame wrt the inertial frame in the format
        defined by "input_format".
    w0: ndarray, shape (3,), optional
        Initial angular velocity between the body and the inertial frame [deg/s].
    J: ndarray, shape (3,3), optional
        Inertia matrix of the satellite in body frame coordinates [kgm^2].
    tau_d: ndarray, shape (3,), optional
        External or disturbance torque applied to the satellite (Nm).
    t_span: ndarray, shape (2,), optional
        Start and end times of the simulation [s].
    dt: float, optional
        
    output_format: int, optional
        Specifies the desired output attitude format for visualization:
        - 1: Quaternion
        - 2: Euler angles [deg]
        - 3: Modified Rodrigues Parameters

    Output
    ------
    t: ndarray, shape (N,)
        Simulation time vector [s].
    att_B2I: ndarray, shape (N, n)
        Time history of the body-to-inertial attitude in the selected output format.
    w_B2I: ndarray, shape (N, 3)
        Time history of the body angular velocity components [deg/s].

    Notes
    -----
    The ODE integration time step is fixed at 0.1s. 
    The integration is performed using "scipy.integrate.solve_ivp" with the fixed tolerances 
    "rtol=1e-9" and "a"tol=1e-12" for numerical stability.
    The function normalizes quaternions at each step.
    """
    if att0 is None:
        att0 = np.array([0.936680, 0.216195, 0.124597, 0.245695])
    if w0 is None:
        w0 = np.array([0.1, 0.2, -0.05])
    M_rot = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
    if J is None:
        J = np.array([[1.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 0.3]])
    J_inv = np.linalg.inv(J)
    if tau_d is None:
        tau_d = np.array([0, 0, 0])
    if t_span is None:
        t_span = np.array([0, 1000])
        
    q0 = convert_to_quat(att0, input_format)
    rot_S2I = R.from_quat(q0)
    rot_B2S = R.from_matrix(M_rot) # fixed!
    rot_B2I = rot_S2I * rot_B2S
    q0 = rot_B2I.as_quat()
    w0 = np.deg2rad(w0)
    x0 = np.concatenate((q0, w0))
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    sol = solve_ivp(propagate_attitude, t_span, x0, args=(M_rot, J, J_inv, tau_d), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    
    q_B2I = np.array([q / np.linalg.norm(q) for q in sol.y[0:4, :].T])
    rot_B2I = R.from_quat(q_B2I)
    att_B2I = convert_output_attitude(sol.t, rot_B2I, output_format)
    w_B2I = np.array([np.rad2deg(w_b) for w_b in sol.y[4:7, :].T])
    
    return sol.t, att_B2I, w_B2I