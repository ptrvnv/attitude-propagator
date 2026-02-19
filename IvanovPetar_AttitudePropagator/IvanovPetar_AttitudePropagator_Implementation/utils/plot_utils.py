import matplotlib.pyplot as plt

def plot_quat(t, att):
    """
    Objective
    ---------
    Plot quaternion components over time.

    Input
    -----
    t: ndarray, shape (N,)
        Time vector [s].
    att: ndarray, shape (N, 4)
        Quaternion history representing the body-to-inertial attitude.

    Output
    ------
    fig: matplotlib.figure.Figure
        The generated Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t, att[:,0], label=r"$q_x$")
    ax.plot(t, att[:,1], label=r"$q_y$")
    ax.plot(t, att[:,2], label=r"$q_z$")
    ax.plot(t, att[:,3], label=r"$q_s$")
    ax.set_title("Quaternion in Body Frame wrt Inertial Frame", fontsize=20)
    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Quaternion Components", fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=16)
    fig.tight_layout()
    return fig

def plot_euler(t, att):
    """
    Objective
    ---------
    Plot Euler angles over time.

    Input
    -----
    t: ndarray, shape (N,)
        Time vector [s].
    att: ndarray, shape (N, 3)
        Euler angles history representing the body-to-inertial attitude.

    Output
    ------
    fig: matplotlib.figure.Figure
        The generated Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t, att[:,0], label=r"$\psi$ (yaw)")
    ax.plot(t, att[:,1], label=r"$\theta$ (pitch)")
    ax.plot(t, att[:,2], label=r"$\phi$ (roll)")
    ax.set_title("Euler Angles in Body Frame wrt Intertial Frame", fontsize=20)
    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Euler Angles [deg]", fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=16)
    fig.tight_layout()
    return fig

def plot_mrp(t, att):
    """
    Objective
    ---------
    Plot Modified Rodrigues Parameters over time.

    Input
    -----
    t: ndarray, shape (N,)
        Time vector [s].
    att: ndarray, shape (N, 3)
        MRP history representing the body-to-inertial attitude.

    Output
    ------
    fig: matplotlib.figure.Figure
        The generated Matplotlib figure object.
    """       
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t, att[:,0], label=r"$\sigma_1$")
    ax.plot(t, att[:,1], label=r"$\sigma_2$")
    ax.plot(t, att[:,2], label=r"$\sigma_3$")
    ax.set_title("MRP in Body Frame wrt Inertial Frame", fontsize=20)
    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Sigma Components", fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=16)
    fig.tight_layout()
    return fig

def plot_w(t, w):
    """
    Objective
    ---------
    Plot angular velocity over time.

    Input
    -----
    t: ndarray, shape (N,)
        Time vector [s].
    w: ndarray, shape (N, 3)
        Angular velocity history between the body frame and the inertial frame.

    Output
    ------
    fig: matplotlib.figure.Figure
        The generated Matplotlib figure object.
    """  
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t, w[:,0], label=r"$\omega_x$")
    ax.plot(t, w[:,1], label=r"$\omega_y$")
    ax.plot(t, w[:,2], label=r"$\omega_z$")
    ax.set_title("Angular velocity", fontsize=20)
    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Angular velocity [deg/s]", fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=16)
    fig.tight_layout()
    return fig