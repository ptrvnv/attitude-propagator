# main.py

import numpy as np
import os
import matplotlib.pyplot as plt

from utils.attitude_utils import run_attitude_propagation
from utils.plot_utils import plot_quat, plot_euler, plot_mrp, plot_w
from utils.general_utils import get_input

def main():    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    rslt_dir = os.path.join(base_dir, "results")
    os.makedirs(rslt_dir, exist_ok=True)
    
    input_input = input(
        "Select attitude input format:\n"
        "  1 - Quaternion [qx, qy, qz, qs]\n"
        "  2 - Euler angles [yaw, pitch, roll] in degrees\n"
        "  3 - MRP [sigma1, sigma2, sigma3]\n"
        "Enter choice (1-3) or press Enter to use default (1 - Quaternion):\n"
        )
    input_options = {1: "Quaternion", 2: "Euler", 3: "MRP"}
    input_choice = int(input_input) if input_input else 1
    if input_choice not in input_options:
        print("Invalid choice, using Quaternion (default).\n")
        input_choice = 1
        
    if input_choice == 1:
        att0 = get_input(
            f"Enter initial attitude as perceived from the sensor in the selected format\n"
            f"or press Enter to use q=[0.936680, 0.216195, 0.124597, 0.245695] as default:\n",
            default=[0.936680, 0.216195, 0.124597, 0.245695],
            shape=(4,)
        )
    if input_choice == 2:
        att0 = get_input(
            f"Enter initial attitude measured by the sensor in the selected format\n"
            f"or press Enter to use the Euler angles corresponding to q=[0.936680, 0.216195, 0.124597, 0.245695]\n"
            f"([psi, theta, phi] = [28.037779, -7.306605, 148.777773] deg) as default:\n",
            default=[-21.4392543, 19.85561543, 154.3998838],
            shape=(3,)
        )
    elif input_choice == 3:
        att0 = get_input(
            f"Enter initial attitude as perceived from the sensor in the selected format\n"
            f"or press Enter to use the sigma corresponding to q=[0.936680, 0.216195, 0.124597, 0.245695]\n"
            f"(sigma = [0.75193362, 0.17355371, 0.10002207]) as default:\n",
            default=[0.75193362, 0.17355371, 0.10002207],
            shape=(3,)
        )
        
    w0 = get_input(
        "Enter initial angular velocity vector w0 [deg/s] as a list\n"
        "or press Enter to use [0.1, 0.2, -0.05] as default:\n",
        default=[0.1, 0.2, -0.05],
        shape=(3,)
        )
    
    J = get_input(
        "Enter inertia matrix J [kgm^2] as a 3x3 list of lists\n"
        "or press Enter to use [[1.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 0.3]] as default:\n",
        default=[[1.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 0.3]],
        shape=(3, 3)
        )

    tau_d = get_input(
        "Enter disturbance torque tau_d [Nm] as a 3-element list, e.g. [0.001, 0.0015, 0.0],\n"
        "or press Enter to use [0, 0, 0] as default (torque-free):\n",
        default=[0, 0, 0],
        shape=(3,)
        )
    
    t_span = get_input(
        "Enter simulation time span [s] as a 2-element list\n"
        "or press Enter to use [0, 1000] as default:\n",
        default=[0, 1000],
        shape=(2,)
        )
    
    output_input = input(
        "Select attitude output format:\n"
        "  1 - Quaternion [qx, qy, qz, qs]\n"
        "  2 - Euler angles [yaw, pitch, roll] in degrees\n"
        "  3 - MRP [sigma1, sigma2, sigma3]\n"
        "Enter choice (1-3) or press Enter to use default (1 - Quaternion):\n"
        )
    output_options = {1: "Quaternion", 2: "Euler", 3: "MRP"}
    output_choice = int(output_input) if output_input else 1
    if output_choice not in output_options:
        print("Invalid choice, using Quaternion (default).\n")
        output_choice = 1
    
    t, att, w = run_attitude_propagation(input_format=input_choice, att0=att0, w0=w0, J=J, tau_d=tau_d, t_span=t_span, output_format=output_choice)
    print("Attitude propagation finished.")
    
    # Save data
    np.savetxt(os.path.join(rslt_dir, "t.csv"), t, delimiter=",")
    np.savetxt(os.path.join(rslt_dir, "att.csv"), att, delimiter=",")
    np.savetxt(os.path.join(rslt_dir, "w.csv"), w, delimiter=",")
    print(f"Simulation data saved in folder {rslt_dir}.")
    
    # Plot figures
    if output_choice == 1:
        fig_att = plot_quat(t, att)
        fig_att.savefig(os.path.join(fig_dir, "attitude_quat.pdf"), bbox_inches="tight")
    if output_choice == 2:
        fig_att = plot_euler(t, att)
        fig_att.savefig(os.path.join(fig_dir, "attitude_euler.pdf"), bbox_inches="tight")
    if output_choice == 3:
        fig_att = plot_mrp(t, att)
        fig_att.savefig(os.path.join(fig_dir, "attitude_mrp.pdf"), bbox_inches="tight")
    fig_w = plot_w(t, w)
    fig_w.savefig(os.path.join(fig_dir, "angular_velocity.pdf"), bbox_inches="tight")
    print(f"Figures saved in folder {fig_dir}.")
    plt.show()
    
if __name__ == "__main__":
    main()