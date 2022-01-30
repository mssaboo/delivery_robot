import numpy as np
import scipy.interpolate


def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    path = np.array(path)
    N = path.shape[0]
    
    timedeltas = [0] + [np.linalg.norm(path[i+1] - path[i]) / V_des for i in range(N-1)]
    nominal_times = np.cumsum(np.array(timedeltas))

    new_len = int(np.ceil(nominal_times[-1] / dt)) + 1
    t_smoothed = [dt * i for i in range(new_len)]
    
    x = path[:, 0]
    y = path[:, 1]
    spl_x = scipy.interpolate.splrep(nominal_times, x, s=alpha)
    x_d = scipy.interpolate.splev(t_smoothed, spl_x)
    xd_d = scipy.interpolate.splev(t_smoothed, spl_x, 1)
    xdd_d = scipy.interpolate.splev(t_smoothed, spl_x, 2)

    spl_y = scipy.interpolate.splrep(nominal_times, y, s=alpha)
    y_d = scipy.interpolate.splev(t_smoothed, spl_y)
    yd_d = scipy.interpolate.splev(t_smoothed, spl_y, 1)
    ydd_d = scipy.interpolate.splev(t_smoothed, spl_y, 2)

    theta_d = np.arctan2(yd_d, xd_d)

    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return traj_smoothed, t_smoothed