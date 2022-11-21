import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, k, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        k (int): The degree of the spline fit.
            For this assignment, k should equal 3 (see documentation for
            scipy.interpolate.splrep)
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        t_smoothed (np.array [N]): Associated trajectory times
        traj_smoothed (np.array [N,7]): Smoothed trajectory
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
def compute_smoothed_traj(path, V_des, k, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        k (int): The degree of the spline fit.
            For this assignment, k should equal 3 (see documentation for
            scipy.interpolate.splrep)
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        t_smoothed (np.array [N]): Associated trajectory times
        traj_smoothed (np.array [N,7]): Smoothed trajectory
    Hint: Use splrep and splev from scipy.interpolate
    """
    assert(path and k > 2 and k < len(path))
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    path = np.array(path)
    t_nom = np.zeros(len(path))
    for i in range(1, len(path)):
        dist = np.linalg.norm(path[i]-path[i-1])
        t = dist / V_des
        t_nom[i] = t + t_nom[i-1]
    x, y = path.T
    t_smoothed = np.arange(t_nom[0], t_nom[-1], dt)
    n = len(t_smoothed)
    x_splines = scipy.interpolate.splrep(t_nom, x, k=k, s=alpha)
    y_splines = scipy.interpolate.splrep(t_nom, y, k=k, s=alpha)
    x_smooth = scipy.interpolate.splev(t_smoothed, x_splines, der=0)
    xd_smooth = scipy.interpolate.splev(t_smoothed, x_splines, der=1)
    xdd_smooth = scipy.interpolate.splev(t_smoothed, x_splines, der=2)
    y_smooth = scipy.interpolate.splev(t_smoothed, y_splines, der=0)
    yd_smooth = scipy.interpolate.splev(t_smoothed, y_splines, der=1)
    ydd_smooth = scipy.interpolate.splev(t_smoothed, y_splines, der=2)
    thetad_smooth = np.array([np.arctan2(yd_smooth[i],xd_smooth[i]) for i in range(n)])
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_smooth, y_smooth, thetad_smooth, xd_smooth, yd_smooth, xdd_smooth, ydd_smooth]).transpose()

    return t_smoothed, traj_smoothed    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return t_smoothed, traj_smoothed
