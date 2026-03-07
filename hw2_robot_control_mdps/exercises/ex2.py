import numpy as np


def generate_quintic_spline_waypoints(start, end, num_points):

    """
    TODO:

    Steps:
    1. Generate `num_points` linearly spaced time steps `s` between 0 and 1.
    2. Apply the quintic time scaling polynomial function which can be found in the slides to get `f_s`.
    3. Interpolate between `start` and `end` using `start + (end - start) * f_s`.
    
    Args:
        start (np.ndarray): Starting waypoint.
        end (np.ndarray): Ending waypoint.
        num_points (int): Number of points in the trajectory.
        
    Returns:
        np.ndarray: Generated waypoints.
    """
    num_points = np.linspace(0,1,num_points)

    # From Lecture: Trajectory Waypoint Generation: Quintic Splines

    f_s = 10 * num_points**(3) - 15 * num_points**(4) + 6 * num_points**(5)
    q = start + np.outer(f_s,end-start)

    return q


def pid_control(tracking_error_history, timestep, Kp=150.0, Ki=0.0, Kd=0.01):
    """
    TODO:
    Compute the PID control signal based on the tracking error history.
    
    Steps:
    1. The Proportional (P) term is the most recent error.
    2. The Integral (I) term is the sum of all past errors, multiplied by the simulation timestep.
    3. The Derivative (D) term is the rate of change of the error (difference between the last two errors divided by the timestep).
       If there is only one error in history, the D term should be zero.
    4. Compute the final control signal: Kp * P + Ki * I + Kd * D.
    
    Args:
        tracking_error_history (np.ndarray): History of tracking errors.
        timestep (float): Simulation timestep.
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        
    Returns:
        np.ndarray: Control signal.
    """
    if len(tracking_error_history) <= 1:
        return 0
    else:
        return Kp*tracking_error_history[-1] + Ki*np.sum(tracking_error_history, axis=0) + Kd*(tracking_error_history[-1]-tracking_error_history[-2])/timestep

# Theoretical questions

# To get a feeling for the choice of the PID gains, you will analyze how their choice influences the behavior of the waypoint tracking. Test different settings of the gains to be able to answer the following:

#     1. If you keep increasing K P , what issue arises when tracking the waypoints?
#       -> system resonds to aggressively to errors and will cause overshoots. it can also lead to oscillations which may grow unstable
#     2. How does K D mitigate the effect you saw above when increasing K P ?
#       -> K-D acts as a damper, and counters the K_P proportionally to how fast the system is moving to/away from a setpoint -> reduces overshoot and settles oscillations faster
#     3. In what scenarios is a non-zero K I needed for the controller to perform well?
#       -> when there is a persistent steady state error, like a bias from gravity, external forces or model mismatches that preven (P+D) to reach the setpoint alone

# There is no need to show these behavior changes in the video and you can just write down your answers in the video. Or say them out loud. The theoretical questions require only short and direct answers. Each question is expected to have a 1-sentence answer.