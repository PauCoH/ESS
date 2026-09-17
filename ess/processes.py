import numpy as np
import control


class PIDController:

    def __init__(self, Kp, b, Ti, Ts, Td, c):
        """Initialize PID coefficients and zero controller state.

        Kp is the proportional gain; Ti, Td and Ts use the same time unit.
        b and c weight the reference in proportional and derivative actions.
        The integral uses the trapezoidal rule; the derivative uses a difference."""
        self.Kp, self.b, self.Ti, self.Ts, self.Td, self.c = Kp, b, Ti, Ts, Td, c
        self.prev_r, self.prev_y, self.prev_integrator = 0, 0, 0
        self.A = (self.Kp * self.Ts) / (2 * self.Ti) if self.Ti != 0 else 0
        self.B = (self.Kp * self.Td) / self.Ts

    def update(self, r_k, y_k):
        """Compute the control action from reference r_k and measurement y_k, then update PID state."""
        prop = self.Kp * (self.b * r_k - y_k)
        # Carry the accumulated integral between samples; initialize a new PID for each trajectory.
        inte = self.A * (r_k + self.prev_r - y_k - self.prev_y) + self.prev_integrator
        der = self.B * (self.c * r_k - self.c * self.prev_r - y_k + self.prev_y)
        u_k = prop + inte + der
        self.prev_r, self.prev_y, self.prev_integrator = r_k, y_k, inte
        return u_k


def simulate_cstr(actuation_matrix, Kv, Kt, sampling_time, x0_u=None, x0_d=None):
    """Advance the linear CSTR control and disturbance paths by one interval.

    actuation_matrix has rows [u_previous, u_current], [dQ_previous, dQ_current]
    and [dCAi_previous, dCAi_current]. Kv scales the control-valve action and
    Kt scales the combined concentration response. sampling_time is in minutes.
    Carry x0_u and x0_d between calls to preserve the two process states.

    Return the interval output samples and the final state of each path."""
    u_prev, u_curr = actuation_matrix[0]
    dQ_prev, dQ_curr = actuation_matrix[1]
    disturbance_prev, disturbance_curr = actuation_matrix[2]


    # Control path includes the inverse-response zero.
    num_u = [0.0085 * -0.38, 0.0085]
    den_u = [0.56 * 0.31, 0.56 + 0.31, 1]
    Pu = control.TransferFunction(num_u, den_u)


    # The concentration disturbance has its own second-order dynamics.
    num_d = [0.0886]
    den_d = [0.43**2, 2 * 0.43, 1]
    Pd = control.TransferFunction(num_d, den_d)

    ss_u = control.tf2ss(Pu)
    ss_d = control.tf2ss(Pd)

    t = np.linspace(0, sampling_time, 2)
    x0_u = np.zeros(ss_u.A.shape[0]) if x0_u is None else x0_u
    x0_d = np.zeros(ss_d.A.shape[0]) if x0_d is None else x0_d

    # Combine flow disturbance with the valve-scaled controller action.
    process_input = np.array([dQ_prev + u_prev * Kv, dQ_curr + u_curr * Kv])
    disturbance_input = np.array([disturbance_prev, disturbance_curr])

    _, y_out_u, x_out_u = control.forced_response(ss_u, T=t, U=process_input, X0=x0_u, return_x=True)
    _, y_out_d, x_out_d = control.forced_response(ss_d, T=t, U=disturbance_input, X0=x0_d, return_x=True)

    # Apply transmitter scaling after summing the two concentration responses.
    y_out = (y_out_u + y_out_d) * Kt
    return y_out.tolist(), x_out_u[:, -1], x_out_d[:, -1]


class CSTRPlant:
    def __init__(self, sample_time):
        """Initialize zero process states and preceding control/disturbance samples."""
        self.sample_time = sample_time
        self.xu = self.xd = None
        self.previous = (0.0, 0.0, 0.0)

    def step(self, action, dq, dcai):
        """Advance the original two-point simulation and retain its final states."""
        current = (float(action), float(dq), float(dcai))
        matrix = list(zip(self.previous, current))
        output, self.xu, self.xd = simulate_cstr(matrix, 0.8, 50, self.sample_time, self.xu, self.xd)
        self.previous = current
        return float(output[-1])
