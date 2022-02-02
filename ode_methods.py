from re import X
import numpy as np

# Newton method (Helper for backward): Solve for x: g(x) = 0
def newton_method(x0, g, dg, num_iter=10):
    x = x0
    for _ in range(num_iter):
        x -= np.linalg.solve(dg(x), g(x))
    return x

## All ODE methods differs only in transition from one timestep to the next
def next_ode_step(f, u, h, t, method='euler_forward', f_u=None):
    # Euler forward: u(n+1) = u(n) + h*f(u(n), t(n))
    if method == 'euler_forward':
        return u + h*f(u, t)
    # Midpoint forward: u(n+1) = u(n) + h*f(u(n)+h/2*f(u(n),t(n)), t(n)+h/2)
    if method == 'midpoint':
        u_mid = u + h/2*f(u, t)
        return u + h*f(u_mid, t+h/2)
    # Euler backward: u(n+1) = u(n) + h*f(u(n+1), t(n+1))
    if method == 'euler_backward':
        # Solve implicitly x: 0 = g(x) = x - u(n) - h*f(x, t(n+1))
        t_next = t + h
        g = lambda x: x - u - h * f(x, t_next)
        dg = lambda x: np.eye(x.shape[0]) - h * f_u(x, t_next)

        return newton_method(np.zeros(u.shape[0]), g, dg)
    # Trapezoid: u(n+1) = u(n) + h/2* [f(u(n), t(n)) + f(u(n+1), t(n+1))]
    if method == 'trapezoid':
        # Solve implicitly x: 0 = g(x) = x - u(n) - h/2*[f(u, t) + f(x, t_next)]
        t_next = t + h
        c0 = -u - h/2*f(u, t)
        g = lambda x: x + c0 - h/2*f(x, t_next)
        dg = lambda x: np.eye(x.shape[0]) - h/2*f_u(x, t_next)

        return newton_method(np.zeros(u.shape[0]), g, dg)
        
## General ODE solver
def ode_solve(f, u0, dt=0.02, num_step=10, method='euler_forward', f_u=None):
    u = np.zeros((num_step, u0.shape[0]), dtype=float)
    u[0] = u0
    for i in range(0, num_step-1):
        u[i+1] = next_ode_step(f, u[i], dt, dt*i, method, f_u)
    return u
