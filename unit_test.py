import unittest
import numpy as np

from ode_methods import ode_solve, newton_method
# TODO: Use pytest later

class TestODESolver(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestODESolver, self).__init__(*args, **kwargs)
        self.f = lambda x, t: x
        self.f_u = lambda x, t: np.eye(x.shape[0])
        self.u_closed = np.exp(np.arange(0, 0.2, 0.02))
        self.u0 = np.ones(1)

    def test_euler_forward(self):
        """
        Test Euler forward method
        """
        u_sol = ode_solve(self.f, self.u0, dt=0.02, num_step=10, method='euler_forward').reshape(-1)
        max_error = np.max(np.abs(u_sol/self.u_closed - 1))
        self.assertLess(max_error, 0.01, 'Produce wrong solution for Euler forward method')

    def test_midpoint_forward(self):
        """
        Test midpoint method
        """
        u_sol = ode_solve(self.f, self.u0, dt=0.02, num_step=10, method='midpoint').reshape(-1)
        max_error = np.max(np.abs(u_sol/self.u_closed - 1))
        self.assertLess(max_error, 0.01, 'Produce wrong solution for mid-point method')

    def test_newton_method(self):
        """
        Test Newton (iterative) method for solving x: g(x) = 0
        """
        x0 = np.ones(1)
        g = lambda x: 2-x**2
        dg = lambda x: -2*x.reshape(x.shape[0], x.shape[0])
        newton_method(x0, g, dg, num_iter=100)
        error = np.abs(x0[0] - np.sqrt(2))
        self.assertLess(error, 1e-3, 'Produce wrong solution for x^2 = 2 using Newton method')
    
    def test_euler_backrward(self):
        """
        Test Euler backward method
        """
        u_sol = ode_solve(self.f, self.u0, dt=0.02, num_step=10, method='euler_backward', f_u=self.f_u).reshape(-1)
        max_error = np.max(np.abs(u_sol/self.u_closed - 1))
        self.assertLess(max_error, 0.01, 'Produce wrong solution for euler backward method')

    def test_trapezoid(self):
        """
        Test (implicit) trapezoid method
        """
        u_sol = ode_solve(self.f, self.u0, dt=0.02, num_step=10, method='trapezoid', f_u=self.f_u).reshape(-1)
        max_error = np.max(np.abs(u_sol/self.u_closed - 1))
        self.assertLess(max_error, 0.001, 'Produce wrong solution for trapezoid method')

if __name__ == '__main__':
    unittest.main()