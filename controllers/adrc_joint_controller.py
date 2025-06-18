import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        self.u_last = 0.

        A = np.array([[0., 1.0 , 0.],
                      [0., 0., 1.0],
                      [0., 0., 0.]])

        B = [[0],
             [self.b],
             [0]]

        L = np.array([[3 * p],
                      [3 * p**2],
                      [p**3]])

        W = [[1.0, 0., 0.]]

        self.eso = ESO(A, B, W, L, q0, Tp)

        # self.model = ManiuplatorModel(Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        self.eso.set_B(np.array([[0], [b], [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        # (q1, q2, q1_dot, q2_dot) = x

        q = x[0]

        u = self.u_last
        self.eso.update(q, u)

        x_hat = self.eso.get_state()
        q_hat = x_hat[0]
        q_dot_hat = x_hat[1]
        f_hat = x_hat[2]

        v = q_d_ddot + self.kd * (q_d_dot - q_dot_hat) + self.kp * (q_d - q_hat)
        u = 0 if self.b == 0 else (v - f_hat) / self.b

        self.u_last = u

        # print(self.b)
        return u
