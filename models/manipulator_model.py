import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=0.1, r3=0.05):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.01
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        self.d1 = self.l1 / 2
        self.d2 = self.l2 / 2

        self.alpha = (self.m1 * self.d1 ** 2 + self.I_1 + self.m2 *
                      (self.l1 ** 2 + self.d2 ** 2) + self.I_2 + self.m3 *
                      (self.l1 ** 2 + self.l2 ** 2) + self.I_3)

        self.beta = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2
        self.gamma = self.m2 * self.d2 ** 2 + self.I_2 + self.m3 * self.l2 ** 2 + self.I_3

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        cq2 = np.cos(q2)

        M_mat = np.array([
            [self.alpha+2*self.beta*cq2, self.gamma+self.beta*cq2],
            [self.gamma+self.beta*cq2, self.gamma]
            ])
        return M_mat

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        sq2 = np.sin(q2)

        C_mat = np.array([
            [-1*self.beta*sq2*q2_dot, -1*self.beta*sq2*(q1_dot+q2_dot)],
            [self.beta*sq2*q1_dot, 0]
            ])
        return C_mat

    def predict(self, x):
        q1, q2, q1_dot, q2_dot = x
        M = self.M(x)
        C = self.C(x)

        q_dot = np.array([q1_dot, q2_dot])

        q_ddot = np.linalg.inv(M) @ (-C @ q_dot[:, np.newaxis])

        pred = x + np.hstack((q_dot, q_ddot.flatten())) * self.Tp

        return pred
