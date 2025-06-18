import numpy as np

from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.model1 = ManiuplatorModel(Tp, m3=0.1, r3=0.05)
        self.model2 = ManiuplatorModel(Tp, m3=0.01, r3=0.01)
        self.model3 = ManiuplatorModel(Tp, m3=1.0, r3=0.3)
        self.models = [self.model1, self.model2, self.model3]
        self.i = 0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        errors = []
        for model in self.models:
            errors.append(np.linalg.norm(x-model.predict(x)))

        self.i = np.argmin(errors)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        print(f"Best model: {self.i}")
        q = x[:2]
        q_dot = x[2:]
        # v = q_r_ddot # TODO: add feedback
        Kp = np.diag([20, 20])
        Kd = np.diag([5, 5])
        v = q_r_ddot - Kd @ (q_dot - q_r_dot) - Kp @ (q-q_r)

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
