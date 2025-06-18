from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update

        curr_state = self.state.copy()

        y_hat = np.dot(self.W, curr_state)
        u = np.array(u).reshape(-1, 1)
        error = np.array(q - y_hat).reshape(-1, 1)

        dx = self.A @ curr_state.reshape(-1, 1) + self.B @ u + self.L @ error

        self.state = (curr_state.reshape(-1, 1) + self.Tp * dx).flatten()


        # self.states.append(self.state.copy())

    def get_state(self):
        return self.state.flatten()
