import numpy as np

class Model:
    def __init__(self, grid_shape, feature_type=0):
        self.grid_shape = grid_shape
        self.feature_type = feature_type
        self.thetas = self.init_thetas()

    def init_thetas(self):
        thetas = None
        if self.feature_type == 0:
            thetas = (1 * np.random.randn(self.grid_shape[0] * self.grid_shape[1]) + 0).reshape(-1, 1)
        elif self.feature_type == 1:
            thetas = (1 * np.random.randn(self.grid_shape[0] * self.grid_shape[1] * 4) + 0).reshape(-1, 1)
        elif self.feature_type == 2:
            thetas = (1 * np.random.randn(4) + 0).reshape(-1, 1)
        elif self.feature_type == 3:
            thetas = (1 * np.random.randn(3) + 0).reshape(-1, 1)
        return thetas

    def sa2x(self, s, a=None):
        x = None
        if self.feature_type == 0:
            x = np.zeros((self.grid_shape[0] * self.grid_shape[1], 1))
            pos = s[0] * self.grid_shape[0] + s[1]
            x[pos, 0] = 1
        elif self.feature_type == 1:
            x = np.zeros((self.grid_shape[0] * self.grid_shape[1], 4))
            pos = s[0] * self.grid_shape[0] + s[1]
            x[pos, a] = 1
            x = x.reshape(-1,1)
        elif self.feature_type == 2:
            x = np.array([s[0]-1.5, s[1]-1.5, s[0]*s[1]-4.5, 1]).reshape(-1,1)
        elif self.feature_type == 3:
            x = np.array([s[0]+1, s[1]+1, 1]).reshape(-1,1)
        return x

    def predict(self, s, a=None):
        x = None
        if a is None:
            x = self.sa2x(s)
        elif a is not None:
            x = self.sa2x(s, a)
        return float(np.dot(self.thetas.T, x))