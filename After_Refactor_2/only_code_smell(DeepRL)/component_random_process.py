import numpy as np

class RandomProcess:
    def reset_states(self) -> None:
        pass

class GaussianProcess(RandomProcess):
    def __init__(self, size: tuple[int, ...], std: float) -> None:
        self.size = size
        self.std = std

    def sample(self) -> np.ndarray:
        return np.random.randn(*self.size) * self.std

class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size: tuple[int, ...], std: float, theta: float = .15, dt: float = 1e-2, x0: np.ndarray = None) -> None:
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self) -> np.ndarray:
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self) -> None:
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class RandomProcess:
    def reset_states(self) -> None:
        pass

class GaussianProcess(RandomProcess):
    def __init__(self, size: tuple[int, ...], std: float) -> None:
        self.size = size
        self.std = std

    def sample(self) -> np.ndarray:
        return np.random.randn(*self.size) * self.std

class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size: tuple[int, ...], std: float, theta: float = .15, dt: float = 1e-2, x0: np.ndarray = None) -> None:
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self) -> np.ndarray:
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self) -> None:
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class RandomProcess:
    def reset_states(self) -> None:
        pass

class GaussianProcess(RandomProcess):
    def __init__(self, size: tuple[int, ...], std: float) -> None:
        self.size = size
        self.std = std

    def sample(self) -> np.ndarray:
        return np.random.randn(*self.size) * self.std

class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size: tuple[int, ...], std: float, theta: float = .15, dt: float = 1e-2, x0: np.ndarray = None) -> None:
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self) -> np.ndarray:
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self) -> None:
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)