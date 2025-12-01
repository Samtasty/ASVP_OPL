import numpy as np
from scipy.special import expit

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    GaussianProcessRegressor = None
    WhiteKernel = None
    RBF = None
    Matern = None


class BernoulliBandit:
    def __init__(self, means: np.ndarray, rng: np.random.RandomState):
        self.means = np.asarray(means, dtype=float)
        self.n_arms = self.means.shape[0]
        self.best_mean = float(self.means.max())
        self.rng = rng

    def pull(self, arm: int) -> float:
        return float(self.rng.rand() < self.means[arm])


class ContextualBandit:
    """Simple Bernoulli contextual bandit with sigmoid reward model."""

    def __init__(self, theta: np.ndarray, rng: np.random.RandomState):
        self.theta = np.asarray(theta, dtype=float)  # (n_arms, d)
        self.n_arms, self.d = self.theta.shape
        self.rng = rng

    def sample_context(self) -> np.ndarray:
        return self.rng.randn(self.d)

    def reward_probs(self, context: np.ndarray) -> np.ndarray:
        logits = self.theta @ context
        return expit(logits)

    def pull(self, arm: int, context: np.ndarray) -> float:
        p = self.reward_probs(context)[arm]
        return float(self.rng.rand() < p)

    def best_mean(self, context: np.ndarray) -> float:
        return float(self.reward_probs(context).max())


class SmoothGPBandit:
    """Smooth contextual bandit: each arm's mean reward is a GP draw over contexts."""

    def __init__(
        self,
        d: int,
        n_arms: int,
        rng: np.random.RandomState,
        kernel_type: str = "matern",
        length_scale: float = 0.4,
        nu: float = 3.0,
        noise: float = 0.5,
        grid_size: int = 200,
    ):
        if GaussianProcessRegressor is None:
            raise ImportError("scikit-learn is required for SmoothGPBandit.")

        self.d = d
        self.n_arms = n_arms
        self.rng = rng
        self.noise = noise
        if kernel_type == "matern":
            self.kernel = Matern(length_scale=length_scale, nu=nu)
        elif kernel_type == "rbf":
            self.kernel = RBF(length_scale=length_scale)
        else:
            raise ValueError("Unknown kernel_type; use 'matern' or 'rbf'.")

        self.X_grid = rng.rand(grid_size, d)

        self.f = []
        for _ in range(n_arms):
            gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, random_state=rng)
            f_vals = gp.sample_y(self.X_grid, n_samples=1, random_state=rng).flatten()
            self.f.append(f_vals)

        K_grid = self.kernel(self.X_grid)
        self.K_inv = np.linalg.pinv(K_grid + 1e-6 * np.eye(grid_size))

    def sample_context(self) -> np.ndarray:
        return self.rng.rand(self.d)

    def reward_mean(self, context: np.ndarray, arm: int) -> float:
        k_vec = self.kernel(context.reshape(1, -1), self.X_grid).reshape(-1)
        return float(k_vec @ self.K_inv @ self.f[arm])

    def pull(self, arm: int, context: np.ndarray) -> float:
        mean = self.reward_mean(context, arm)
        return float(mean + self.noise * self.rng.randn())

    def best_mean(self, context: np.ndarray) -> float:
        means = [self.reward_mean(context, a) for a in range(self.n_arms)]
        return float(np.max(means))


class GPContextualBandit:
    """Contextual bandit where each arm's reward is a GP draw, evaluated via NN interpolation."""

    def __init__(
        self,
        d: int,
        n_arms: int,
        rng: np.random.RandomState,
        kernel_type: str = "matern",
        length_scale: float = 0.5,
        nu: float = 1.5,
        noise: float = 0.1,
        grid_size: int = 400,
    ):
        if GaussianProcessRegressor is None:
            raise ImportError("scikit-learn is required for GPContextualBandit.")
        self.d = d
        self.n_arms = n_arms
        self.rng = rng
        self.noise = noise

        if kernel_type == "rbf":
            self.kernel = RBF(length_scale=length_scale)
        elif kernel_type == "matern":
            self.kernel = Matern(length_scale=length_scale, nu=nu)
        else:
            raise ValueError("Unknown kernel_type; use 'rbf' or 'matern'.")

        self.X_grid = rng.rand(grid_size, d)
        self.gp_samples = []
        for _ in range(n_arms):
            gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-8, random_state=rng)
            f_vals = gp.sample_y(self.X_grid, n_samples=1, random_state=rng).flatten()
            f_vals += 0.1 * rng.randn()  # arm-specific shift
            f_vals += 0.05 * rng.randn(len(self.X_grid))  # jitter per grid point
            self.gp_samples.append(f_vals)

    def sample_context(self) -> np.ndarray:
        return self.rng.rand(self.d)

    def reward_mean(self, context: np.ndarray, arm: int) -> float:
        idx = int(np.argmin(np.linalg.norm(self.X_grid - context, axis=1)))
        return float(self.gp_samples[arm][idx])

    def pull(self, arm: int, context: np.ndarray) -> float:
        mean = self.reward_mean(context, arm)
        return float(mean + self.noise * self.rng.randn())

    def best_mean(self, context: np.ndarray) -> float:
        means = [self.reward_mean(context, a) for a in range(self.n_arms)]
        return float(np.max(means))
