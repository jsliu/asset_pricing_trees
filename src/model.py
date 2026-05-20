import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLars, ElasticNet, lars_path
from sklearn.base import BaseEstimator
from scipy.linalg import eigh

from .constants import Columns


MIN_EIG_VALUE = 1e-10
EPSILON = 1e-20


class TreeElastic(BaseEstimator):
    def __init__(self, mean_shrinkage: float = 0., ridge_lambda: float = 0.5,
                 k_min: int = 0, k_max: float = np.inf):
        self.mean_shrinkage = mean_shrinkage
        self.ridge_lambda = ridge_lambda
        self.base_model = LassoLars(alpha=1e-20, fit_intercept=False, random_state=42)
        self.k_min = k_min
        self.k_max = k_max
        self.feature_weights = None
        self.betas = None

    @staticmethod
    def decompose_covariance(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Decomposes input feature df into Eigen-values/vectors. Because the decomposition could return complex numbers,
        take only the real part of them.

        Parameters
        ----------
        feature_df:
            Input feature df of size (n_observation, n_features)

        Returns
        -------
            Tuple of Eigen Values and Eigen Vectors
        """
        # sigma = feature_df.cov()
        # eig_values, eig_vectors = eigh(sigma)
        # eig_values = eig_values[::-1]
        # eig_vectors = np.flip(eig_vectors, axis=1)
        # gamma = min(feature_df.shape[0], sum(eig_values > MIN_EIG_VALUE))
        # return eig_values[:gamma], eig_vectors[:, :gamma]


        # Convert to ndarray and center (to match pandas' cov(): division by (n-1))
        X = np.asarray(feature_df, dtype=np.float64)
        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 observations to compute a sample covariance.")

        Xc = X - X.mean(axis=0, keepdims=True)

        # Economy SVD of centered data
        # Xc = U * S * Vt   with S sorted descending by numpy default
        # Right singular vectors V are eigenvectors of covariance
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Eigenvalues of covariance: S^2 / (n-1)
        eig_values = (S ** 2) / (n_samples - 1)

        # Right singular vectors are the eigenvectors (as columns)
        eig_vectors = Vt.T  # shape: (n_features, r)

        # Filter by eigenvalue threshold (descending already)
        mask = eig_values > MIN_EIG_VALUE
        eig_values = np.real(eig_values[mask])
        eig_vectors = np.real(eig_vectors[:, mask])
        return eig_values, eig_vectors

    def process_input(self, feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        eig_values, eig_vectors = self.decompose_covariance(feature_df.fillna(0))
        sigma_tilde = eig_vectors @ np.diag(np.sqrt(eig_values)) @ eig_vectors.T

        mu_tilde = eig_vectors @  np.diag(1 / np.sqrt(eig_values)) @ eig_vectors.T
        mu_tilde = mu_tilde @ (
                feature_df.mean() + self.mean_shrinkage * np.nanmean(feature_df.values)
        ).values

        n_feats = sigma_tilde.shape[0]
        sigma_tilde = np.vstack([sigma_tilde, np.diag(np.array([np.sqrt(self.ridge_lambda)] * n_feats))])
        mu_tilde = np.hstack([mu_tilde, np.array([0] * n_feats)])

        return sigma_tilde, mu_tilde

    @staticmethod
    def get_feature_weights(feature_df: pd.DataFrame) -> np.ndarray:
        """
        Based on the position of a Node in the tree - calculate its weight as 1 / SQRT(2 ^ node-depth).

        Parameters
        ----------
        feature_df:
            Input feature df of size (n_observation, n_features)

        Returns
        -------
            Array with adjusting weight per feature
        """
        depth_per_col = np.array([
            int(c.replace(f"{Columns.port_col}{Columns.col_sep}", ""))
            for c in feature_df.columns.get_level_values(Columns.port_col)
        ])
        return pd.Series(1 / np.sqrt(2 ** depth_per_col), index=feature_df.columns)
        # return 1 / np.sqrt(2 ** depth_per_col)

    def fit(self, X: pd.DataFrame, y=None, *args, **kwargs) -> 'TreeElastic':
        feature_df = X.copy()
        
        # Feature weights
        self.feature_weights = self.get_feature_weights(feature_df)
        feature_df = feature_df.multiply(self.feature_weights)

        # Transform the returns into the model inputs
        input_x, input_y = self.process_input(feature_df)
        self.base_model.fit(input_x, input_y)

        # Adjust coefficients and normalize
        self.betas = self.base_model.coef_path_.T * self.feature_weights.to_numpy()

        # This forces the sum of betas being 0
        betas_tmp = self.betas.copy()
        betas_mask = betas_tmp != 0
        counts = betas_mask.sum(axis=1, keepdims=True)
        counts = np.where(counts == 0, 1, counts)
        row_means = (betas_tmp * betas_mask).sum(axis=1, keepdims=True) / counts
        self.betas = betas_tmp - row_means * betas_mask

        # This forces the sum of betas being 1
        # self.betas = (self.betas.T / (np.abs(np.sum(self.betas, axis=1)) + EPSILON)).T

        num_not_zero = np.sum(self.betas != 0, axis=1)
        mask = (num_not_zero >= self.k_min) & (num_not_zero <= self.k_max)
        if not mask.any():
            self.betas = np.zeros(self.betas.shape)
        else:
            self.betas = self.betas[mask]
        return self
    
    def predict(self, X: pd.DataFrame, y=None, *args, **kwargs) -> np.ndarray:
        # feature_df = X.copy()
        feature_df = X[self.feature_weights.index]
        feature_df = feature_df.multiply(self.feature_weights)
        return feature_df @ (self.betas / self.feature_weights.to_numpy()).T

    def score(self, X: pd.DataFrame, y=None, *args, **kwargs):
        sdf = self.predict(X)
        sharpe_values = np.mean(sdf, axis=0) / (np.std(sdf, axis=0) + EPSILON)
        return max(sharpe_values)
