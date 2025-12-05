import logging
import numpy as np

_logger = logging.getLogger(__name__)

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier implementation from scratch.
    
    Attributes:
        classes_ (np.ndarray): Unique class labels
        class_prior_ (np.ndarray): Prior probabilities for each class
        theta_ (np.ndarray): Mean of each feature per class
        sigma_ (np.ndarray): Variance of each feature per class
    """
    
    def __init__(self):
        """Initialize the Gaussian Naive Bayes classifier."""
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # means
        self.sigma_ = None  # variances
        
    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes according to X, y.
        
        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features)
            y (np.ndarray): Target values, shape (n_samples,)
            
        Returns:
            self: Returns the instance itself
        """
        _logger.info("Training Gaussian Naive Bayes classifier...")
        
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        
        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.sigma_[idx, :] = X_c.var(axis=0)
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]
            
        _logger.info(f"Training complete. Classes: {self.classes_}")
        return self
        
    def _calculate_likelihood(self, class_idx, x):
        """
        Calculate Gaussian probability density function.
        
        Args:
            class_idx (int): Index of the class
            x (np.ndarray): Feature vector
            
        Returns:
            float: Log likelihood
        """
        mean = self.theta_[class_idx]
        var = self.sigma_[class_idx]
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-9
        var_safe = var + epsilon
    
        log_likelihood = -0.5 * np.log(2 * np.pi * var_safe) - 0.5 * ((x - mean) ** 2) / var_safe
        
        return np.sum(log_likelihood)
        
    def predict(self, X):
        """
        Perform classification on test vectors X.
        
        Args:
            X (np.ndarray): Test vectors, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted target values, shape (n_samples,)
        """
        _logger.info(f"Making predictions on {X.shape[0]} samples...")
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
        
    def _predict_single(self, x):
        """Predict class for a single sample."""
        posteriors = []
        
        for idx, c in enumerate(self.classes_):
            prior = np.log(self.class_prior_[idx])
            likelihood = self._calculate_likelihood(idx, x)
            posterior = prior + likelihood
            posteriors.append(posterior)
            
        return self.classes_[np.argmax(posteriors)]