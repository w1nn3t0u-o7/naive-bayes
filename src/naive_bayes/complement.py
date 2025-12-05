import logging
import numpy as np

_logger = logging.getLogger(__name__)

class ComplementNaiveBayes:
    """
    Complement Naive Bayes classifier.
    
    Particularly suited for imbalanced datasets. Uses statistics from
    the complement of each class to compute weights.
    Similar to Multinomial NB but more stable for imbalanced data.
    """
    
    def __init__(self, alpha=1.0, norm=False):
        """
        Initialize Complement Naive Bayes.
        
        Args:
            alpha (float): Additive (Laplace) smoothing parameter
            norm (bool): Whether to perform a second normalization
        """
        self.alpha = alpha
        self.norm = norm
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        
    def fit(self, X, y):
        """
        Fit Complement Naive Bayes classifier.
        
        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features)
                           Contains count data (non-negative)
            y (np.ndarray): Target values, shape (n_samples,)
            
        Returns:
            self
        """
        _logger.info("Training Complement Naive Bayes classifier...")
        
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        # Calculate complement statistics for each class
        for idx, c in enumerate(self.classes_):
            # Class prior
            n_samples_c = np.sum(y == c)
            self.class_log_prior_[idx] = np.log(n_samples_c / X.shape[0])
            
            # Complement: all samples NOT in class c
            X_complement = X[y != c]
            
            # Count features in complement with smoothing
            complement_count = X_complement.sum(axis=0) + self.alpha
            total_count = complement_count.sum()
            
            # Compute log probabilities (inverse because it's complement)
            self.feature_log_prob_[idx, :] = np.log(complement_count / total_count)
            
            # Optional normalization
            if self.norm:
                norm_factor = np.sum(self.feature_log_prob_[idx, :])
                self.feature_log_prob_[idx, :] /= norm_factor
        
        # Complement NB uses negative weights
        self.feature_log_prob_ = -self.feature_log_prob_
        
        _logger.info(f"Training complete. Classes: {self.classes_}")
        return self
        
    def predict(self, X):
        """
        Perform classification on test vectors X.
        
        Args:
            X (np.ndarray): Test vectors, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted target values
        """
        _logger.info(f"Making predictions on {X.shape[0]} samples...")
        X = np.array(X)
        
        # Compute log probabilities for all classes
        log_prob = X @ self.feature_log_prob_.T + self.class_log_prior_
        
        return self.classes_[np.argmax(log_prob, axis=1)]