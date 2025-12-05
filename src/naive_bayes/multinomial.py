import logging
import numpy as np

_logger = logging.getLogger(__name__)

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier for discrete/count data.
    
    Suitable for text classification with word counts.
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize Multinomial Naive Bayes.
        
        Args:
            alpha (float): Additive (Laplace) smoothing parameter (0 for no smoothing)
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        
    def fit(self, X, y):
        """
        Fit Multinomial Naive Bayes classifier.
        
        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features)
                           Contains count data (non-negative integers)
            y (np.ndarray): Target values, shape (n_samples,)
            
        Returns:
            self
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            
            # Class prior: P(class)
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])
            
            # Feature probabilities with Laplace smoothing
            # P(feature|class) = (count of feature in class + alpha) / (total count in class + alpha * n_features)
            feature_count = X_c.sum(axis=0)
            total_count = feature_count.sum()
            
            self.feature_log_prob_[idx, :] = np.log(
                (feature_count + self.alpha) / (total_count + self.alpha * n_features)
            )
        
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
        X = np.array(X)
        log_prob = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(log_prob, axis=1)]