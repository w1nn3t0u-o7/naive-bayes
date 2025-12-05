import logging
import numpy as np

_logger = logging.getLogger(__name__)

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes classifier for binary/boolean features.
    
    Suitable for text classification with binary word occurrence.
    """
    
    def __init__(self, alpha=1.0, binarize=0.0):
        """
        Initialize Bernoulli Naive Bayes.
        
        Args:
            alpha (float): Additive (Laplace) smoothing parameter
            binarize (float): Threshold for binarizing features (None for no binarization)
        """
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        
    def fit(self, X, y):
        """
        Fit Bernoulli Naive Bayes classifier.
        
        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features)
                           Binary features (0/1)
            y (np.ndarray): Target values, shape (n_samples,)
            
        Returns:
            self
        """
        X = np.array(X)
        y = np.array(y)
        
        # Binarize if threshold provided
        if self.binarize is not None:
            X = (X > self.binarize).astype(int)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features, 2))
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            
            # Class prior
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])
            
            # Feature probabilities with smoothing
            # P(feature=1|class) and P(feature=0|class)
            feature_count = X_c.sum(axis=0)
            n_samples_c = X_c.shape[0]
            
            # P(feature=1|class)
            self.feature_log_prob_[idx, :, 1] = np.log(
                (feature_count + self.alpha) / (n_samples_c + 2 * self.alpha)
            )
            
            # P(feature=0|class)
            self.feature_log_prob_[idx, :, 0] = np.log(
                (n_samples_c - feature_count + self.alpha) / (n_samples_c + 2 * self.alpha)
            )
        
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
        
        # Binarize if threshold provided
        if self.binarize is not None:
            X = (X > self.binarize).astype(int)
        
        predictions = []
        for x in X:
            log_probs = []
            for idx in range(len(self.classes_)):
                # Start with class prior
                log_prob = self.class_log_prior_[idx]
                
                # Add feature probabilities
                for feature_idx, feature_value in enumerate(x):
                    log_prob += self.feature_log_prob_[idx, feature_idx, int(feature_value)]
                
                log_probs.append(log_prob)
            
            predictions.append(self.classes_[np.argmax(log_probs)])
        
        return np.array(predictions)