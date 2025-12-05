import logging
import numpy as np

_logger = logging.getLogger(__name__)

class CategoricalNaiveBayes:
    """
    Categorical Naive Bayes classifier for categorical features.
    
    Suitable for features that are categorically distributed (discrete values).
    Each feature is assumed to be from a different categorical distribution.
    Categories should be encoded as integers 0, 1, 2, ... n-1
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize Categorical Naive Bayes.
        
        Args:
            alpha (float): Additive (Laplace) smoothing parameter
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.category_count_ = None
        self.n_categories_ = None
        
    def fit(self, X, y):
        """
        Fit Categorical Naive Bayes classifier.
        
        Args:
            X (np.ndarray): Training vectors, shape (n_samples, n_features)
                           Each feature contains categorical values (0, 1, 2, ...)
            y (np.ndarray): Target values, shape (n_samples,)
            
        Returns:
            self
        """
        _logger.info("Training Categorical Naive Bayes classifier...")
        
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Determine number of categories per feature
        self.n_categories_ = np.max(X, axis=0) + 1
        
        # Initialize class priors
        self.class_log_prior_ = np.zeros(n_classes)
        
        # Initialize category counts for each class and feature
        # Shape: (n_classes, n_features, max_categories)
        max_categories = int(np.max(self.n_categories_))
        self.category_count_ = []
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_samples_c = X_c.shape[0]
            
            # Class prior
            self.class_log_prior_[idx] = np.log(n_samples_c / X.shape[0])
            
            # Count categories for each feature
            feature_counts = []
            for feature_idx in range(n_features):
                n_cats = int(self.n_categories_[feature_idx])
                counts = np.zeros(n_cats)
                
                for cat in range(n_cats):
                    counts[cat] = np.sum(X_c[:, feature_idx] == cat)
                
                # Apply Laplace smoothing and convert to log probabilities
                counts = (counts + self.alpha) / (n_samples_c + self.alpha * n_cats)
                feature_counts.append(np.log(counts))
            
            self.category_count_.append(feature_counts)
        
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
        
        predictions = []
        for x in X:
            log_probs = []
            
            for class_idx in range(len(self.classes_)):
                # Start with class prior
                log_prob = self.class_log_prior_[class_idx]
                
                # Add log probabilities for each feature
                for feature_idx, feature_value in enumerate(x):
                    feature_value = int(feature_value)
                    log_prob += self.category_count_[class_idx][feature_idx][feature_value]
                
                log_probs.append(log_prob)
            
            predictions.append(self.classes_[np.argmax(log_probs)])
        
        return np.array(predictions)