"""
Gaussian Naive Bayes classifier implementation.

Includes Gaussian, Multinomial, Bernoulli, Categorical, and Complement Naive Bayes variants.
"""

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

