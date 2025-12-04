"""
Comprehensive tests for Naive Bayes implementations.

This test suite compares custom implementations with scikit-learn
on multiple datasets.

Run with: pytest tests/test_naive_bayes.py -v -s
"""

import pytest
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import (
    GaussianNB, 
    MultinomialNB, 
    BernoulliNB, 
    CategoricalNB, 
    ComplementNB
)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer

from naive_bayes import (
    GaussianNaiveBayes,
    MultinomialNaiveBayes,
    BernoulliNaiveBayes,
    CategoricalNaiveBayes,
    ComplementNaiveBayes,
)

__author__ = "jmarchew"
__copyright__ = "jmarchew"
__license__ = "MIT"

@pytest.fixture(scope="module")
def iris_data():
    """Iris dataset - continuous features, 3 classes."""
    data = datasets.load_iris()
    return train_test_split(data.data, data.target, test_size=0.3, random_state=42)


@pytest.fixture(scope="module")
def wine_data():
    """Wine dataset - continuous features, 3 classes."""
    data = datasets.load_wine()
    return train_test_split(data.data, data.target, test_size=0.3, random_state=42)


@pytest.fixture(scope="module")
def breast_cancer_data():
    """Breast cancer dataset - continuous features, 2 classes."""
    data = datasets.load_breast_cancer()
    return train_test_split(data.data, data.target, test_size=0.3, random_state=42)


@pytest.fixture(scope="module")
def digits_data():
    """Digits dataset - discrete pixel values, 10 classes."""
    data = datasets.load_digits()
    return train_test_split(data.data, data.target, test_size=0.3, random_state=42)

class TestBasicFunctionality:
    """Basic checks for all classifiers."""
    
    def test_gaussian_initialization(self):
        """Test that GaussianNB initializes correctly."""
        model = GaussianNaiveBayes()
        assert model.classes_ is None
        assert model.theta_ is None
        assert model.sigma_ is None
        assert model.class_prior_ is None
    
    def test_multinomial_initialization(self):
        """Test MultinomialNB initialization."""
        model = MultinomialNaiveBayes(alpha=1.0)
        assert model.alpha == 1.0
        assert model.classes_ is None
    
    def test_bernoulli_initialization(self):
        """Test BernoulliNB initialization."""
        model = BernoulliNaiveBayes(alpha=1.0, binarize=0.0)
        assert model.alpha == 1.0
        assert model.binarize == 0.0
    
    def test_categorical_initialization(self):
        """Test CategoricalNB initialization."""
        model = CategoricalNaiveBayes(alpha=1.0)
        assert model.alpha == 1.0
    
    def test_complement_initialization(self):
        """Test ComplementNB initialization."""
        model = ComplementNaiveBayes(alpha=1.0, norm=False)
        assert model.alpha == 1.0
        assert model.norm == False
    
    @pytest.mark.parametrize("ModelClass", [
        GaussianNaiveBayes,
        MultinomialNaiveBayes,
        BernoulliNaiveBayes,
        CategoricalNaiveBayes,
        ComplementNaiveBayes,
    ])
    def test_fit_returns_self(self, ModelClass, iris_data):
        """Test that fit() returns self for method chaining."""
        X_train, X_test, y_train, y_test = iris_data
        
        if ModelClass in [MultinomialNaiveBayes, ComplementNaiveBayes]:
            X_train = np.abs(X_train)
        elif ModelClass == CategoricalNaiveBayes:
            discretizer = KBinsDiscretizer(n_bins=5, 
                                           encode='ordinal', 
                                           strategy='uniform', 
                                           quantile_method='averaged_inverted_cdf')
            X_train = discretizer.fit_transform(X_train)
        
        model = ModelClass()
        result = model.fit(X_train, y_train)
        assert result is model

def test_all_algorithms_comprehensive_comparison(iris_data, wine_data, breast_cancer_data, digits_data):
    """
    Comprehensive comparison of all Naive Bayes implementations vs scikit-learn.
    
    This single test demonstrates completion of the scikit-learn comparison requirement
    by testing all 5 algorithm variants on all 4 datasets where applicable.
    
    Outputs a detailed comparison table for documentation purposes.
    """
    
    results = []
    
    # Prepare all datasets
    datasets = {
        "Iris": iris_data,
        "Wine": wine_data,
        "Breast Cancer": breast_cancer_data,
        "Digits": digits_data,
    }
    
    print("\nTesting Gaussian Naive Bayes...")
    for dataset_name, data in datasets.items():
        X_train, X_test, y_train, y_test = data
        
        custom_acc = accuracy_score(
            y_test, 
            GaussianNaiveBayes().fit(X_train, y_train).predict(X_test)
        )
        sklearn_acc = accuracy_score(
            y_test,
            GaussianNB().fit(X_train, y_train).predict(X_test)
        )
        
        results.append(("Gaussian", dataset_name, custom_acc, sklearn_acc, 0.05))
    
    print("Testing Multinomial Naive Bayes...")
    for dataset_name, data in datasets.items():
        X_train, X_test, y_train, y_test = data
        
        # Make data non-negative
        X_train_abs = np.abs(X_train)
        X_test_abs = np.abs(X_test)
        
        custom_acc = accuracy_score(
            y_test,
            MultinomialNaiveBayes(alpha=1.0).fit(X_train_abs, y_train).predict(X_test_abs)
        )
        sklearn_acc = accuracy_score(
            y_test,
            MultinomialNB(alpha=1.0).fit(X_train_abs, y_train).predict(X_test_abs)
        )
        
        results.append(("Multinomial", dataset_name, custom_acc, sklearn_acc, 0.15))
    
    print("Testing Bernoulli Naive Bayes...")
    for dataset_name, data in datasets.items():
        X_train, X_test, y_train, y_test = data
        
        custom_acc = accuracy_score(
            y_test,
            BernoulliNaiveBayes(alpha=1.0, binarize=0.0).fit(X_train, y_train).predict(X_test)
        )
        sklearn_acc = accuracy_score(
            y_test,
            BernoulliNB(alpha=1.0, binarize=0.0).fit(X_train, y_train).predict(X_test)
        )
        
        results.append(("Bernoulli", dataset_name, custom_acc, sklearn_acc, 0.15))
    
    print("Testing Categorical Naive Bayes...")
    for dataset_name, data in datasets.items():
        X_train, X_test, y_train, y_test = data
        
        # Discretize continuous features
        discretizer = KBinsDiscretizer(n_bins=5, 
                                       encode='ordinal', 
                                       strategy='uniform', 
                                       quantile_method='averaged_inverted_cdf')
        X_train_disc = discretizer.fit_transform(X_train)
        X_test_disc = discretizer.transform(X_test)
        
        custom_acc = accuracy_score(
            y_test,
            CategoricalNaiveBayes(alpha=1.0).fit(X_train_disc, y_train).predict(X_test_disc)
        )
        sklearn_acc = accuracy_score(
            y_test,
            CategoricalNB(alpha=1.0).fit(X_train_disc, y_train).predict(X_test_disc)
        )
        
        results.append(("Categorical", dataset_name, custom_acc, sklearn_acc, 0.15))
    
    print("Testing Complement Naive Bayes...")
    for dataset_name, data in datasets.items():
        X_train, X_test, y_train, y_test = data
        
        # Make data non-negative
        X_train_abs = np.abs(X_train)
        X_test_abs = np.abs(X_test)
        
        custom_acc = accuracy_score(
            y_test,
            ComplementNaiveBayes(alpha=1.0).fit(X_train_abs, y_train).predict(X_test_abs)
        )
        sklearn_acc = accuracy_score(
            y_test,
            ComplementNB(alpha=1.0).fit(X_train_abs, y_train).predict(X_test_abs)
        )
        
        results.append(("Complement", dataset_name, custom_acc, sklearn_acc, 0.15))
    
    print("\n" + "="*80)
    print(f"{'Algorithm':<15} {'Dataset':<18} {'Custom':<10} {'Sklearn':<10} {'Diff':<10} {'Status'}")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for algo, dataset, custom, sklearn, tolerance in results:
        diff = abs(custom - sklearn)
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        
        if diff < tolerance:
            passed += 1
        else:
            failed += 1
        
        print(f"{algo:<15} {dataset:<18} {custom:<10.4f} {sklearn:<10.4f} {diff:<10.4f} {status}")
    
    # Group by algorithm for summary
    print("\n" + "-"*80)
    print("SUMMARY BY ALGORITHM:")
    print("-"*80)
    
    algorithms = ["Gaussian", "Multinomial", "Bernoulli", "Categorical", "Complement"]
    for algo in algorithms:
        algo_results = [r for r in results if r[0] == algo]
        avg_diff = np.mean([abs(r[2] - r[3]) for r in algo_results])
        print(f"{algo:<15} - Avg difference: {avg_diff:.4f} (tested on {len(algo_results)} datasets)")
    
    print("\n" + "="*80)
    print(f"TOTAL: {passed}/{len(results)} tests passed, {failed} failed")
    print("="*80 + "\n")
    
    failed_tests = []
    for algo, dataset, custom, sklearn, tolerance in results:
        diff = abs(custom - sklearn)
        if diff >= tolerance:
            failed_tests.append(f"{algo} on {dataset}: diff={diff:.4f} > tolerance={tolerance}")
    
    assert len(failed_tests) == 0, \
        f"Some tests failed:\n" + "\n".join(failed_tests)
    
    print("✓ All implementations match scikit-learn within acceptable tolerances!")

