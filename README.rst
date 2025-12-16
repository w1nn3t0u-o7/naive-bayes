.. image:: https://readthedocs.org/projects/naive-bayes/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://naive-bayes.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===========
naive-bayes
===========

**Five Naive Bayes classifiers**

This project provides educational implementations of Gaussian, Multinomial, Bernoulli, Categorical, and Complement Naive Bayes algorithms. Each classifier follows the scikit-learn API and has been validated against scikit-learn on multiple real-world datasets.

Quick Start
===========

.. code-block:: python

    from naive_bayes import GaussianNaiveBayes
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train and predict
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

Documentation
=============

Full documentation available at https://naive-bayes.readthedocs.io/

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.

