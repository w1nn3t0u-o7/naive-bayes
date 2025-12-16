============
Contributing
============

Welcome to ``naive-bayes`` contributor's guide! This document focuses on how to contribute to the codebase, but other types of contributions (like documentation improvements) are also appreciated.

If you are new to contributing, `contribution-guide.org`_ is an excellent place to start.

All contributors are expected to be **open, considerate, and respectful**. When in doubt, `Python Software Foundation's Code of Conduct`_ is a good reference.

Issue Reports
=============

If you experience bugs, please check the `issue tracker`_ to see if a similar problem has already been reported. If not, feel free to create a new issue report.

New reports should include your environment details (OS, Python version) and minimal steps to reproduce the problem.

Documentation Improvements
==========================

The project's documentation uses reStructuredText_ and is compiled with Sphinx_.

To work on documentation locally, you can compile the docs with `tox`_::

    tox -e docs

Then preview the changes in your browser by running a local server::

    python3 -m http.server --directory 'docs/_build/html'

Code Contributions
==================

Project Internals
-----------------

The project is structured to have a clean separation between the core algorithms and the testing suite:
*   **`src/naive_bayes/`**: Contains the five Naive Bayes algorithm implementations, each in its own class. The code is pure NumPy.
*   **`tests/`**: Contains `pytest`-compatible tests that validate each algorithm against scikit-learn's implementations using several standard datasets.

Contribution Workflow
---------------------

1.  **Submit an issue:** Before starting significant work, create an issue to discuss the proposed changes.
2.  **Create an environment:**
    It is recommended to use a `virtual environment`_::

        virtualenv .venv
        source .venv/bin/activate
        pip install -e .[testing]

3.  **Clone the repository:**
    Fork the project, then clone your fork locally::

        git clone git@github.com:YourLogin/naive-bayes.git
        cd naive-bayes

4.  **Implement your changes:**
    Create a new branch for your feature::

        git checkout -b my-feature

    As you work, please add docstrings_ to new functions/classes and add yourself to ``AUTHORS.rst``. When done, commit your changes with a `descriptive commit message`_::

        git add <MODIFIED FILES>
        git commit

5.  **Run tests:**
    Check that your changes don't break existing tests by running `tox`_::

        tox

6.  **Submit your contribution:**
    Push your branch to your fork::

        git push -u origin my-feature

    Then, go to your fork on GitHub and create a pull request.

Maintainer tasks
================

Releases
--------

If you are a maintainer with permissions on PyPI_, a new version can be released with these steps:

1.  Ensure all tests are passing on the ``main`` branch.
2.  Tag the current commit with a release tag (e.g., ``v1.2.3``).
3.  Push the tag to the upstream repository: ``git push upstream v1.2.3``.
4.  Clean old builds: ``tox -e clean``.
5.  Build the new package: ``tox -e build``.
6.  Publish to PyPI: ``tox -e publish -- --repository pypi``.

.. _repository: https://github.com/w1nn3t0u/naive-bayes
.. _issue tracker: https://github.com/w1nn3t0u/naive-bayes/issues

.. _contribution-guide.org: https://www.contribution-guide.org/
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _PyPI: https://pypi.org/
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tox: https://tox.wiki/en/stable/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/

