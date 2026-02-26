
Accelerating non-convex graphical Lasso
=====================
|Build Status| |Python 3.6+|

This code is a fork of https://github.com/Perceptronium/benchmark_graphical_lasso to compare acceleration of the solving of non-convex graphical Lasso formulation using the framework developed in https://arxiv.org/abs/2601.21467. 

This repository runs three solvers to solve the non-convex Graphical Lasso estimator (Banerjee et al., 2008):

$$\\min_{\\Theta \\succ 0} - \\log \\det (\\Theta) + \\langle \\Theta, S \\rangle + \\alpha \\sum_{i,j=1}^d \\phi_{i,j}(\\|[\\Theta}]^{(i,j)}\\|),$$

where $\\Theta$ is the optimization variable, $S$ is the empirical covariance matrix, $\\alpha$ is the regularization hyperparameter and $\\phi_{i,j}$ encodes a block-wise non-convex penalty.

The three solvers are Graphical ISTA (Rolfs et al., 2012), QUIC (Hsieh et al., 2014) and Primal GLasso (Mazumder et al., 2012). 

Install
--------

This benchmark can be run using the following commands, which first create a dedicated Conda environment:

.. code-block::

   $ conda create -n glasso_bench_env python=3.10
   $ conda activate glasso_bench_env
   $ pip install -U benchopt
   $ git clone https://github.com/Perceptronium/benchmark_graphical_lasso
   $ pip install gglasso
   $ git clone https://github.com/skggm/skggm ./benchmark_graphical_lasso/benchmark_utils/skggm
   $ pip install Cython
   $ pip install -e ./benchmark_graphical_lasso/benchmark_utils/skggm/
   $ bash execute_all_reweights.sh
   $ bash execute_all_plots.sh

execute_all_reweights.sh computes 1, 2, 3, 4, 5, 10, or 100 iterations per reweighting (20 reweighting in total) for each solver. Then all the plots of the article are created by execute_all_plots.sh

Please refer to https://github.com/Perceptronium/benchmark_graphical_lasso for a thorough benchmark of solvers.

.. |Build Status| image:: https://github.com/Perceptronium/benchmark_graphical_lasso/actoiworkflows/main.yml/badge.svg
   :target: https://github.com/Perceptronium/benchmark_graphical_lasso/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
