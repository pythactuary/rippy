.. rippy documentation master file, created by
   sphinx-quickstart on Sat Feb 10 12:23:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Rippy's documentation!
=================================

**Rippy** (ReInsurance Pricing in Python) is a simple, fast and lightweight simulation-based reinsurance modeling package. It has two main components:

1. Frequency-Severity, otherwise known as compound distribution simulation

   Rippy contains the ``FrequencySeverityModel`` class to set up and simulate from compound distributions with a variety of frequency and severity distributions

   .. code-block:: python

      from rippy import FrequencySeverityModel, Distributions
      claims = FrequencySeverityModel(Distributions.Poisson(5),Distributions.Pareto(0.5,100000)).generate()
   

2. Reinsurance contract calculation
   
   Rippy can then calculate the recoveries of a number of types of reinsurance contract for the simulated claims.

   .. code-block:: python

      from rippy import XoL
      xol_layer = XoL(name="Layer 1", limit=1000000, excess=100000, premium=1000)
      xol_layer.apply(claims)
      xol_layer.print_summary()
 

.. note::

   This project is under active development.


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   modules
