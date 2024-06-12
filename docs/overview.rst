.. py:currentmodule:: drjit

.. _overview:

Overview
========

.. code-block:: python

   from drjit.auto import Array3i

   x0 = dr.zeros(Array3i)
   x1 = dr.zeros(Array3i, shape=(3, 1000))

Functions to create arrays include :py:func:`zeros`, :py:func:`empty`,
:py:func:`ones`, :py:func:`full`, :py:func:`arange`, and :py:func:`linspace`.


.. list-table:: Basic arithmetic

The system provides standard transcendental functions.

.. list-table:: Basic arithmetic
   :header-rows: 0

   * - :py:func:`abs`
     - :py:func:`fma`
     - :py:func:`clip`
     - :py:func:`lerp`
   * - :py:func:`sqrt`
     - :py:func:`cbrt`
     - :py:func:`rcp`
     - :py:func:`rsqrt`
   * - :py:func:`min`
     - :py:func:`minimum`
     - :py:func:`max`
     - :py:func:`maximum`
   * - :py:func:`round`
     - :py:func:`ceil`
     - :py:func:`floor`
     - :py:func:`trunc`
   * - :py:func:`sign`
     - :py:func:`copysign`
     - :py:func:`mulsign`
     -

.. list-table:: Trigonometry
   :header-rows: 1

   * - Ordinary
     - Inverse ordinary
     - Hyperbolic
     - Inverse hyperbolic
   * - :py:func:`sin`
     - :py:func:`asin`
     - :py:func:`sinh`
     - :py:func:`asinh`
   * - :py:func:`cos`
     - :py:func:`acos`
     - :py:func:`cosh`
     - :py:func:`acosh`
   * - :py:func:`tan`
     - :py:func:`atan`
     - :py:func:`tanh`
     - :py:func:`atanh`
   * - :py:func:`sincos`
     - :py:func:`atan2`
     - :py:func:`sincosh`
     -

.. list-table:: Other transcendental functions
   :header-rows: 0

   * - :py:func:`log`
     - :py:func:`exp`
     - :py:func:`log2`
     - :py:func:`exp2`
   * - :py:func:`erf`
     - :py:func:`erfinv`
     - :py:func:`lgamma`
     - :py:func:`power`
