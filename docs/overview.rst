.. py:currentmodule:: drjit

.. _overview:

Overview
========

This document quickly reviews various standard operations. It assumes
the following import declarations

.. code-block:: python

   import drjit as dr
   from drjit.auto import Float, Array3f, UInt


Creating arrays
---------------

Recall that Dr.Jit array types are dynamically sized, so :py:class:`Float
<drjit.auto.Float>` refers to a 1D array of single precision floats:

The simplest way to create such an array is to call its constructor with
a list of explicit values:
You can create such arrays by explicitly listing the array values.

.. code-block:: python

   x = Float(1, 2, 3, 4)
   print(x) # [1, 2, 3, 4]

The constructor also accepts ``Sequence`` types (e.g. lists, tuples, NumPy
arrays, PyTorch tensors, etc.):

.. code-block:: python

   x = Float([1, 2, 3, 4])

Nested array types like :py:class:`Array3f <drjit.auto.Array3f>` wrap several
instances of their base type (in this case, three :py:class:`Float
<drjit.auto.Float>` objects) that can be passed to the constructor explicitly,
or via implicit conversion from constants, lists, etc. Scalar arrays always
broadcast as needed to be compatible.

.. code-block:: python

   y = Array3f([1, 2], 0, Float(10, 20))
   print(y)
   # Prints:
   # [[1, 0, 10],
   #  [2, 0, 20]]

You can also create default-initialized arrays using helper functions
that take the desired output type as first argment:

.. code-block:: python

   x0 = dr.zeros(Array3f)
   print(x0.shape) # Prints: (3, 1)

   x1 = dr.zeros(Array3f, shape=(3, 1000))
   print(x1.shape) # Prints: (3, 1000)

Functions to do so include :py:func:`zeros`, :py:func:`empty`, :py:func:`ones`,
:py:func:`full`, :py:func:`arange`, and :py:func:`linspace`.

Element access
--------------

.. code-block:: python

   print(y.shape)
   (3, 2)



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
