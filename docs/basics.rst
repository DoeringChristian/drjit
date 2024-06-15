.. py:currentmodule:: drjit

.. _basics:

Basics
======

This document briefly reviews various standard operations. It assumes
the following import declarations

.. code-block:: python

   import drjit as dr
   from drjit.auto import Float, Array3f, UInt


Creating arrays
---------------

Recall that Dr.Jit array types are dynamically sized--for example,
:py:class:`Float <drjit.auto.Float>` refers to a 1D array of single
precision variables.

The simplest way to create such an array is to call its constructor with
a list of explicit values:

.. code-block:: python

   a = Float(1, 2, 3, 4)
   print(a) # [1, 2, 3, 4]

The constructor also accepts ``Sequence`` types (e.g. lists, tuples, NumPy
arrays, PyTorch tensors, etc.):

.. code-block:: python

   x = Float([1, 2, 3, 4])

Nested array types store several variable---for example, :py:class:`Array3f
<drjit.auto.Array3f>` just wraps 3 :py:class:`Float <drjit.auto.Float>` instances.
They can be passed to the constructor explicitly, or via implicit conversion
from constants, lists, etc.

.. code-block:: python

   a = Array3f([1, 2], 0, Float(10, 20))
   print(a)
   # Prints (with 'y' component broadcast to full size)
   # [[1, 0, 10],
   #  [2, 0, 20]]

Various functions can also create default-initialized arrays:

- :py:func:`dr.zeros() <zeros>`: ``[0, 0, ...]``.
- :py:func:`dr.ones() <ones>`: ``[1, 1, ...]``.
- :py:func:`dr.full() <full>`: ``[x, x, ...]`` given ``x``.
- :py:func:`dr.arange() <arange>`: ``[0, 1, 2, ...]``.
- :py:func:`dr.linspace() <linspace>`: linear interpolation of two endpoints.
- :py:func:`dr.empty() <empty>`: allocate uninitialized memory.

These always take the desired output type as first argment. You can optionally
request a given size along the dynamic axis, e.g.:

.. code-block:: python

   b = dr.zeros(Array3f)
   print(b.shape) # Prints: (3, 1)

   b = dr.zeros(Array3f, shape=(3, 1000))
   print(b.shape) # Prints: (3, 1000)


Element access
--------------

Use the default ``array[index]`` syntax to read/write array entries. Nested
Static 1-4D arrays further expose equivalent ``.x`` / ``.y`` / ``.z`` / ``.w``
members:

.. code-block:: python

   a = Array3f(1, 2, 3)
   a.x += a.z + a[1]

Static 1-4D arrays also support `swizzling
<https://en.wikipedia.org/wiki/Swizzling_(computer_graphics)>`__, which arbitrarily reorders elements:

.. code-block:: python

   a.xy = a.xx + a.yx



.. code-block:: python

   print(a.shape)
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
     - .. inverse
     - Hyperbolic
     - .. inverse
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
