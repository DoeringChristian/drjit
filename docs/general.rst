.. py:module:: drjit

General information
===================

.. _optimizations:

Optimizations
-------------

This section reviews optimizations that Dr.Jit performs while tracing code. The
examples all use the following import:

.. code-block:: pycon

   >>> from drjit.llvm import Int

Vectorization and parallelization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dr.Jit automatically *vectorizes* and *parallelizes* traced code. The
implications of these transformations are backend-specific.

Consider the following simple calculation, which squares an integer
sequence with 10000 elements.

.. code-block:: pycon

   >>> dr.arange(dr.llvm.Int, 10000)**2
   [0, 1, 4, .. 9994 skipped .., 99940009, 99960004, 99980001]

On the LLVM backend, *vectorization* implies that generated code uses
instruction set extensions such as Intel AVX/AVX2/AVX512, or ARM NEON when they
are available. When the machine, e.g., supports `AVX512
<https://en.wikipedia.org/wiki/AVX-512>`__, each machine instruction processes a
*packet* of 16 values, which means that a total of 625 packets need to be
processed.

When there are more than 1K packets (the default), each successive group of 1K
packets forms a *block* for parallel processing using the built-in `nanothread
<https://github.com/mitsuba-renderer/nanothread>`__ thread pool. In this case,
there is not enough work for multi-core parallelism, and the computation
immediately runs on the calling thread.

You can use the functions :py:func:`drjit.thread_count`,
:py:func:`drjit.set_thread_count`, :py:func:`drjit.block_size`,
:py:func:`drjit.set_block_size` to fine-tune this process.

On the CUDA backend, the system automatically determines a number of *threads*
that maximize occupancy along with a suitable number of *blocks* and then
launches a parallel program that spreads out over the entire GPU (assuming that
there is enough work).

.. _cow:

Copy-on-Write 
^^^^^^^^^^^^^

Arrays are reference-counted and use a `Copy-on-Write
<https://en.wikipedia.org/wiki/Copy-on-write>`__ (CoW) strategy. This means
that copying an array is cheap since the copy can reference the original array
without requiring a device memory copy. The matching variable indices in the
example below demonstrate the lack of an actual copy.

.. code-block:: pycon

   >>> a = Int(1, 2, 3)
   >>> b = Int(a)        # <- create a copy of 'a'
   >>> a.index, b.index
   (1, 1)

However, subsequent modification causes this copy to be made.

.. code-block:: pycon

   >>> b[0] = 0
   >>> (a.index, b.index)
   (1, 2)

This optimization is always active and cannot be disabled.

Constant propagation
^^^^^^^^^^^^^^^^^^^^

Dr.Jit immediately performs arithmetic involving *literal constant* arrays:

.. code-block:: pycon

   >>> a = Int(4) + Int(5)
   >>> a.state
   dr.VarState.Literal

In other words, the addition does not become part of the generated device code.
This optimization reduces the size of the generated LLVM/PTX IR and can be
controlled via :py:attr:`drjit.JitFlag.ConstantPropagation`.

Value numbering
^^^^^^^^^^^^^^^

Dr.Jit collapses identical expressions into the same variable (this is safe
given the :ref:`CoW <cow>` strategy explained above).

.. code-block:: pycon

   >>> a, b = Int(1, 2, 3), Int(4, 5, 6)
   >>> c = a + b
   >>> d = a + b
   >>> c.index == d.index
   True

This optimization reduces the size of the generated LLVM/PTX IR and can be
controlled via :py:attr:`drjit.JitFlag.ValueNumbering`.

Local atomic reduction
^^^^^^^^^^^^^^^^^^^^^^

Atomic memory operations can be a bottleneck whenever they encounter *write
contention*, which refers to a situation where many threads attempt to write to
the same array element at once.

For example, the following operation causes 1000'000 threads to write to
``a[0]``.

.. code-block:: pycon

   >>> a = dr.zeros(Int, 10)
   >>> dr.scatter_add(target=a, index=dr.zeros(Int, 1000000), value=...)

Since Dr.Jit vectorizes the program during execution, the computation is
grouped into *packets* that typically contain 16 to 32 elements. By locally
adding values within each packet and then performing only 31-62K atomic memory
operations, performance can be considerably improved.

This optimization is particularly important when combined with *reverse-mode
automatic differentiation*, which turns differentiable scalar reads into atomic
scatter-additions that are sometimes subject to write contentions.

Other
^^^^^

Some other optimizations are specific to symbolic operations, such as

- :py:attr:`drjit.JitFlag.OptimizeCalls`,
- :py:attr:`drjit.JitFlag.MergeFunctions`,
- :py:attr:`drjit.JitFlag.OptimizeLoops`.

Please refer the documentation of these flags for details.

.. _horizontal-reductions:

Horizontal reductions
---------------------

Dr.Jit offers the following *horizontal operations* that reduce the dimension
of an input array, tensor, or Python sequence:

- :py:func:`drjit.sum`, which reduces using ``+``,
- :py:func:`drjit.prod`, which reduces using ``*``,
- :py:func:`drjit.min`, which reduces using ``min()``,
- :py:func:`drjit.max`, which reduces using ``max()``,
- :py:func:`drjit.all`, which reduces using ``&``, and
- :py:func:`drjit.any`, which reduces using ``|``.

By default, these functions reduce along the outermost dimension and return an
instance of the array's element type. For instance, sum-reducing an array ``a`` of
type :py:class:`drjit.cuda.Array3f` would just be a convenient abbreviation for
the expression ``a[0] + a[1] + a[2]`` of type :py:class:`drjit.cuda.Float`.
Dr.Jit can execute this operation symbolically.

Reductions of dynamic 1D arrays (e.g., :py:class:`drjit.cuda.Float`) are an
important special case. Since each value of such an array represents a
different execution thread of a parallel program, Dr.Jit must first invoke
:py:func:`drjit.eval` to evaluate and store the array in memory and then launch
a device-specific implementation of a horizontal reduction. This interferes
Dr.Jit's regular mode of operation, which is to capture a maximally large
program without intermediate evaluation. In other words, use of such 1D
reductions may have a negative effect on performance. The operation will fail
in execution contexts where evaluation is forbidden, e.g., while capturing
symbolic loops and function calls.

Furthermore Dr.Jit does *not* reduce 1D arrays to their element type (e.g., a
standard Python `float`). Instead, it returns a dynamic array of the same type,
containing only a single element. This is intentional--unboxing the array into
a Python scalar would require transferring the value to the CPU, which would
incur GPU<->CPU synchronization overheads. You must explicitly index into the
result (``result[0]``) to obtain a value with the underlying element type.
Boolean arrays define a ``__bool__`` method so that such indexing can be
avoided. For example, the following works as expected:

.. code-block:: python

   a = drjit.cuda.Float(...)
   # The line below is simply a nicer way of writing "if dr.any(a < 0)[0]:"
   if dr.any(a < 0):
      # ...

All reduction operations take an optional argument ``axis`` that specifies the
axis of the reduction (default: ``0``). The value ``None`` implies a reduction
over all array axes. Arguments other than ``0`` and ``None`` are currently
unsupported.

.. _pytrees:

PyTrees
-------

The word *PyTree* (borrowed from `JAX
<https://jax.readthedocs.io/en/latest/pytrees.html>`_) refers to a tree-like
data structure made of Python container types including ``list``, ``tuple``,
and ``dict``, which can be further extended to encompass user-defined classes.

Various Dr.Jit operations will automatically traverse such PyTrees to process
any Dr.Jit arrays or tensors found within. For example, it might be convenient
to store differentiable parameters of an optimization within a dictionary and
then batch-enable gradients:

.. code-block:: python

   from drjit.cuda.ad import Array3f, Float

   params = {
       'foo': Array3f(...),
       'bar': Float(...)
   }

   dr.enable_grad(params)

PyTrees can similarly be used as state variables in symbolic loops and
conditionals, as arguments and return values of symbolic calls, as arguments of
scatter/gather operations, and many others (the :ref:`reference <reference>`
explicitly lists the word *PyTree* in all supported operations).

Limitations
^^^^^^^^^^^

You may not use Dr.Jit types as *keys* of a dictionary occurring within a
PyTree. Furthermore, PyTrees may not contain cycles. For example, the following
data structure will cause PyTree-compatible operations to fail with a
``RecursionError``.

.. code-block:: python

   x = []
   x.append(x)

Finally, Dr.Jit automatically traverses tuples, lists, and dictionaries,
but it does not traverse subclasses of basic containers and other generalized
sequences or mappings.

.. _custom_types_py:

Custom types
^^^^^^^^^^^^

To turn a user-defined type into a PyTree, define a static ``DRJIT_STRUCT``
member dictionary describing the names and types of all fields. It should also
be default-constructible without the need to specify any arguments. For
instance, the following snippet defines a named 2D point, containing (amongst
others) two nested Dr.Jit arrays.

.. code-block:: python

   from drjit.cuda.ad import Float

   class MyPoint2f:
       DRJIT_STRUCT = { 'x' : Float, 'y': Float }

       def __init__(self, x: Float = None, y: Float = None):
           self.x = x
           self.y = y

   # Create a vector representing 100 2D points. Dr.Jit will
   # automatically populate the 'x' and 'y' members
   value = dr.zeros(MyPoint2f, 100)

Fields don't exclusively have to be containers or Dr.Jit types. For example, we
could have added an extra ``datetime`` entry to record when a set of points was
captured. Such fields will be ignored by traversal operations.


.. _transcendental-accuracy:

Accuracy of transcendental operations
-------------------------------------

Single precision
^^^^^^^^^^^^^^^^

.. note::

    The trigonometric functions *sin*, *cos*, and *tan* are optimized for low
    error on the domain :math:`|x| < 8192` and don't perform as well beyond
    this range.

.. list-table::
    :widths: 5 8 8 10 8 10
    :header-rows: 1
    :align: center

    * - Function
      - Tested domain
      - Abs. error (mean)
      - Abs. error (max)
      - Rel. error (mean)
      - Rel. error (max)
    * - :math:`\text{sin}()`
      - :math:`-8192 < x < 8192`
      - :math:`1.2 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`1.9 \cdot 10^{-8}\,(0.25\,\text{ulp})`
      - :math:`1.8 \cdot 10^{-6}\,(19\,\text{ulp})`
    * - :math:`\text{cos}()`
      - :math:`-8192 < x < 8192`
      - :math:`1.2 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`1.9 \cdot 10^{-8}\,(0.25\,\text{ulp})`
      - :math:`3.1 \cdot 10^{-6}\,(47\,\text{ulp})`
    * - :math:`\text{tan}()`
      - :math:`-8192 < x < 8192`
      - :math:`4.7 \cdot 10^{-6}`
      - :math:`8.1 \cdot 10^{-1}`
      - :math:`3.4 \cdot 10^{-8}\,(0.42\,\text{ulp})`
      - :math:`3.1 \cdot 10^{-6}\,(30\,\text{ulp})`
    * - :math:`\text{asin}()`
      - :math:`-1 < x < 1`
      - :math:`2.3 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`2.9 \cdot 10^{-8}\,(0.33\,\text{ulp})`
      - :math:`2.3 \cdot 10^{-7}\,(2\,\text{ulp})`
    * - :math:`\text{acos}()`
      - :math:`-1 < x < 1`
      - :math:`4.7 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`2.9 \cdot 10^{-8}\,(0.33\,\text{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\text{ulp})`
    * - :math:`\text{atan}()`
      - :math:`-1 < x < 1`
      - :math:`1.8 \cdot 10^{-7}`
      - :math:`6 \cdot 10^{-7}`
      - :math:`4.2 \cdot 10^{-7}\,(4.9\,\text{ulp})`
      - :math:`8.2 \cdot 10^{-7}\,(12\,\text{ulp})`
    * - :math:`\text{sinh}()`
      - :math:`-10 < x < 10`
      - :math:`2.6 \cdot 10^{-5}`
      - :math:`2 \cdot 10^{-3}`
      - :math:`2.8 \cdot 10^{-8}\,(0.34\,\text{ulp})`
      - :math:`2.7 \cdot 10^{-7}\,(3\,\text{ulp})`
    * - :math:`\text{cosh}()`
      - :math:`-10 < x < 10`
      - :math:`2.9 \cdot 10^{-5}`
      - :math:`2 \cdot 10^{-3}`
      - :math:`2.9 \cdot 10^{-8}\,(0.35\,\text{ulp})`
      - :math:`2.5 \cdot 10^{-7}\,(4\,\text{ulp})`
    * - :math:`\text{tanh}()`
      - :math:`-10 < x < 10`
      - :math:`4.8 \cdot 10^{-8}`
      - :math:`4.2 \cdot 10^{-7}`
      - :math:`5 \cdot 10^{-8}\,(0.76\,\text{ulp})`
      - :math:`5 \cdot 10^{-7}\,(7\,\text{ulp})`
    * - :math:`\text{asinh}()`
      - :math:`-30 < x < 30`
      - :math:`2.8 \cdot 10^{-8}`
      - :math:`4.8 \cdot 10^{-7}`
      - :math:`1 \cdot 10^{-8}\,(0.13\,\text{ulp})`
      - :math:`1.7 \cdot 10^{-7}\,(2\,\text{ulp})`
    * - :math:`\text{acosh}()`
      - :math:`1 < x < 10`
      - :math:`2.9 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`1.5 \cdot 10^{-8}\,(0.18\,\text{ulp})`
      - :math:`2.4 \cdot 10^{-7}\,(3\,\text{ulp})`
    * - :math:`\text{atanh}()`
      - :math:`-1 < x < 1`
      - :math:`9.9 \cdot 10^{-9}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`1.5 \cdot 10^{-8}\,(0.18\,\text{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\text{ulp})`
    * - :math:`\text{exp}()`
      - :math:`-20 < x < 30`
      - :math:`0.72 \cdot 10^{4}`
      - :math:`0.1 \cdot 10^{7}`
      - :math:`2.4 \cdot 10^{-8}\,(0.27\,\text{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\text{ulp})`
    * - :math:`\text{log}()`
      - :math:`10^{-20} < x < 2\cdot 10^{30}`
      - :math:`9.6 \cdot 10^{-9}`
      - :math:`7.6 \cdot 10^{-6}`
      - :math:`1.4 \cdot 10^{-10}\,(0.0013\,\text{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\text{ulp})`
    * - :math:`\text{erf}()`
      - :math:`-1 < x < 1`
      - :math:`3.2 \cdot 10^{-8}`
      - :math:`1.8 \cdot 10^{-7}`
      - :math:`6.4 \cdot 10^{-8}\,(0.78\,\text{ulp})`
      - :math:`3.3 \cdot 10^{-7}\,(4\,\text{ulp})`
    * - :math:`\text{erfc}()`
      - :math:`-1 < x < 1`
      - :math:`3.4 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`6.4 \cdot 10^{-8}\,(0.79\,\text{ulp})`
      - :math:`1 \cdot 10^{-6}\,(11\,\text{ulp})`

Double precision
^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 5 8 8 10 8 10
    :header-rows: 1
    :align: center

    * - Function
      - Tested domain
      - Abs. error (mean)
      - Abs. error (max)
      - Rel. error (mean)
      - Rel. error (max)
    * - :math:`\text{sin}()`
      - :math:`-8192 < x < 8192`
      - :math:`2.2 \cdot 10^{-17}`
      - :math:`2.2 \cdot 10^{-16}`
      - :math:`3.6 \cdot 10^{-17}\,(0.25\,\text{ulp})`
      - :math:`3.1 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{cos}()`
      - :math:`-8192 < x < 8192`
      - :math:`2.2 \cdot 10^{-17}`
      - :math:`2.2 \cdot 10^{-16}`
      - :math:`3.6 \cdot 10^{-17}\,(0.25\,\text{ulp})`
      - :math:`3 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{tan}()`
      - :math:`-8192 < x < 8192`
      - :math:`6.8 \cdot 10^{-16}`
      - :math:`1.2 \cdot 10^{-10}`
      - :math:`5.4 \cdot 10^{-17}\,(0.35\,\text{ulp})`
      - :math:`4.1 \cdot 10^{-16}\,(3\,\text{ulp})`
    * - :math:`\text{cot}()`
      - :math:`-8192 < x < 8192`
      - :math:`4.9 \cdot 10^{-16}`
      - :math:`1.2 \cdot 10^{-10}`
      - :math:`5.5 \cdot 10^{-17}\,(0.36\,\text{ulp})`
      - :math:`4.4 \cdot 10^{-16}\,(3\,\text{ulp})`
    * - :math:`\text{asin}()`
      - :math:`-1 < x < 1`
      - :math:`1.3 \cdot 10^{-17}`
      - :math:`2.2 \cdot 10^{-16}`
      - :math:`1.5 \cdot 10^{-17}\,(0.098\,\text{ulp})`
      - :math:`2.2 \cdot 10^{-16}\,(1\,\text{ulp})`
    * - :math:`\text{acos}()`
      - :math:`-1 < x < 1`
      - :math:`5.4 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`3.5 \cdot 10^{-17}\,(0.23\,\text{ulp})`
      - :math:`2.2 \cdot 10^{-16}\,(1\,\text{ulp})`
    * - :math:`\text{atan}()`
      - :math:`-1 < x < 1`
      - :math:`4.3 \cdot 10^{-17}`
      - :math:`3.3 \cdot 10^{-16}`
      - :math:`1 \cdot 10^{-16}\,(0.65\,\text{ulp})`
      - :math:`7.1 \cdot 10^{-16}\,(5\,\text{ulp})`
    * - :math:`\text{sinh}()`
      - :math:`-10 < x < 10`
      - :math:`3.1 \cdot 10^{-14}`
      - :math:`1.8 \cdot 10^{-12}`
      - :math:`3.3 \cdot 10^{-17}\,(0.22\,\text{ulp})`
      - :math:`4.3 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{cosh}()`
      - :math:`-10 < x < 10`
      - :math:`2.2 \cdot 10^{-14}`
      - :math:`1.8 \cdot 10^{-12}`
      - :math:`2 \cdot 10^{-17}\,(0.13\,\text{ulp})`
      - :math:`2.9 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{tanh}()`
      - :math:`-10 < x < 10`
      - :math:`5.6 \cdot 10^{-17}`
      - :math:`3.3 \cdot 10^{-16}`
      - :math:`6.1 \cdot 10^{-17}\,(0.52\,\text{ulp})`
      - :math:`5.5 \cdot 10^{-16}\,(3\,\text{ulp})`
    * - :math:`\text{asinh}()`
      - :math:`-30 < x < 30`
      - :math:`5.1 \cdot 10^{-17}`
      - :math:`8.9 \cdot 10^{-16}`
      - :math:`1.9 \cdot 10^{-17}\,(0.13\,\text{ulp})`
      - :math:`4.4 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{acosh}()`
      - :math:`1 < x < 10`
      - :math:`4.9 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`2.6 \cdot 10^{-17}\,(0.17\,\text{ulp})`
      - :math:`6.6 \cdot 10^{-16}\,(5\,\text{ulp})`
    * - :math:`\text{atanh}()`
      - :math:`-1 < x < 1`
      - :math:`1.8 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`3.2 \cdot 10^{-17}\,(0.21\,\text{ulp})`
      - :math:`3 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{exp}()`
      - :math:`-20 < x < 30`
      - :math:`4.7 \cdot 10^{-6}`
      - :math:`2 \cdot 10^{-3}`
      - :math:`2.5 \cdot 10^{-17}\,(0.16\,\text{ulp})`
      - :math:`3.3 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{log}()`
      - :math:`10^{-20} < x < 2\cdot 10^{30}`
      - :math:`1.9 \cdot 10^{-17}`
      - :math:`1.4 \cdot 10^{-14}`
      - :math:`2.7 \cdot 10^{-19}\,(0.0013\,\text{ulp})`
      - :math:`2.2 \cdot 10^{-16}\,(1\,\text{ulp})`
    * - :math:`\text{erf}()`
      - :math:`-1 < x < 1`
      - :math:`4.7 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`9.6 \cdot 10^{-17}\,(0.63\,\text{ulp})`
      - :math:`5.9 \cdot 10^{-16}\,(5\,\text{ulp})`
    * - :math:`\text{erfc}()`
      - :math:`-1 < x < 1`
      - :math:`4.8 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`9.6 \cdot 10^{-17}\,(0.64\,\text{ulp})`
      - :math:`2.5 \cdot 10^{-15}\,(16\,\text{ulp})`