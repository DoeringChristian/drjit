import drjit as dr
import pytest
from dataclasses import dataclass
import sys

dr.set_log_level(dr.LogLevel.Debug)

def get_single_entry(x):
    tp = type(x)
    result = x
    shape = dr.shape(x)
    if len(shape) == 2:
        result = result[shape[0]-1]
    if len(shape) == 3:
        result = result[shape[0]-1][shape[1]-1]
    return result

@pytest.test_arrays("uint32, jit, shape=(*)")
def test01_basic(t):
    @dr.freeze
    def func(x, y):
        return x + y

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    o0 = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    o0 = func(i0, i1)
    assert dr.all(t(4, 4, 4) == o0)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test02_output_tuple(t):
    @dr.freeze
    def func(x, y):
        return (x + y, x * y)

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    (o0, o1) = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)
    assert dr.all(t(0, 1, 0) == o1)

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    (o0, o1) = func(i0, i1)
    assert dr.all(t(4, 4, 4) == o0)
    assert dr.all(t(3, 4, 3) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test03_output_list(t):
    @dr.freeze
    def func(x, y):
        return [x + y, x * y]

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    [o0, o1] = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)
    assert dr.all(t(0, 1, 0) == o1)

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    [o0, o1] = func(i0, i1)
    assert dr.all(t(4, 4, 4) == o0)
    assert dr.all(t(3, 4, 3) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test04_output_dict(t):
    @dr.freeze
    def func(x, y):
        return {"add": x + y, "mul": x * y}

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    o = func(i0, i1)
    o0 = o["add"]
    o1 = o["mul"]
    assert dr.all(t(2, 2, 2) == o0)
    assert dr.all(t(0, 1, 0) == o1)

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    o = func(i0, i1)
    o0 = o["add"]
    o1 = o["mul"]
    assert dr.all(t(4, 4, 4) == o0)
    assert dr.all(t(3, 4, 3) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test05_nested_tuple(t):
    @dr.freeze
    def func(x):
        return (x + 1, x + 2, (x + 3, x + 4))

    i0 = t(0, 1, 2)

    (o0, o1, (o2, o3)) = func(i0)
    assert dr.all(t(1, 2, 3) == o0)
    assert dr.all(t(2, 3, 4) == o1)
    assert dr.all(t(3, 4, 5) == o2)
    assert dr.all(t(4, 5, 6) == o3)

    i0 = t(1, 2, 3)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test06_drjit_struct(t):
    class Point:
        x: t
        y: t
        DRJIT_STRUCT = {"x": t, "y": t}

    @dr.freeze
    def func(x):
        p = Point()
        p.x = x + 1
        p.y = x + 2
        return p

    i0 = t(0, 1, 2)

    o = func(i0)
    o0 = o.x
    o1 = o.y
    assert dr.all(t(1, 2, 3) == o0)
    assert dr.all(t(2, 3, 4) == o1)

    i0 = t(1, 2, 3)

    o = func(i0)
    o0 = o.x
    o1 = o.y
    assert dr.all(t(2, 3, 4) == o0)
    assert dr.all(t(3, 4, 5) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test06_dataclass(t):
    @dataclass
    class Point:
        x: t
        y: t

    @dr.freeze
    def func(x):
        p = Point(x + 1, x + 2)
        return p

    i0 = t(0, 1, 2)

    o = func(i0)
    o0 = o.x
    o1 = o.y
    assert dr.all(t(1, 2, 3) == o0)
    assert dr.all(t(2, 3, 4) == o1)

    i0 = t(1, 2, 3)

    o = func(i0)
    o0 = o.x
    o1 = o.y
    assert dr.all(t(2, 3, 4) == o0)
    assert dr.all(t(3, 4, 5) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test07_scatter(t):
    @dr.freeze
    def func(x):
        dr.scatter(x, 0, dr.arange(t, 3))

    x = t(0, 1, 2)
    func(x)

    x = t(0, 1, 2)
    y = x + 1
    z = x
    w = t(x)

    func(x)

    assert dr.all(t(0, 0, 0) == x)
    assert dr.all(t(1, 2, 3) == y)
    assert dr.all(t(0, 0, 0) == z)
    assert dr.all(t(0, 1, 2) == w)


@pytest.test_arrays("float32, is_diff, shape=(*)")
def test08_optimization(t):
    @dr.freeze
    def func(state, ref):
        for k, x in state.items():
            dr.enable_grad(x)
            loss = dr.mean(dr.square(x - ref))

            dr.backward(loss)

            grad = dr.grad(x)
            dr.disable_grad(x)
            state[k] = x - grad

    state = {"x": t(0, 0, 0, 0)}

    ref = t(1, 1, 1, 1)
    func(state, ref)
    assert dr.allclose(t(0.5, 0.5, 0.5, 0.5), state["x"])

    state = {"x": t(0, 0, 0, 0)}
    ref = t(1, 1, 1, 1)
    func(state, ref)

    assert dr.allclose(t(0.5, 0.5, 0.5, 0.5), state["x"])


@pytest.test_arrays("uint32, jit, shape=(*)")
def test09_resized(t):
    @dr.freeze
    def func(x, y):
        return x + y

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    o0 = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)

    i0 = dr.arange(t, 64)
    i1 = dr.arange(t, 64)
    r0 = i0 + i1
    dr.eval(i0, i1, r0)

    o0 = func(i0, i1)
    assert dr.all(r0 == o0)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test10_changed_input_dict(t):
    @dr.freeze
    def func(d: dict):
        d["y"] = d["x"] + 1

    x = t(0, 1, 2)
    d = {"x": x}

    func(d)
    assert dr.all(t(1, 2, 3) == d["y"])

    x = t(1, 2, 3)
    d = {"x": x}

    func(d)
    assert dr.all(t(2, 3, 4) == d["y"])


@pytest.test_arrays("uint32, jit, shape=(*)")
def test11_changed_input_dataclass(t):
    @dataclass
    class Point:
        x: t

    @dr.freeze
    def func(p: Point):
        p.x = p.x + 1

    p = Point(x=t(0, 1, 2))

    func(p)
    assert dr.all(t(1, 2, 3) == p.x)

    p = Point(x=t(1, 2, 3))

    func(p)
    assert dr.all(t(2, 3, 4) == p.x)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test12_kwargs(t):
    @dr.freeze
    def func(x=t(0, 1, 2)):
        return x + 1

    y = func(x=t(0, 1, 2))
    assert dr.all(t(1, 2, 3) == y)

    y = func(x=t(1, 2, 3))
    assert dr.all(t(2, 3, 4) == y)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test13_opaque(t):
    @dr.freeze
    def func(x, y):
        return x + y

    x = t(0, 1, 2)
    dr.set_label(x, "x")
    y = dr.opaque(t, 1)
    dr.set_label(y, "y")
    z = func(x, y)
    assert dr.all(t(1, 2, 3) == z)

    with pytest.raises(RuntimeError):
        x = t(1, 2, 3)
        y = t(1, 2, 3)
        z = func(x, y)
        assert dr.all(t(2, 3, 4) == z)


@pytest.test_arrays("float32, jit, -is_diff, shape=(*)")
def test14_performance(t):
    import time

    n = 1024
    n_iter = 1_000
    n_iter_warmeup = 10

    def func(x, y):
        z = 0.5
        result = dr.fma(dr.square(x), y, z)
        result = dr.sqrt(dr.abs(result) + dr.power(result, 10))
        result = dr.log(1 + result)
        return result

    frozen = dr.freeze(func)

    for name, fn in [("normal", func), ("frozen", frozen)]:
        x = dr.arange(t, n)  # + dr.opaque(t, i)
        y = dr.arange(t, n)  # + dr.opaque(t, i)
        dr.eval(x, y)
        for i in range(n_iter + n_iter_warmeup):
            if i == n_iter_warmeup:
                t0 = time.time()

            result = fn(x, y)

            dr.eval(result)

        dr.sync_thread()
        elapsed = time.time() - t0
        print(f"{name}: average {1000 * elapsed / n_iter:.3f} ms / iteration")


@pytest.test_arrays("uint32, jit, shape=(*)")
def test15_aliasing(t):
    @dr.freeze
    def func(x, y):
        return x + y

    print("aliased:")
    x = t(0, 1, 2)
    y = x
    z = func(x, y)
    assert dr.all(t(0, 2, 4) == z)

    print("aliased:")
    x = t(1, 2, 3)
    y = x
    z = func(x, y)
    assert dr.all(t(2, 4, 6) == z)

    with pytest.raises(RuntimeError):
        print("non-aliased:")
        x = t(1, 2, 3)
        y = t(2, 3, 4)
        z = func(x, y)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test16_non_jit_types(t):
    @dr.freeze
    def func(x, y):
        return x + y

    x = t(1, 2, 3)
    y = 1

    with pytest.raises(RuntimeError):
        z = func(x, y)


@pytest.test_arrays("uint32, jit, cuda, -is_diff, shape=(*)")
def test17_literal(t):
    dr.set_log_level(dr.LogLevel.Trace)
    dr.set_flag(dr.JitFlag.KernelHistory, True)

    @dr.freeze
    def func(x, y):
        z = x + y
        w = t(1)
        return z, w

    # Literals
    x = t(0, 1, 2)
    dr.set_label(x, "x")
    y = t(1)
    dr.set_label(y, "y")
    z, w = func(x, y)
    assert dr.all(z == t(1, 2, 3))
    assert dr.all(w == t(1))

    x = t(0, 1, 2)
    y = t(1)
    z, w = func(x, y)
    assert dr.all(z == t(1, 2, 3))
    assert dr.all(w == t(1))

    with pytest.raises(RuntimeError):
        x = t(0, 1, 2)
        y = t(2)
        z = func(x, y)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test18_pointers(t):
    UInt32 = dr.uint32_array_t(t)

    @dr.freeze
    def func(x):
        idx = dr.arange(UInt32, 0, dr.width(x), 3)

        return dr.gather(t, x, idx)

    y = func(t(0, 1, 2, 3, 4, 5, 6))

    print(y)


def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("call_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda


@pytest.mark.parametrize("symbolic", [True])
@pytest.test_arrays("float32, jit, shape=(*)")
def test19_vcall(t, symbolic):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    c = BasePtr(a, a, None, a, a)

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    @dr.freeze
    def func(c, xi, yi):
        return c.f(xi, yi)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = func(c, xi, yi)
        
    assert dr.all(xo == t(10, 12, 0, 14, 16))
    assert dr.all(yo == t(-1, -2, 0, -3, -4))
    
    c = BasePtr(a, a, None, b, b)
        
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = func(c, xi, yi)
        
    assert dr.all(xo == t(10, 12, 0, 21, 24))
    assert dr.all(yo == t(-1, -2, 0, 3, 4))

    
@pytest.test_arrays("float32, jit, shape=(*)")
def test01_freeze(t):
    UInt32 = dr.uint32_array_t(t)
    Float = dr.float32_array_t(t)

    @dr.freeze
    def my_kernel(x):
        x_int = UInt32(x)
        result = x * x
        result_int = UInt32(result)

        return result, x_int, result_int

    for i in range(3):
        print(f"------------------------------ {i}")
        x = Float([1.0, 2.0, 3.0]) + dr.opaque(Float, i)

        y, x_int, y_int = my_kernel(x)
        dr.schedule(y, x_int, y_int)
        print("Input was:", x)
        print("Outputs were:", y, x_int, y_int)
        assert dr.allclose(y, dr.square(x))
        assert dr.allclose(x_int, UInt32(x))
        assert dr.allclose(y_int, UInt32(y))
        print("------------------------------")
        
        
@pytest.mark.parametrize("freeze_first", (True, False))
@pytest.test_arrays("float32, jit, shape=(*)")
def test12_calling_frozen_from_frozen(t, freeze_first):
    mod = sys.modules[t.__module__]
    Float = mod.Float32
    Array3f = mod.Array3f
    n = 37
    x = dr.full(Float, 1.5, n) + dr.opaque(Float, 2)
    y = dr.full(Float, 0.5, n) + dr.opaque(Float, 10)
    dr.eval(x, y)

    @dr.freeze
    def fun1(x):
        return dr.square(x)

    @dr.freeze
    def fun2(x, y):
        return fun1(x) + fun1(y)

    # Calling a frozen function from a frozen function.
    if freeze_first:
        dr.eval(fun1(x))

    result1 = fun2(x, y)
    assert dr.allclose(result1, dr.square(x) + dr.square(y))

    if not freeze_first:
        # If the nested function hasn't been recorded yet, calling it
        # while freezing the outer function shouldn't freeze it with
        # those arguments.
        # In other words, any freezing mechanism should be completely
        # disabled while recording a frozen function.
        # assert fun1.frozen.kernels is None

        # We should therefore be able to freeze `fun1` with a different
        # type of argument, and both `fun1` and `fun2` should work fine.
        result2 = fun1(Array3f(0.5, x, y))
        assert dr.allclose(result2, Array3f(0.5 * 0.5, dr.square(x), dr.square(y)))

        result3 = fun2(2 * x, 0.5 * y)
        assert dr.allclose(result3, dr.square(2 * x) + dr.square(0.5 * y))
        
        
@pytest.test_arrays("float32, jit, shape=(*)")
def test17_no_inputs(t):
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32
    Float = mod.Float

    @dr.freeze
    def fun(a):
        x = t(dr.linspace(Float, -1, 1, 10)) + a
        source = get_single_entry(x + 2 * x)
        index = dr.arange(UInt32, dr.width(source))
        active = index % UInt32(2) != 0

        return dr.gather(Float, source, index, active)

    a = t(0.1)
    res1 = fun(a)
    res2 = fun(a)
    res3 = fun(a)

    assert dr.allclose(res1, res2)
    assert dr.allclose(res1, res3)

    
@pytest.test_arrays("float32, jit, shape=(*)")
def test18_with_gathers(t):
    import numpy as np

    n = 20
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32
    # dr.set_log_level(dr.LogLevel.Debug)

    rng = np.random.default_rng(seed=1234)
    shape = tuple(reversed(dr.shape(dr.zeros(t, n))))

    def fun(x, idx):
        active = idx % 2 != 0
        source = get_single_entry(x)
        return dr.gather(type(source), source, idx, active=active)

    fun_frozen = dr.freeze(fun)

    # 1. Recording call
    x1 = t(rng.uniform(low=-1, high=1, size=shape))
    idx1 = dr.arange(UInt32, n)
    result1 = fun_frozen(x1, idx1)
    assert dr.allclose(result1, fun(x1, idx1))

    # 2. Different source as during recording
    x2 = t(rng.uniform(low=-2, high=-1, size=shape))
    idx2 = idx1

    result2 = fun_frozen(x2, idx2)
    assert dr.allclose(result2, fun(x2, idx2))

    x3 = x2
    idx3 = UInt32([i for i in reversed(range(n))])
    result3 = fun_frozen(x3, idx3)
    assert dr.allclose(result3, fun(x3, idx3))

    # 3. Same source as during recording
    result4 = fun_frozen(x1, idx1)
    assert dr.allclose(result4, result1)
    
@pytest.test_arrays("float32, jit, shape=(*)")
def test20_scatter_with_op(t):
    import numpy as np

    n = 20
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32
    
    rng = np.random.default_rng(seed=1234)

    def func(x, idx):
        active = idx % 2 != 0

        result = x - 0.5
        dr.scatter(x, result, idx, active = active)
        dr.eval((x, idx, result))
        return result

    func_frozen = dr.freeze(func)

    # 1. Recording call
    # print('-------------------- start result1')
    x1 = t(rng.uniform(low=-1, high=1, size=[n]))
    x1_copy = t(x1)
    x1_copy_copy = t(x1)
    idx1 = dr.arange(UInt32, n)
    
    result1 = func_frozen(x1, idx1)

    # assert dr.allclose(x1, x1_copy)
    assert dr.allclose(result1, func(x1_copy, idx1))
    
    # 2. Different source as during recording
    # print('-------------------- start result2')
    # TODO: problem: during trace, the actual x1 Python variable changes
    #       from index r2 to index r12 as a result of the `scatter`.
    #       But in subsequent launches, even if we successfully create a new
    #       output buffer equivalent to r12, it doesn't get assigned to `x2`.
    x2 = t(rng.uniform(low=-2, high=-1, size=[n]))
    x2_copy = t(x2)
    idx2 = idx1
    # print(f'Before: {x2.index=}, {idx2.index=}')

    result2 = func_frozen(x2, idx2)
    # print(f'After : {x2.index=}, {idx2.index=}')
    # print('-------------------- done with result2')
    assert dr.allclose(result2, func(x2_copy, idx2))
    # assert dr.allclose(x2, x2_copy)

    x3 = x2
    x3_copy = t(x3)
    idx3 = UInt32([i for i in reversed(range(n))])
    result3 = func_frozen(x3, idx3)
    assert dr.allclose(result3, func(x3_copy, idx3))
    # assert dr.allclose(x3, x3_copy)

    # 3. Same source as during recording
    result4 = func_frozen(x1_copy_copy, idx1)
    assert dr.allclose(result4, result1)
    # assert dr.allclose(x1_copy_copy, x1)
    
@pytest.test_arrays("float32, jit, shape=(*)")
def test21_with_gather_and_scatter(t):
    # TODO: this function seems to be causing some problems with pytest,
    # something about `repr()` being called on a weird / uninitialized JIT variable.
    # This crash is triggered even when the test should otherwise pass.

    import numpy as np

    n = 20
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32
    # dr.set_log_level(dr.LogLevel.Debug)

    rng = np.random.default_rng(seed=1234)
    shape = tuple(reversed(dr.shape(dr.zeros(t, n))))

    def fun(x, idx):
        active = idx % 2 != 0
        dest = get_single_entry(x)

        values = dr.gather(UInt32, idx, idx, active=active)
        values = type(dest)(values)
        dr.scatter(dest, values, idx, active=active)
        return dest, values

    fun_frozen = dr.freeze(fun)

    # 1. Recording call
    x1 = t(rng.uniform(low=-1, high=1, size=shape))
    x1_copy = t(x1)
    x1_copy_copy = t(x1)
    idx1 = dr.arange(UInt32, n)

    result1 = fun_frozen(x1, idx1)
    assert dr.allclose(result1, fun(x1_copy, idx1))
    assert dr.allclose(x1, x1_copy)

    # 2. Different source as during recording
    x2 = t(rng.uniform(low=-2, high=-1, size=shape))
    x2_copy = t(x2)
    idx2 = idx1

    result2 = fun_frozen(x2, idx2)
    assert dr.allclose(result2, fun(x2_copy, idx2))
    assert dr.allclose(x2, x2_copy)

    x3 = x2
    x3_copy = t(x3)
    idx3 = UInt32([i for i in reversed(range(n))])
    result3 = fun_frozen(x3, idx3)
    assert dr.allclose(result3, fun(x3_copy, idx3))
    assert dr.allclose(x3, x3_copy)

    # 3. Same source as during recording
    result4 = fun_frozen(x1_copy_copy, idx1)
    assert dr.allclose(result4, result1)
    assert dr.allclose(x1_copy_copy, x1)
    
# @pytest.mark.parametrize("relative_size", ["<", "=", ">"])
# @pytest.mark.parametrize("relative_size", ["<"])
# @pytest.test_arrays("float32, jit, shape=(*)")
# def test22_gather_only_pointer_as_input(t, relative_size):
#     
#     mod = sys.modules[t.__module__]
#     Array3f = mod.Array3f
#     Float = mod.Float32
#     UInt32 = mod.UInt32
#     
#     import numpy as np
#
#
#     rng = np.random.default_rng(seed=1234)
#
#     if relative_size == "<":
#
#         def fun(v):
#             idx = dr.arange(UInt32, 0, dr.width(v), 3)
#             return Array3f(dr.gather(Float, v, idx), dr.gather(Float, v, idx + 1), dr.gather(Float, v, idx + 2))
#
#     elif relative_size == "=":
#
#         def fun(v):
#             idx = dr.arange(UInt32, 0, dr.width(v)) // 2
#             return Array3f(dr.gather(Float, v, idx), dr.gather(Float, v, idx + 1), dr.gather(Float, v, idx + 2))
#
#     elif relative_size == ">":
#
#         def fun(v):
#             max_width = dr.width(v)
#             idx = dr.arange(UInt32, 0, 5 * max_width)
#             # TODO(!): what can we do against this literal being baked into the kernel?
#             active = (idx + 2) < max_width
#             return Array3f(
#                 dr.gather(Float, v, idx, active=active),
#                 dr.gather(Float, v, idx + 1, active=active),
#                 dr.gather(Float, v, idx + 2, active=active),
#             )
#
#     fun_freeze = dr.freeze(fun)
#
#     def check_results(v, result):
#         size = v.size
#         if relative_size == "<":
#             expected = v.T
#         if relative_size == "=":
#             idx = np.arange(0, size) // 2
#             expected = v.ravel()
#             expected = np.stack(
#                 [
#                     expected[idx],
#                     expected[idx + 1],
#                     expected[idx + 2],
#                 ],
#                 axis=0,
#             ).T
#         elif relative_size == ">":
#             idx = np.arange(0, 5 * size)
#             mask = (idx + 2) < size
#             expected = v.ravel()
#             expected = np.stack(
#                 [
#                     np.where(mask, expected[(idx) % size], 0),
#                     np.where(mask, expected[(idx + 1) % size], 0),
#                     np.where(mask, expected[(idx + 2) % size], 0),
#                 ],
#                 axis=0,
#             )
#
#         assert np.allclose(result.numpy(), expected)
#
#     # Note: Does not fail for n=1
#     n = 7
#     # dr.set_log_level(dr.LogLevel.Debug)
#
#     for i in range(3):
#         v = rng.uniform(size=[n, 3])
#         result = fun(Float(v.ravel()))
#         print(f"{dr.shape(result)=}")
#         check_results(v, result)
#
#     for i in range(10):
#         if i <= 5:
#             n_lanes = n
#         else:
#             n_lanes = n + i
#
#         print(f"{n_lanes=}")
#         v = rng.uniform(size=[n_lanes, 3])
#         result = fun_freeze(Float(v.ravel()))
#         # print(f'{i=}, {n_lanes=}, {v.shape=}, {result.numpy().shape=}')
#
#         expected_width = {
#             "<": n_lanes,
#             "=": n_lanes * 3,
#             ">": n_lanes * 3 * 5,
#         }[relative_size]
#
#         # if i == 0:
#             # assert len(fun_freeze.frozen.kernels)
#             # for kernel in fun_freeze.frozen.kernels.values():
#             #     assert kernel.original_input_size == n * 3
#             #     if relative_size == "<":
#             #         assert kernel.original_launch_size == expected_width
#             #         assert kernel.original_launch_size_ratio == (False, 3, True)
#             #     elif relative_size == "=":
#             #         assert kernel.original_launch_size == expected_width
#             #         assert kernel.original_launch_size_ratio == (False, 1, True)
#             #     else:
#             #         assert kernel.original_launch_size == expected_width
#             #         assert kernel.original_launch_size_ratio == (True, 5, True)
#
#         print(f"{dr.shape(result)=}")
#         assert dr.width(result) == expected_width
#         if relative_size == ">" and n_lanes != n:
#             pytest.xfail(
#                 reason="The width() of the original input is baked into the kernel to compute the `active` mask during the first launch, so results are incorrect once the width changes."
#             )
#
#         check_results(v, result)
