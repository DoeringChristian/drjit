import drjit as dr
import pytest
from dataclasses import dataclass

dr.set_log_level(dr.LogLevel.Info)


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
        y = func(x, y)
