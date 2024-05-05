import drjit as dr
import pytest

@pytest.test_arrays('uint32, jit, shape=(*)')
def test01_basic(t):
    dr.set_log_level(dr.LogLevel.Info)

    @dr.freeze
    def func(x, y):
        return x + y

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)
    o0 = func(i0, i1)

    assert dr.all(t(2, 2, 2) ==  o0)
