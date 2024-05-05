import drjit as dr
import pytest

@pytest.test_arrays('uint32, jit, shape=(*)')
def test01_basic(t):

    @dr.freeze
    def func(x):
        return x + 1

    i0 = t(0, 1, 2)
    o0 = func(i0)

    assert dr.all(t(1, 2, 3) ==  o0)
