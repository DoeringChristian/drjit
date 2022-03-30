import drjit as dr
import pytest


def test01_init_zero():
    from drjit.scalar import Array0f, Array3f, ArrayXf
    a = Array0f()
    assert len(a) == 0

    a = Array3f()
    assert len(a) == 3
    for i in range(3):
        assert a[i] == 0

    a = ArrayXf()
    assert len(a) == 0
    with pytest.raises(IndexError) as ei:
        a[0]
    assert "entry 0 is out of bounds (the array is of size 0)." in str(ei.value)


def test02_init_sequence_static():
    from drjit.scalar import Array0f, Array1f, Array3f

    with pytest.raises(TypeError) as ei:
        Array3f("test")
    assert "input sequence has wrong size (expected 3, got 4)" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        Array0f(1)
    assert "too many arguments provided" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        a = Array3f((0, 1))
    assert "input sequence has wrong size" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        a = Array3f((0, 1, 2, 3))
    assert "input sequence has wrong size" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        Array3f("tst")
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        Array3f((0, "foo", 2))
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        Array3f(0, "foo", 2)
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    a = Array3f((0, 1, 2))
    for i in range(3):
        assert a[i] == i

    a = Array3f([0, 1, 2])
    for i in range(3):
        assert a[i] == i

    class my_list(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    a = Array3f(my_list([0, 1, 2]))
    for i in range(3):
        assert a[i] == i

    a = Array3f(0, 1, 2)
    for i in range(3):
        assert a[i] == i

    assert Array1f(1)[100] == 1


def test03_init_sequence_dynamic():
    from drjit.scalar import ArrayXf

    with pytest.raises(TypeError) as ei:
        ArrayXf("test")
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        ArrayXf((0, "foo", 2))
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        ArrayXf(0, "foo", 2)
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    a = ArrayXf((0, 1, 2, 3, 4))
    assert len(a) == 5
    for i in range(5):
        assert a[i] == i

    a = ArrayXf([0, 1, 2])
    assert len(a) == 3
    for i in range(3):
        assert a[i] == i

    class my_list(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    a = ArrayXf(my_list([0, 1, 2]))
    for i in range(3):
        assert a[i] == i

    a = ArrayXf(0, 1, 2)
    for i in range(3):
        assert a[i] == i

    for i in range(3):
        a[i] += 1

    for i in range(3):
        assert a[i] == i + 1

    assert ArrayXf(1)[100] == 1


def test04_indexing():
    from drjit.scalar import ArrayXf, Array3f
    a = ArrayXf([0, 1, 2])
    b = Array3f([0, 1, 2])
    assert a[-1] == 2 and b[-1] == 2

    with pytest.raises(IndexError) as ei:
        a[3]

    with pytest.raises(IndexError) as ei:
        a[-4]

    with pytest.raises(IndexError) as ei:
        b[3]

    with pytest.raises(IndexError) as ei:
        b[-4]


def test05_indexed_assignment():
    from drjit.scalar import ArrayXf, Array3f

    a = Array3f([0]*3)
    b = ArrayXf([0]*5)
    for i in range(3):
        a[i] = i
    for i in range(len(a)):
        a[i] += 1
    for i in range(len(b)):
        b[i] = i
    for i in range(len(b)):
        b[i] += 1
    for i in range(len(a)):
        assert a[i] == i + 1
    for i in range(len(b)):
        assert b[i] == i + 1


def test06_constructor_copy():
    from drjit.scalar import Array3f, ArrayXf
    a = Array3f(1, 2, 3)
    b = Array3f(a)
    c = ArrayXf(1, 2, 3, 4)
    d = ArrayXf(c)
    assert len(a) == len(b)
    assert len(c) == len(d)
    for i in range(len(a)):
        assert a[i] == b[i]
    for i in range(len(c)):
        assert c[i] == d[i]


def test07_constructor_broadcast():
    from drjit.scalar import Array3f, ArrayXf, ArrayXb
    a = Array3f(3)
    assert len(a) == 3 and a[0] == 3 and a[1] == 3 and a[2] == 3
    a = ArrayXf(3)
    assert len(a) == 1 and a[0] == 3
    a = ArrayXb(True)
    assert len(a) == 1 and a[0] == True


def test08_all_any():
    from drjit.scalar import Array2b, ArrayXb

    assert dr.all(True) == True
    assert dr.all(False) == False
    assert dr.any(True) == True
    assert dr.any(False) == False
    assert dr.any(()) == False
    assert dr.all(()) == True
    assert dr.all((True,)) == True
    assert dr.all((False,)) == False
    assert dr.any((True,)) == True
    assert dr.any((False,)) == False
    assert dr.all([True, True]) == True
    assert dr.all([True, False]) == False
    assert dr.all([False, False]) == False
    assert dr.any([True, True]) == True
    assert dr.any([True, False]) == True
    assert dr.any([False, False]) == False
    assert type(dr.all(Array2b(True, True))) is bool
    assert dr.all(Array2b(True, True)) == True
    assert dr.all(Array2b(True, True)) == True
    assert dr.all(Array2b(True, False)) == False
    assert dr.all(Array2b(False, False)) == False
    assert dr.any(Array2b(True, True)) == True
    assert dr.any(Array2b(True, False)) == True
    assert dr.any(Array2b(False, False)) == False
    assert type(dr.all(ArrayXb(True, True))) is ArrayXb
    assert len(dr.all(ArrayXb(True, True))) == 1
    assert dr.all(ArrayXb(True, True))[0] == True
    assert dr.all(ArrayXb(True, False))[0] == False
    assert dr.all(ArrayXb(False, False))[0] == False
    assert dr.any(ArrayXb(True, True))[0] == True
    assert dr.any(ArrayXb(True, False))[0] == True
    assert dr.any(ArrayXb(False, False))[0] == False

    assert type(dr.all(dr.llvm.Array1b(dr.llvm.Bool([True, False, False])))) is dr.llvm.Bool
    assert len(dr.all(dr.llvm.Array1b(dr.llvm.Bool([True, False, False])))) == 3
    assert type(dr.all_nested(dr.llvm.Array1b(dr.llvm.Bool([True, False, False])))) is dr.llvm.Bool
    assert len(dr.all_nested(dr.llvm.Array1b(dr.llvm.Bool([True, False, False])))) == 1

    with pytest.raises(TypeError) as ei:
        dr.all((True, "hello"))
    assert "unsupported operand type(s)" in str(ei.value)


def test09_implicit_to_bool():
    from drjit.scalar import Array3f, ArrayXf, Array3b, ArrayXb
    with pytest.raises(TypeError) as ei:
        bool(ArrayXf(1))
    assert "ArrayXf.__bool__(): implicit conversion to 'bool' is only supported for scalar mask arrays!" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        bool(Array3f(1))
    assert "Array3f.__bool__(): implicit conversion to 'bool' is only supported for scalar mask arrays!" in str(ei.value)

    with pytest.raises(RuntimeError) as ei:
        bool(ArrayXb(True, False))
    assert "ArrayXb.__bool__(): implicit conversion to 'bool' requires a scalar mask array (array size was 2)." in str(ei.value)

    assert bool(ArrayXb(True))
    assert not bool(ArrayXb(False))


@pytest.mark.parametrize('value', [(dr.scalar.Array3f, dr.scalar.Array3b), (dr.scalar.ArrayXf, dr.scalar.ArrayXb)])
def test10_comparison(value):
    c, m = value
    assert dr.all(c(1, 2, 3) == c(1, 2, 3))
    assert not dr.all(c(1, 2, 3) == c(1, 3, 3))
    assert dr.all(c(1, 2, 3) != c(4, 5, 6))
    assert not dr.all(c(1, 2, 3) != c(4, 2, 6))
    assert dr.any(c(1, 2, 3) == c(1, 2, 3))
    assert not dr.any(c(1, 2, 3) == c(4, 5, 6))
    assert dr.any(c(1, 2, 3) != c(1, 3, 3))
    assert not dr.any(c(1, 2, 3) != c(1, 2, 3))

    assert dr.all((c(1, 2, 3) < c(0, 2, 4)) == m(False, False, True))
    assert dr.all((c(1, 2, 3) <= c(0, 2, 4)) == m(False, True, True))
    assert dr.all((c(1, 2, 3) > c(0, 2, 4)) == m(True, False, False))
    assert dr.all((c(1, 2, 3) >= c(0, 2, 4)) == m(True, True, False))
    assert dr.all((c(1, 2, 3) == c(0, 2, 4)) == m(False, True, False))
    assert dr.all((c(1, 2, 3) != c(0, 2, 4)) == m(True, False, True))


def test11_shape():
    import drjit.scalar as s
    import drjit.llvm as l

    assert dr.shape(s.Array0f()) == (0,) and s.Array0f().shape == (0,)
    assert dr.shape(s.Array2f()) == (2,) and s.Array2f().shape == (2,)
    assert dr.shape(l.Float()) == (0,) and l.Float().shape == (0,)
    assert dr.shape(l.Float(1, 2, 3)) == (3,) and l.Float(1, 2, 3).shape == (3,)
    assert dr.shape(l.Array2f()) == (2, 0) and l.Array2f().shape == (2, 0)
    assert dr.shape(l.Array2f(l.Float(1, 2, 3))) == (2,3) and \
           l.Array2f(l.Float(1, 2, 3)).shape == (2,3)
    assert dr.shape(l.Array2f(l.Float(1, 2, 3),
                              l.Float(2, 3, 4))) == (2,3) and \
            l.Array2f(l.Float(1, 2, 3), l.Float(2, 3, 4)).shape == (2,3)
    assert dr.shape(l.Array2f(l.Float(1, 2, 3),
                              l.Float(2, 3))) is None and \
           l.Array2f(l.Float(1, 2, 3), l.Float(2, 3)).shape is None

def test11_repr():
  import drjit.scalar as s
  import drjit.llvm as l

  assert repr(s.Array0f()) == '[]'
  assert repr(s.ArrayXf()) == '[]'
  assert repr(s.Array1f(1)) == '[1]'
  assert repr(s.Array1f(1.5)) == '[1.5]'
  assert repr(s.Array2f(1, 2)) == '[1, 2]'
  assert repr(s.ArrayXf(1, 2)) == '[1, 2]'

  assert repr(l.Array0f()) == '[]'
  assert repr(l.ArrayXf()) == '[]'
  assert repr(l.Array1f(1)) == '[[1]]'
  assert repr(l.Array1f(1.5)) == '[[1.5]]'
  assert repr(l.Array2f(1, 2)) == '[[1, 2]]'
  assert repr(l.Array2f(1, [2, 3])) == '[[1, 2],\n' \
                                       ' [1, 3]]'
  assert repr(l.ArrayXf(1, 2)) == '[[1, 2]]'


def test12_binop_simple():
    from drjit.scalar import Array3f, ArrayXf, Array3u, ArrayXb
    a = Array3f(1, 2, 3)
    assert dr.all(a + a == Array3f(2, 4, 6))
    assert dr.all(a - a == Array3f(0, 0, 0))
    assert dr.all(a * a == Array3f(1, 4, 9))
    assert dr.all(a / a == Array3f(1, 1, 1))

    with pytest.raises(TypeError) as ei:
        a // a
    assert "unsupported operand type(s)" in str(ei.value)

    a = ArrayXf(1, 2, 3)
    assert dr.all(a + a == ArrayXf(2, 4, 6))
    assert dr.all(a - a == ArrayXf(0, 0, 0))
    assert dr.all(a * a == ArrayXf(1, 4, 9))
    assert dr.all(a / a == ArrayXf(1, 1, 1))
    a = Array3u(1, 2, 3)
    assert dr.all(a + a == Array3u(2, 4, 6))
    assert dr.all(a - a == Array3u(0, 0, 0))
    assert dr.all(a * a == Array3u(1, 4, 9))
    assert dr.all(a // a == Array3u(1, 1, 1))
    assert dr.all(a << 1 == Array3u(2, 4, 6))
    assert dr.all(a >> 1 == Array3u(0, 1, 1))

    with pytest.raises(TypeError) as ei:
        a / a
    assert "unsupported operand type(s)" in str(ei.value)

    assert dr.all(ArrayXb([True, True, False, False]) & ArrayXb([True, False, True, False]) == ArrayXb(True, False, False, False))
    assert dr.all(ArrayXb([True, True, False, False]) | ArrayXb([True, False, True, False]) == ArrayXb(True, True, True, False))
    assert dr.all(ArrayXb([True, True, False, False]) ^ ArrayXb([True, False, True, False]) == ArrayXb(False, True, True, False))


def test13_binop_broadcast():
    from drjit.scalar import Array3f, ArrayXf
    a = Array3f(1, 2, 3)
    b = a + 1
    assert dr.all(a + 1 == Array3f(2, 3, 4))
    assert dr.all(1 + a == Array3f(2, 3, 4))
    a = ArrayXf(1, 2, 3)
    b = a + 1
    assert dr.all(a + 1 == ArrayXf(2, 3, 4))
    assert dr.all(1 + a == ArrayXf(2, 3, 4))


def test14_binop_inplace():
    import drjit.scalar as s
    import drjit.llvm as l

    a = s.Array3f(1, 2, 3)
    b = s.Array3f(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == s.Array3f(3, 5, 4))
    a += 1
    assert a is c and dr.all(a == s.Array3f(4, 6, 5))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == s.Array3f(3, 4, 2))

    a = s.ArrayXf(1, 2, 3)
    b = s.ArrayXf(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == s.ArrayXf(3, 5, 4))
    a += 1
    assert a is c and dr.all(a == s.ArrayXf(4, 6, 5))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == s.ArrayXf(3, 4, 2))

    a = l.Float(1, 2, 3)
    b = l.Float(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == l.Float(3, 5, 4))
    a += 1
    assert a is c and dr.all(a == l.Float(4, 6, 5))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == l.Float(3, 4, 2))

    a = l.ArrayXf(1, 2, 3)
    b = l.ArrayXf(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == l.ArrayXf(3, 5, 4))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == l.ArrayXf(3, 4, 2))


@pytest.mark.parametrize('m', [dr.scalar, dr.llvm])
def test15_unop(m):
    assert dr.all(-m.ArrayXf(1, 2, 3) == m.ArrayXf(-1, -2, -3))
    assert dr.all(+m.ArrayXf(1, 2, 3) == m.ArrayXf(1, 2, 3))
    assert dr.all(abs(m.ArrayXf(1, -2, 3)) == m.ArrayXf(1, 2, 3))
    assert dr.all(-m.Array3f(1, 2, 3) == m.Array3f(-1, -2, -3))
    assert dr.all(+m.Array3f(1, 2, 3) == m.Array3f(1, 2, 3))
    assert dr.all(abs(m.Array3f(1, -2, 3)) == m.Array3f(1, 2, 3))
    assert dr.all(-m.Array3i(1, 2, 3) == m.Array3i(-1, -2, -3))
    assert dr.all(+m.Array3i(1, 2, 3) == m.Array3i(1, 2, 3))
    assert dr.all(~m.Array3i(1, 2, 3) == m.Array3i(-2, -3, -4))
    assert dr.all(abs(m.Array3i(1, -2, 3)) == m.Array3i(1, 2, 3))
    assert dr.all(~m.Array3b(True, False, True) == m.Array3b(False, True, False))
    assert dr.all(~m.ArrayXb(True, False, True) == m.ArrayXb(False, True, False))



def test16_type_promotion_errors():
    from drjit.scalar import Array3f
    a = Array3f()
    with pytest.raises(TypeError) as ei:
        a + "asdf"
    assert "Array3f.__add__(): encountered an unsupported argument of type 'str' (must be a Dr.Jit array or a Python scalar)" in str(ei.value)

    with pytest.raises(RuntimeError) as ei:
        a + 2**100
    assert "integer overflow during type promotion" in str(ei.value)


def test18_traits():
    import drjit.scalar as s
    import drjit.llvm as l
    from drjit import Dynamic

    assert not dr.is_array_v(()) and not dr.is_array_v(1.0)
    assert dr.is_array_v(s.Array3f) and dr.is_array_v(s.Array3f())
    assert dr.is_array_v(s.ArrayXf) and dr.is_array_v(s.ArrayXf())
    assert dr.is_array_v(l.Array3f) and dr.is_array_v(l.Array3f())
    assert dr.is_array_v(l.ArrayXf) and dr.is_array_v(l.ArrayXf())
    assert dr.array_size_v(1) == 1
    assert dr.array_size_v("test") == 1
    assert dr.array_size_v(s.Array3f) == 3 and dr.array_size_v(s.Array3f()) == 3 
    assert dr.array_size_v(l.Array3f) == 3 and dr.array_size_v(l.Array3f()) == 3 
    assert dr.array_size_v(s.ArrayXf) == Dynamic and dr.array_size_v(s.ArrayXf()) == Dynamic 
    assert dr.array_size_v(l.ArrayXf) == Dynamic and dr.array_size_v(l.ArrayXf()) == Dynamic 

    assert dr.array_depth_v(1) == 0
    assert dr.array_depth_v("test") == 0
    assert dr.array_depth_v(s.Array3f) == 1 and dr.array_depth_v(s.Array3f()) == 1
    assert dr.array_depth_v(s.ArrayXf) == 1 and dr.array_depth_v(s.ArrayXf()) == 1
    assert dr.array_depth_v(l.Float) == 1 and dr.array_depth_v(l.Float()) == 1
    assert dr.array_depth_v(l.Array3f) == 2 and dr.array_depth_v(l.Array3f()) == 2
    assert dr.array_depth_v(l.ArrayXf) == 2 and dr.array_depth_v(l.ArrayXf()) == 2

    assert dr.scalar_t(1) is int
    assert dr.scalar_t("test") is str
    assert dr.scalar_t(s.Array3f) is float and dr.scalar_t(s.Array3f()) is float
    assert dr.scalar_t(s.Array3b) is bool and dr.scalar_t(s.Array3b()) is bool
    assert dr.scalar_t(s.ArrayXf) is float and dr.scalar_t(s.ArrayXf()) is float
    assert dr.scalar_t(s.ArrayXb) is bool and dr.scalar_t(s.ArrayXb()) is bool
    assert dr.scalar_t(l.Float) is float and dr.scalar_t(l.Float()) is float
    assert dr.scalar_t(l.Bool) is bool and dr.scalar_t(l.Bool()) is bool
    assert dr.scalar_t(l.Array3f) is float and dr.scalar_t(l.Array3f()) is float
    assert dr.scalar_t(l.Array3b) is bool and dr.scalar_t(l.Array3b()) is bool
    assert dr.scalar_t(l.ArrayXf) is float and dr.scalar_t(l.ArrayXf()) is float
    assert dr.scalar_t(l.ArrayXb) is bool and dr.scalar_t(l.ArrayXb()) is bool

    assert dr.value_t(1) is int
    assert dr.value_t("test") is str
    assert dr.value_t(s.Array3f) is float and dr.value_t(s.Array3f()) is float
    assert dr.value_t(s.Array3b) is bool and dr.value_t(s.Array3b()) is bool
    assert dr.value_t(s.ArrayXf) is float and dr.value_t(s.ArrayXf()) is float
    assert dr.value_t(s.ArrayXb) is bool and dr.value_t(s.ArrayXb()) is bool
    assert dr.value_t(l.Float) is float and dr.value_t(l.Float()) is float
    assert dr.value_t(l.Bool) is bool and dr.value_t(l.Bool()) is bool
    assert dr.value_t(l.Array3f) is l.Float and dr.value_t(l.Array3f()) is l.Float
    assert dr.value_t(l.Array3b) is l.Bool and dr.value_t(l.Array3b()) is l.Bool
    assert dr.value_t(l.ArrayXf) is l.Float and dr.value_t(l.ArrayXf()) is l.Float
    assert dr.value_t(l.ArrayXb) is l.Bool and dr.value_t(l.ArrayXb()) is l.Bool

    assert dr.mask_t(1) is bool
    assert dr.mask_t("test") is bool
    assert dr.mask_t(s.Array3f) is s.Array3b and dr.mask_t(s.Array3f()) is s.Array3b
    assert dr.mask_t(s.Array3b) is s.Array3b and dr.mask_t(s.Array3b()) is s.Array3b
    assert dr.mask_t(s.ArrayXf) is s.ArrayXb and dr.mask_t(s.ArrayXf()) is s.ArrayXb
    assert dr.mask_t(s.ArrayXb) is s.ArrayXb and dr.mask_t(s.ArrayXb()) is s.ArrayXb
    assert dr.mask_t(l.Float) is l.Bool and dr.mask_t(l.Float()) is l.Bool
    assert dr.mask_t(l.Bool) is l.Bool and dr.mask_t(l.Bool()) is l.Bool
    assert dr.mask_t(l.Array3f) is l.Array3b and dr.mask_t(l.Array3f()) is l.Array3b
    assert dr.mask_t(l.Array3b) is l.Array3b and dr.mask_t(l.Array3b()) is l.Array3b
    assert dr.mask_t(l.ArrayXf) is l.ArrayXb and dr.mask_t(l.ArrayXf()) is l.ArrayXb
    assert dr.mask_t(l.ArrayXb) is l.ArrayXb and dr.mask_t(l.ArrayXb()) is l.ArrayXb

    assert dr.is_integral_v(1) and dr.is_integral_v(int)
    assert dr.is_integral_v(s.Array3i()) and dr.is_integral_v(s.Array3i)
    assert not dr.is_integral_v(1.0) and not dr.is_integral_v(float)
    assert not dr.is_integral_v(s.Array3f()) and not dr.is_integral_v(s.Array3f)
    assert not dr.is_integral_v("str") and not dr.is_integral_v(str)
    assert not dr.is_integral_v(False) and not dr.is_integral_v(bool)
    assert not dr.is_integral_v(s.Array3b()) and not dr.is_integral_v(s.Array3b)

    assert not dr.is_float_v(1) and not dr.is_float_v(int)
    assert not dr.is_float_v(s.Array3i()) and not dr.is_float_v(s.Array3i)
    assert dr.is_float_v(1.0) and dr.is_float_v(float)
    assert dr.is_float_v(s.Array3f()) and dr.is_float_v(s.Array3f)
    assert not dr.is_float_v("str") and not dr.is_float_v(str)
    assert not dr.is_float_v(False) and not dr.is_float_v(bool)
    assert not dr.is_float_v(s.Array3b()) and not dr.is_float_v(s.Array3b)

    assert dr.is_arithmetic_v(1) and dr.is_arithmetic_v(int)
    assert dr.is_arithmetic_v(s.Array3i()) and dr.is_arithmetic_v(s.Array3i)
    assert dr.is_arithmetic_v(1.0) and dr.is_arithmetic_v(float)
    assert dr.is_arithmetic_v(s.Array3f()) and dr.is_arithmetic_v(s.Array3f)
    assert not dr.is_arithmetic_v("str") and not dr.is_arithmetic_v(str)
    assert not dr.is_arithmetic_v(False) and not dr.is_arithmetic_v(bool)
    assert not dr.is_arithmetic_v(s.Array3b()) and not dr.is_arithmetic_v(s.Array3b)

    assert not dr.is_mask_v(1) and not dr.is_mask_v(int)
    assert not dr.is_mask_v(s.Array3i()) and not dr.is_mask_v(s.Array3i)
    assert not dr.is_mask_v(1.0) and not dr.is_mask_v(float)
    assert not dr.is_mask_v(s.Array3f()) and not dr.is_mask_v(s.Array3f)
    assert not dr.is_mask_v("str") and not dr.is_mask_v(str)
    assert dr.is_mask_v(False) and dr.is_mask_v(bool)
    assert dr.is_mask_v(s.Array3b()) and dr.is_mask_v(s.Array3b)


def test19_select():
    import drjit.scalar as s
    import drjit.llvm as l
    assert dr.select(True, "hello", "world") == "hello"
    result = dr.select(s.Array2b(True, False), 1, 2)
    assert isinstance(result, s.Array2i) and dr.all(result == s.Array2i(1, 2))
    result = dr.select(s.Array2b(True, False), 1, 2.0)
    assert isinstance(result, s.Array2f) and dr.all(result == s.Array2f(1, 2))
    result = dr.select(s.ArrayXb(True, False), 1, 2)
    assert isinstance(result, s.ArrayXi) and dr.all(result == s.ArrayXi(1, 2))
    result = dr.select(s.ArrayXb(True, False), 1, 2.0)
    assert isinstance(result, s.ArrayXf) and dr.all(result == s.ArrayXf(1, 2))

    result = dr.select(s.Array2b(True, False), s.Array2i(3, 4), s.Array2i(5, 6))
    assert isinstance(result, s.Array2i) and dr.all(result == s.Array2i(3, 6))
    result = dr.select(s.Array2b(True, False), s.Array2i(3, 4), s.Array2f(5, 6))
    assert isinstance(result, s.Array2f) and dr.all(result == s.Array2f(3, 6))
    result = dr.select(s.ArrayXb(True, False), s.ArrayXi(3, 4), s.ArrayXi(5, 6))
    assert isinstance(result, s.ArrayXi) and dr.all(result == s.ArrayXi(3, 6))
    result = dr.select(s.ArrayXb(True, False), s.ArrayXi(3, 4), s.ArrayXf(5, 6))
    assert isinstance(result, s.ArrayXf) and dr.all(result == s.ArrayXf(3, 6))

    result = dr.select(l.Array2b(True, False), 1, 2)
    assert isinstance(result, l.Array2i) and dr.all(result == l.Array2i(1, 2))
    result = dr.select(l.Array2b(True, False), 1, 2.0)
    assert isinstance(result, l.Array2f) and dr.all(result == l.Array2f(1, 2))
    result = dr.select(l.ArrayXb(True, False), 1, 2)
    assert isinstance(result, l.ArrayXi) and dr.all(result == l.ArrayXi(1, 2))
    result = dr.select(l.ArrayXb(True, False), 1, 2.0)
    assert isinstance(result, l.ArrayXf) and dr.all(result == l.ArrayXf(1, 2))

    result = dr.select(l.Array2b(True, False), l.Array2i(3, 4), l.Array2i(5, 6))
    assert isinstance(result, l.Array2i) and dr.all(result == l.Array2i(3, 6))
    result = dr.select(l.Array2b(True, False), l.Array2i(3, 4), l.Array2f(5, 6))
    assert isinstance(result, l.Array2f) and dr.all(result == l.Array2f(3, 6))
    result = dr.select(l.ArrayXb(True, False), l.ArrayXi(3, 4), l.ArrayXi(5, 6))
    assert isinstance(result, l.ArrayXi) and dr.all(result == l.ArrayXi(3, 6))
    result = dr.select(l.ArrayXb(True, False), l.ArrayXi(3, 4), l.ArrayXf(5, 6))
    assert isinstance(result, l.ArrayXf) and dr.all(result == l.ArrayXf(3, 6))


def test20_component_access():
    from drjit.scalar import Array3f, Array4f
    from drjit.llvm import Array3f as Array3fL

    a = Array4f(4, 5, 6, 7)
    assert a.x == 4 and a.y == 5 and a.z == 6 and a.w == 7
    a.x, a.y, a.z, a.w = 1, 2, 3, 4
    assert a.x == 1 and a.y == 2 and a.z == 3 and a.w == 4
    a = Array3f(1, 2, 3)
    assert a.x == 1 and a.y == 2 and a.z == 3

    assert a.index == 0 and a.index_ad == 0

    with pytest.raises(TypeError) as ei:
        a.w == 4
    assert "Array3f: array does not have a 'w' component!" in str(ei.value)

    a = Array3fL(1, 2, 3)
    assert a.index == 0 and a.index_ad == 0
    assert a.x.index != 0 and a.y.index != 0 and a.z.index != 0


def test21_zero_or_full():
    import drjit.scalar as s
    import drjit.llvm as l
    assert type(dr.zero(dtype=int)) is int and dr.zero(dtype=int) == 0
    assert type(dr.zero(dtype=int, shape=(1,))) is int and dr.zero(dtype=int, shape=(1,)) == 0
    assert type(dr.zero(float)) is float and dr.zero(float) == 0.0
    assert type(dr.zero(float, shape=(1,))) is float and dr.zero(float, shape=(1,)) == 0.0
    with pytest.raises(TypeError) as ei:
        dr.zero(str)
    assert "Unsupported dtype!" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.zero(5)
    assert "incompatible function arguments" in str(ei.value)

    with pytest.raises(RuntimeError) as ei:
        dr.zero(int, shape=(2,))
    with pytest.raises(RuntimeError) as ei:
        dr.zero(int, shape=(1,1))

    assert type(dr.zero(s.Array3f)) is s.Array3f and dr.all(dr.zero(s.Array3f) == s.Array3f(0))
    assert type(dr.zero(s.Array3f, shape=(3,))) is s.Array3f and dr.all(dr.zero(s.Array3f, shape=(3,)) == s.Array3f(0))
    assert type(dr.zero(l.Array3f)) is l.Array3f and dr.all(dr.zero(l.Array3f) == l.Array3f(0))
    assert type(dr.zero(l.Array3f, shape=(3, 5))) is l.Array3f and dr.shape(dr.zero(l.Array3f, shape=(3, 5))) == (3, 5)
    assert type(dr.zero(l.Array3f, shape=10)) is l.Array3f and dr.shape(dr.zero(l.Array3f, shape=10)) == (3, 10)
    assert type(dr.zero(l.ArrayXf, shape=(8, 5))) is l.ArrayXf and dr.shape(dr.zero(l.ArrayXf, shape=(8, 5))) == (8, 5)
    assert type(dr.zero(l.Array3b, shape=10)) is l.Array3b and dr.shape(dr.zero(l.Array3b, shape=10)) == (3, 10)
    assert type(dr.zero(l.ArrayXb, shape=(8, 5))) is l.ArrayXb and dr.shape(dr.zero(l.ArrayXb, shape=(8, 5))) == (8, 5)
    assert type(dr.full(l.ArrayXf, value=123, shape=(8, 5))) is l.ArrayXf and dr.all_nested(dr.zero(l.ArrayXf, value=123, shape=(8, 5)) == 123)


#@pytest.mark.parametrize('name', ['sqrt', 'cbrt', 'sin', 'cos', 'tan', 'asin',
#                                  'acos', 'atan', 'sinh', 'cosh', 'tanh',
#                                  'asinh', 'acosh', 'atanh', 'exp', 'exp2',
#                                  'log', 'log2', 'floor', 'ceil', 'trunc',
#                                  'round', 'rcp', 'rsqrt'])
#def test17_spotcheck_unary_math(name):
#    from drjit.scalar import ArrayXf, PCG32
#    import math
#    func_ref = getattr(math, name, None)
#    if name == 'cbrt':
#        func_ref = lambda x: x**(1/3)
#    elif name == 'exp2':
#        func_ref = lambda x: 2**x
#    elif name == 'log2':
#        log2 = lambda x: math.log(x) / math.log(2)
#    elif name == 'round':
#        func_ref = round
#    elif name == 'rcp':
#        func_ref = lambda x : 1/x
#    elif name == 'rsqrt':
#        func_ref = lambda x : math.sqrt(1/x)
#
#    rng = PCG32()
#    x = ArrayXf((rng.next_float32() for i in range(10)))
#    if name == 'acosh':
#        x += 1
#    ref = ArrayXf([func_ref(y) for y in x])
#    func = getattr(dr, name)
#    value = func(x)
#    value_2 = ArrayXf(func(y) for y in x)
#    assert dr.allclose(value, func)
#
