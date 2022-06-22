#include "python.h"
#include "../ext/nanobind/src/buffer.h"

NAMESPACE_BEGIN(drjit)
NAMESPACE_BEGIN(detail)

namespace nb = nanobind;

static nb::detail::Buffer buffer;

static const char *type_name[] = {
    "Void", "Bool", "Int8", "UInt8", "Int16", "UInt16", "Int", "UInt",
    "Int64", "UInt64", "Pointer", "Float16", "Float", "Float64"
};

static const char *type_suffix[] = {
    "?", "b", "i8", "u8", "i16", "u16", "i", "u",
    "i64", "u64", "p", "f16", "f", "f64"
};

const char *array_name(array_metadata meta) {
    buffer.clear();

    if (!meta.is_tensor) {
        int ndim = meta.ndim;

        if (meta.is_llvm || meta.is_cuda)
            ndim--;

        const char *suffix = nullptr;

        if (ndim == 0) {
            buffer.put_dstr(type_name[meta.type]);
        } else {
            const char *prefix = "Array";
            if (meta.is_complex)
                prefix = "Complex";
            else if (meta.is_quaternion)
                prefix = "Quaternion";
            else if (meta.is_matrix)
                prefix = "Matrix";
            buffer.put_dstr(prefix);
            suffix = type_suffix[meta.type];
        }

        for (int i = 0; i < ndim; ++i) {
            if (meta.shape[i] == DRJIT_DYNAMIC)
                buffer.put('X');
            else
                buffer.put_uint32(meta.shape[i]);
        }

        if (suffix)
            buffer.put_dstr(suffix);
    } else {
        buffer.put_dstr("TensorX");
        buffer.put_dstr(type_suffix[meta.type]);
    }

    return buffer.get();
}

static nb::handle submodules[6];
static nb::handle array_base;
static void *sq_len_tmp, *sq_item_tmp, *sq_ass_item_tmp = nullptr;

void prepare_imports() {
    submodules[0] = nb::module_::import_("drjit");
    submodules[1] = nb::module_::import_("drjit.scalar");
    submodules[2] = nb::module_::import_("drjit.cuda");
    submodules[3] = nb::module_::import_("drjit.cuda.ad");
    submodules[4] = nb::module_::import_("drjit.llvm");
    submodules[5] = nb::module_::import_("drjit.llvm.ad");
    array_base = submodules[0].attr("ArrayBase");
}

static nb::handle array_module(array_metadata meta) {
    int index = 1;
    if (meta.is_llvm || meta.is_cuda) {
        index = meta.is_llvm ? 4 : 2;
        index += (int) meta.is_diff;
    }

    if (!array_base.is_valid())
        prepare_imports();
    return submodules[index];
}

const nb::handle array_get(array_metadata meta) {
    return array_module(meta).attr(array_name(meta));
}

extern nb::handle bind(const char *name, array_supplement &supp,
                       const std::type_info *type,
                       const std::type_info *value_type,
                       void (*copy)(void *, const void *),
                       void (*move)(void *, void *) noexcept,
                       void (*destruct)(void *) noexcept) noexcept {
    if (!name)
        name = array_name(supp.meta);

    nb::detail::type_data d;
    d.flags = (uint32_t) nb::detail::type_flags::has_scope |
              (uint32_t) nb::detail::type_flags::has_base_py |
              (uint32_t) nb::detail::type_flags::is_destructible |
              (uint32_t) nb::detail::type_flags::is_copy_constructible |
              (uint32_t) nb::detail::type_flags::is_move_constructible |
              (uint32_t) nb::detail::type_flags::is_final |
              (uint32_t) nb::detail::type_flags::has_type_callback |
              (uint32_t) nb::detail::type_flags::has_supplement;

    d.type_callback = [](PyType_Slot **s) noexcept {
        *(*s)++ = { Py_mp_length, (void *) sq_len_tmp };
        *(*s)++ = { Py_sq_length, (void *) sq_len_tmp };
        *(*s)++ = { Py_sq_item, (void *) sq_item_tmp };
        *(*s)++ = { Py_sq_ass_item, (void *) sq_ass_item_tmp };
    };

    if (move) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_move;
        d.move = move;
    }

    if (copy) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_copy;
        d.copy = copy;
    }

    if (destruct) {
        d.flags |= (uint32_t) nb::detail::type_flags::has_destruct;
        d.destruct = destruct;
    }

    d.align = supp.meta.talign;
    d.size = supp.meta.tsize_rel * supp.meta.talign;
    d.supplement = malloc(sizeof(array_supplement));
    d.name = strdup(name);
    d.type = type;

    d.scope = array_module(supp.meta).ptr();
    d.base_py = (PyTypeObject *) array_base.ptr();

    nb::handle h = nb::detail::nb_type_new(&d);

    VarType vt = (VarType) supp.meta.type;
    bool is_mask     = vt == VarType::Bool;
    bool is_float    = vt == VarType::Float16 ||
                       vt == VarType::Float32 ||
                       vt == VarType::Float64;

    nb::handle value_type_py;
    if (!value_type) {
        if (is_mask)
            value_type_py = &PyBool_Type;
        else if (is_float)
            value_type_py = &PyFloat_Type;
        else
            value_type_py = &PyLong_Type;
    } else {
        value_type_py = nb::detail::nb_type_lookup(value_type);
        if (!value_type_py.is_valid())
            nb::detail::fail("bind(): element type '%s' not found!",
                             value_type->name());
    }

    auto pred = [](PyTypeObject *tp, PyObject *o,
                   nb::detail::cleanup_list *) -> bool {
        detail::array_supplement &s =
            nb::type_supplement<detail::array_supplement>(tp);

        if (PySequence_Check(o)) {
            Py_ssize_t size = s.meta.shape[0], len = PySequence_Length(o);
            if (len == -1)
                PyErr_Clear();
            return size == DRJIT_DYNAMIC || len == size;
        } else if (s.value == Py_TYPE(o)) {
            return true;
        } else {
            VarType v = (VarType) s.meta.type;
            return (v == VarType::Float16 ||
                    v == VarType::Float32 ||
                    v == VarType::Float64) &&
                   Py_TYPE(o) == &PyLong_Type;
        }
    };
    nb::detail::implicitly_convertible(pred, type);

    nb::handle mask_type_py;
    if (is_mask) {
        mask_type_py = h.ptr();
    } else {
        array_metadata m2 = supp.meta;
        m2.type = (uint16_t) VarType::Bool;
        m2.is_vector = m2.is_complex = m2.is_quaternion =
            m2.is_matrix = false;
        mask_type_py = array_get(m2);
    }

    nb::handle array_type_py;
    if (!supp.meta.is_tensor && !supp.meta.is_complex &&
        !supp.meta.is_quaternion && !supp.meta.is_matrix) {
        array_type_py = h.ptr();
    } else {
        array_metadata m2 = supp.meta;
        if (supp.meta.is_tensor) {
            m2.shape[0] = DRJIT_DYNAMIC;
            m2.ndim = 1;
        }
        m2.is_vector = m2.is_complex = m2.is_quaternion =
            m2.is_matrix = m2.is_tensor = false;
        array_type_py = array_get(m2);
    }

    supp.value = (PyTypeObject *) value_type_py.ptr();
    supp.mask  = (PyTypeObject *) mask_type_py.ptr();
    supp.array = (PyTypeObject *) array_type_py.ptr();

    nb::type_supplement<detail::array_supplement>(h.ptr()) = supp;

    return h.ptr();
}

NAMESPACE_END(detail)
NAMESPACE_END(drjit)