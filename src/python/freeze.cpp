#include "freeze.h"
#include "base.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "drjit/extra.h"
#include "eval.h"
#include "listobject.h"
#include "nanobind/intrusive/counter.h"
#include "nanobind/nanobind.h"
#include "object.h"
#include "pyerrors.h"
#include "tupleobject.h"
#include <tsl/robin_map.h>
#include <vector>

static const char *doc_freeze = R"(
    
)";

/// Stores information about python objects, such as their type, their number of
/// sub-elements or their field keys. This can be used to reconstruct a pytree
/// from a flattened variable array.
struct Layout {
    /// Nanobind type of the container/variable
    nb::type_object type;
    /// Number of members in this container
    size_t num = 0;
    /// Optional field identifiers of the container
    /// for example: keys in dictionary
    std::vector<nb::object> fields;
    /// Optional drjit type of the variable
    VarType vt = VarType::Void;
    /// Optional evaluation state of the variable
    VarState vs = VarState::Invalid;
    /// Weather the variable is an array with a single entry.
    /// Such arrays are handled differently by the compiler.
    bool singleton_array = false;
    bool unaligned = false;
    /// The literal data
    uint64_t literal = 0;
    /// The index in the flat_variables array of this variable.
    /// This can be used to determine aliasing.
    uint32_t index = 0;

    /// If a non drjit type is passed as function arguments or result, we simply
    /// cache it here.
    /// TODO: possibly do the same for literals?
    nb::object py_object = nb::none();

    bool operator==(const Layout &rhs) const {
        if (!(this->type.equal(rhs.type))) {
            jit_log(LogLevel::Warn, "    type");
            return false;
        }
        if (this->num != rhs.num) {
            jit_log(LogLevel::Warn, "    num: %u != %u", this->num, rhs.num);
            return false;
        }
        if (this->fields.size() != rhs.fields.size()) {
            jit_log(LogLevel::Warn, "    fields.size");
            return false;
        }
        for (uint32_t i = 0; i < this->fields.size(); ++i) {
            if (!(this->fields[i].equal(rhs.fields[i]))) {
                jit_log(LogLevel::Warn, "    fields[%u]", i);
                return false;
            }
        }
        if (this->vt != rhs.vt) {
            jit_log(LogLevel::Warn, "    vt");
            return false;
        }
        if (this->vs != rhs.vs) {
            jit_log(LogLevel::Warn, "    vs: %u != %u", (uint32_t)this->vs,
                    (uint32_t)rhs.vs);
            return false;
        }
        if (this->singleton_array != rhs.singleton_array) {
            jit_log(LogLevel::Warn, "    singleton_array");
            return false;
        }
        if (this->unaligned != rhs.unaligned) {
            jit_log(LogLevel::Warn, "    unaligned");
            return false;
        }
        if (this->index != rhs.index) {
            jit_log(LogLevel::Warn, "    index");
            return false;
        }
        if (this->literal != rhs.literal) {
            jit_log(LogLevel::Warn, "    literal");
            return false;
        }
        if (!this->py_object.equal(rhs.py_object)) {
            jit_log(LogLevel::Warn, "    object");
            return false;
        }
        return true;
    }
};

nb::object init_from_index(nb::type_object type, uint32_t variable_index) {
    auto result = nb::inst_alloc_zero(type);
    const ArraySupplement &s = supp(result.type());
    s.init_index(variable_index, inst_ptr(result));
    return result;
}

struct FlatVariables {

    // Variables, used to iterate over the variables/layouts when constructing
    // python objects
    uint32_t layout_index = 0;

    /// The flattened variable indices of the input/output to a frozen function
    std::vector<uint32_t> variables;
    /// Mapping from drjit variable index to index in flat variables
    tsl::robin_map<uint32_t, uint32_t> index_to_flat;
    /// This saves information about the type, size and fields of pytree
    /// objects. The information is stored in DFS order.
    std::vector<Layout> layout;
    JitBackend backend = JitBackend::None;

    // Wether variables should be borrowed, instead of stealing them
    bool borrow = true;

    FlatVariables() {
    }
    FlatVariables(bool borrow) : borrow(borrow) {
    }

    void clear() {
        this->layout_index = 0;
        this->variables.clear();
        this->index_to_flat.clear();
        this->layout.clear();
        this->backend = JitBackend::None;
    }
    void drop_variables() {
        for (uint32_t &index : this->variables) {
            jit_var_dec_ref(index);
        }
    }

    /**
     * Adds a variable to the flattened array, deduplicating it.
     * This allows for checking for aliasing conditions, as aliasing inputs map
     * to the same flat variable index.
     */
    uint32_t add_variable(uint32_t variable_index) {
        auto it = this->index_to_flat.find(variable_index);

        if (it == this->index_to_flat.end()) {
            uint32_t flat_index = this->variables.size();
            jit_log(LogLevel::Info, "    aliasing var(%u) -> flat_var(%u)",
                    variable_index, flat_index);

            // NOTE: an alternative to borrowing here would be to make `refcount
            // > 1` part of the layout, which would allow us to selectively
            // enable COW if it is necessary.
            if (borrow)
                jit_var_inc_ref(variable_index);
            this->variables.push_back(variable_index);

            this->index_to_flat.insert({variable_index, flat_index});
            return flat_index;
        } else {
            uint32_t flat_index = it.value();
            jit_log(
                LogLevel::Info,
                "collect(): Found aliasing condition var(%u) -> flat_var(%u)",
                variable_index, flat_index);
            return flat_index;
        }
    }

    /**
     * Add flat drjit variable by index.
     */
    void add_flat_dr_var_index(uint32_t index, nb::handle tp = nb::none()) {
        JitBackend var_backend = (JitBackend)jit_var_backend(index);

        if (this->backend == var_backend || this->backend == JitBackend::None) {
            this->backend = var_backend;
        } else {
            nb::raise("freeze(): backend missmatch error (backend of this "
                      "variable %u does not match backend of others %u)!",
                      (uint32_t)var_backend, (uint32_t)this->backend);
        }

        jit_log(LogLevel::Info,
                "collect(): collecting var(%u, backend=%u, type=%u)", index,
                var_backend, jit_var_type(index));

        if (jit_var_type(index) == VarType::Pointer) {
            // In order to support pointer inputs,
            // we would have to get the source variable, handle the case when
            // it's rc > 1 and potentially create a new pointer pointing to the
            // new source variable. Then we could add the new variable to the
            // flat variables.
            nb::raise("Pointer inputs not yet supported!");
        }

        raise_if(ad_grad_enabled(index), "Passing gradients into/out of a "
                                         "frozen function is not supported!");

        Layout layout;
        VarState vs = jit_var_state(index);
        layout.type = nb::borrow<nb::type_object>(tp);
        layout.vs = vs;
        layout.vt = jit_var_type(index);

        if (vs == VarState::Literal) {
            jit_log(LogLevel::Info, "    vs=Literal");

            raise_if(jit_var_size(index) > 1,
                     "collect(): Size larger than 1 not supported yet!");

            // NOTE: This should not cause any new operations to be recorded, as
            // we have stablished that the variable is a literal.
            jit_var_read(index, 0, &layout.literal);
        } else if (vs == VarState::Evaluated) {
            jit_log(LogLevel::Info, "    vs=%s", jit_var_kind_name(index));
            layout.index = this->add_variable(index);
            layout.singleton_array = jit_var_size(index) == 1;
            layout.unaligned = jit_var_is_unaligned(index);
        } else {

            jit_log(LogLevel::Error,
                    "collect(): found variable %u in unsupported state %u!",
                    index, (uint32_t)vs);
            nb::raise("");
        }
        this->layout.push_back(layout);
    }

    void add_flat_dr_var(nb::handle h) {
        auto s = supp(h.type());

        raise_if(s.index == nullptr, "freeze(): ArraySupplement index function "
                                     "pointer is nullptr.");
        // raise_if(s.is_class,
        //          "freeze(): Class variables are not yet supported!");

        uint64_t index = s.index(inst_ptr(h));

        this->add_flat_dr_var_index(index, h.type());
    }

    /**
     * Traverse a variable and it's derived variables.
     */
    void add_dr_class_var(nb::handle h) {
        const ArraySupplement &s = supp(h.type());

        jit_log(LogLevel::Debug,
                "traverse(): Found class variable on backend %u", backend);

        // Add the base type (hopefully)
        size_t layout_index = this->layout.size();
        add_flat_dr_var(h);

        JitBackend backend = (JitBackend)s.backend;
        nb::str domain = nb::borrow<nb::str>(nb::getattr(h.type(), "Domain"));

        uint32_t id_bound = jit_registry_id_bound(backend, domain.c_str());

        jit_log(LogLevel::Debug,
                "traverse(): Traversing %u instances of domain \"%s\".",
                id_bound, domain.c_str());

        // We use types of the subtypes as fields.
        std::vector<nb::object> fields;
        for (uint32_t id = 0; id < id_bound; ++id) {
            void *ptr = jit_registry_ptr(backend, domain.c_str(), id + 1);

            // WARN: very unsafe cast!
            nb::intrusive_base *base = (nb::intrusive_base *)ptr;
            nb::handle inst_obj = base->self_py();

            fields.push_back(nb::borrow(inst_obj.type()));

            traverse(inst_obj);
        }

        this->layout[layout_index].num = id_bound;
        this->layout[layout_index].fields = std::move(fields);
    }

    void add_dr_var(nb::handle h) {
        nb::handle tp = h.type();

        auto s = supp(h.type());

        // Handle class var
        if (s.is_class) {
            add_dr_class_var(h);
        } else {
            add_flat_dr_var(h);
        }
    }

    /**
     * Traverses a PyTree in DFS order, and records it's layout in the
     * `layout` vector.
     *
     * When hitting a drjit primitive type, it calls the
     * `collect` method, which will add their indices to the `flat_variables`
     * vector.
     * The collect method will also record metadata about the drjit variable in
     * the layout.
     * Therefore, the layout can be used as an identifier to the recording of
     * the frozen function.
     */
    void traverse(nb::handle h) {
        nb::handle tp = h.type();

        try {
            if (is_drjit_type(tp)) {
                const ArraySupplement &s = supp(tp);
                if (s.is_tensor) {
                    add_dr_var(h);
                } else if (s.ndim != 1) {
                    Py_ssize_t len = s.shape[0];
                    if (len == DRJIT_DYNAMIC)
                        len = s.len(inst_ptr(h));

                    Layout layout;
                    layout.type = nb::borrow<nb::type_object>(tp);
                    layout.num = len;
                    this->layout.push_back(layout);

                    for (Py_ssize_t i = 0; i < len; ++i)
                        traverse(nb::steal(s.item(h.ptr(), i)));
                } else {
                    add_dr_var(h);
                }
            } else if (tp.is(&PyTuple_Type)) {
                nb::tuple tuple = nb::borrow<nb::tuple>(h);

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.num = tuple.size();
                this->layout.push_back(layout);

                for (nb::handle h2 : tuple) {
                    traverse(h2);
                }
            } else if (tp.is(&PyList_Type)) {
                nb::list list = nb::borrow<nb::list>(h);

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.num = list.size();
                this->layout.push_back(layout);

                for (nb::handle h2 : list) {
                    traverse(h2);
                }
            } else if (tp.is(&PyDict_Type)) {
                nb::dict dict = nb::borrow<nb::dict>(h);

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.num = dict.size();
                layout.fields.reserve(layout.num);
                for (auto k : dict.keys()) {
                    layout.fields.push_back(nb::borrow(k));
                }
                this->layout.push_back(layout);

                for (auto [k, v] : dict) {
                    traverse(v);
                }
            } else if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.num = ds.size();
                layout.fields.reserve(layout.num);
                for (auto k : ds.keys()) {
                    layout.fields.push_back(nb::borrow(k));
                }
                this->layout.push_back(layout);

                for (auto [k, v] : ds) {
                    traverse(nb::getattr(h, k));
                }
            } else if (nb::object df = get_dataclass_fields(tp);
                       df.is_valid()) {

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                for (auto field : df) {
                    nb::object k = field.attr(DR_STR(name));
                    layout.fields.push_back(nb::borrow(k));
                }
                layout.num = layout.fields.size();
                this->layout.push_back(layout);

                for (nb::handle field : df) {
                    nb::object k = field.attr(DR_STR(name));
                    traverse(nb::getattr(h, k));
                }
            } else if (nb::object cb = get_traverse_cb_ro(tp); cb.is_valid()) {
                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                size_t layout_index = this->layout.size();
                this->layout.push_back(layout);

                uint32_t num_fileds = 0;

                cb(h, nb::cpp_function([&](uint64_t index) {
                       num_fileds++;
                       this->add_flat_dr_var_index(index, nb::none());
                   }));

                // Update layout number of fields
                this->layout[layout_index].num = num_fileds;
            } else if (tp.is(&_PyNone_Type)) {
                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                this->layout.push_back(layout);
            } else {
                jit_log(LogLevel::Warn,
                        "traverse(): You passed a value to a frozen function, "
                        "that could not be converted to Dr.Jit types. This is "
                        "not recommended and the value will be cached.",
                        nb::type_name(tp).c_str());

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.py_object = nb::borrow<nb::object>(h);
                this->layout.push_back(layout);
            }
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_RuntimeError,
                           "FlatVariables::traverse(): error encountered while "
                           "processing an argument "
                           "of type '%U' (see above).",
                           nb::type_name(tp).ptr());
        } catch (const std::exception &e) {
            nb::chain_error(PyExc_RuntimeError,
                            "FlatVariables::traverse(): error encountered "
                            "while processing an argument "
                            "of type '%U': %s",
                            nb::type_name(tp).ptr(), e.what());
            nb::raise_python_error();
        }
    }

    uint64_t construct_dr_var_index(Layout &layout) {
        if (layout.vs == VarState::Literal) {
            uint32_t index =
                jit_var_literal(this->backend, layout.vt, &layout.literal);

            return index;
        } else {
            uint32_t index = this->variables[layout.index];

            return index;
        }
    }

    nb::object construct_flat_dr_var(Layout &layout) {
        if (layout.vs == VarState::Literal) {
            uint32_t index =
                jit_var_literal(this->backend, layout.vt, &layout.literal);

            auto result = nb::inst_alloc_zero(layout.type);
            const ArraySupplement &s = supp(result.type());
            s.init_index(index, inst_ptr(result));

            // Have to decrement reference, as it is not part of `variables` and
            // will not be freed
            jit_var_dec_ref(index);
            return result;
        } else {
            uint32_t index = this->variables[layout.index];

            auto result = nb::inst_alloc_zero(layout.type);
            const ArraySupplement &s = supp(result.type());
            s.init_index(index, inst_ptr(result));
            return result;
        }
    }

    nb::object construct_dr_class_var(Layout &layout) {
        const ArraySupplement &s = supp(layout.type);

        JitBackend backend = (JitBackend)s.backend;
        nb::str domain =
            nb::borrow<nb::str>(nb::getattr(layout.type, "Domain"));

        uint32_t id_bound = jit_registry_id_bound(backend, domain.c_str());
        if (id_bound != layout.num) {
            jit_fail("Number of sub-types registered with backend changed "
                     "during recording!");
        }

        nb::object result = construct_flat_dr_var(layout);

        for (uint32_t id = 0; id < id_bound; ++id) {
            void *ptr = jit_registry_ptr(backend, domain.c_str(), id + 1);

            // WARN: very unsafe cast!
            nb::intrusive_base *base = (nb::intrusive_base *)ptr;
            nb::handle inst_obj = base->self_py();

            // TODO: what should we acctually do in this case?
            construct();
        }

        return result;
    }

    nb::object construct_dr_var(Layout &layout) {
        const ArraySupplement &s = supp(layout.type);

        if (s.is_class) {
            return construct_dr_class_var(layout);
        } else {
            return construct_flat_dr_var(layout);
        }
    }

    /**
     * This is the counterpart to the traverse method.
     * Given a layout vector and flat_variables, it re-constructs the PyTree.
     */
    nb::object construct() {
        if (this->layout.size() == 0) {
            return nb::none();
        }

        Layout &layout = this->layout[layout_index++];
        if (layout.type.is(nb::none().type())) {
            return nb::none();
        }
        if (is_drjit_type(layout.type)) {
            const ArraySupplement &s = supp(layout.type);

            if (s.is_tensor) {
                return construct_dr_var(layout);
            } else if (s.ndim != 1) {
                auto result = nb::inst_alloc_zero(layout.type);
                dr::ArrayBase *p = inst_ptr(result);
                size_t size = s.shape[0];
                if (size == DRJIT_DYNAMIC) {
                    size = s.len(p);
                    s.init(size, p);
                }
                for (size_t i = 0; i < size; ++i) {
                    result[i] = construct();
                }
                return result;
            } else {
                return construct_dr_var(layout);
            }
        } else if (layout.type.is(&PyTuple_Type)) {
            nb::list list;
            for (uint32_t i = 0; i < layout.num; ++i) {
                list.append(construct());
            }
            return nb::tuple(list);
        } else if (layout.type.is(&PyList_Type)) {
            nb::list list;
            for (uint32_t i = 0; i < layout.num; ++i) {
                list.append(construct());
            }
            return list;
        } else if (layout.type.is(&PyDict_Type)) {
            nb::dict dict;
            for (auto k : layout.fields) {
                dict[k] = construct();
            }
            return dict;
        } else if (nb::dict ds = get_drjit_struct(layout.type); ds.is_valid()) {
            nb::object tmp = layout.type();
            // TODO: validation against `ds`
            for (auto k : layout.fields) {
                nb::setattr(tmp, k, construct());
            }
            return tmp;
        } else if (nb::object df = get_dataclass_fields(layout.type);
                   df.is_valid()) {
            nb::dict dict;
            for (auto k : layout.fields) {
                dict[k] = construct();
            }
            return layout.type(**dict);
        } else if (nb::object cb = get_traverse_cb_rw(layout.type);
                   cb.is_valid()) {
            nb::object result = nb::inst_alloc_zero(layout.type);

            cb(result, nb::cpp_function([&](uint64_t) {
                   Layout &layout = this->layout[layout_index++];
                   return construct_dr_var_index(layout);
               }));

            return result;

        } else {
            return layout.py_object;
        }
    }
};

/**
 * Recursively assigns the drjit variables in the PyTree `src` to the object
 * `dst`. This is used to update the input of frozen functions, if their indices
 * should have changed.
 * It allso asigns new key-value pairs to dictionaries if these have been added
 * during the recording of a frozen function.
 */
void assign(nb::handle dst, nb::handle src) {
    nb::handle src_tp = src.type();
    nb::handle dtp = dst.type();
    raise_if(!src_tp.equal(dtp), "Type missmatch!");

    if (is_drjit_type(src_tp)) {
        const ArraySupplement &s = supp(src_tp);
        if (s.is_tensor) {
            s.reset_index(s.index(inst_ptr(src)), inst_ptr(dst));
        } else if (s.ndim > 1) {
            Py_ssize_t len = s.shape[0];
            if (len == DRJIT_DYNAMIC)
                len = s.len(inst_ptr(src));

            for (Py_ssize_t i = 0; i < len; ++i) {
                assign(dst[i], src[i]);
            }
        } else {
            s.reset_index(s.index(inst_ptr(src)), inst_ptr(dst));
        }
    } else if (src_tp.is(&PyTuple_Type)) {
        nb::tuple src_tuple = nb::borrow<nb::tuple>(src);
        nb::tuple dst_tuple = nb::borrow<nb::tuple>(dst);
        for (uint32_t i = 0; i < src_tuple.size(); ++i) {
            assign(dst_tuple[i], src_tuple[i]);
        }
    } else if (src_tp.is(&PyList_Type)) {
        nb::list src_list = nb::borrow<nb::list>(src);
        nb::list dst_list = nb::borrow<nb::list>(dst);
        for (uint32_t i = 0; i < src_list.size(); ++i) {
            assign(dst_list[i], src_list[i]);
        }
    } else if (src_tp.is(&PyDict_Type)) {
        nb::dict src_dict = nb::borrow<nb::dict>(src);
        nb::dict dst_dict = nb::borrow<nb::dict>(dst);
        for (auto k : src_dict.keys()) {
            if (dst_dict.contains(k)) {
                assign(dst_dict[k], src_dict[k]);
            } else {
                dst_dict[k] = src_dict[k];
            }
        }
    } else {
        if (nb::dict ds = get_drjit_struct(src_tp); ds.is_valid()) {
            for (auto [k, v] : ds) {
                assign(nb::getattr(dst, k), nb::getattr(src, k));
            }
        } else if (nb::object df = get_dataclass_fields(src_tp)) {
            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                assign(nb::getattr(dst, k), nb::getattr(src, k));
            }
        } else if (nb::object src_cb = get_traverse_cb_ro(src_tp),
                   dst_cb = get_traverse_cb_rw(dtp);
                   src_cb.is_valid() && dst_cb.is_valid()) {
            std::vector<uint64_t> tmp;
            src_cb(src, nb::cpp_function(
                            [&](uint64_t index) { tmp.push_back(index); }));
            size_t i = 0;
            dst_cb(dst, nb::cpp_function([&](uint64_t) { return tmp[i++]; }));
        } else {
        }
    }
}

struct TransformInPlaceCallback {
    // The transform operation, applied to each index.
    // Should return an owning reference.
    virtual uint64_t operator()(uint64_t index) {
        return index;
    };
};

static void transform_in_place(nb::handle h, TransformInPlaceCallback &op);

static void transform_in_place_dr_flat(nb::handle h,
                                       TransformInPlaceCallback &op) {
    nb::handle tp = h.type();

    const ArraySupplement &s = supp(tp);

    uint64_t index = s.index(inst_ptr(h));
    uint64_t new_index = op(index);
    s.reset_index(new_index, inst_ptr(h));
    ad_var_dec_ref(new_index);
}

static void transform_in_place_dr_class(nb::handle h,
                                        TransformInPlaceCallback &op) {
    nb::handle tp = h.type();
    const ArraySupplement &s = supp(tp);

    transform_in_place_dr_flat(h, op);

    JitBackend backend = (JitBackend)s.backend;
    nb::str domain = nb::borrow<nb::str>(nb::getattr(tp, "Domain"));

    uint32_t id_bound = jit_registry_id_bound(backend, domain.c_str());

    jit_log(LogLevel::Debug, "transforming %u subtypes of domain %s", id_bound,
            domain.c_str());
    for (uint32_t id = 0; id < id_bound; ++id) {
        void *ptr = jit_registry_ptr(backend, domain.c_str(), id + 1);

        // WARN: very unsafe cast!
        nb::intrusive_base *base = (nb::intrusive_base *)ptr;
        nb::handle inst_obj = base->self_py();

        transform_in_place(inst_obj, op);
    }
}
static void transform_in_place_dr(nb::handle h, TransformInPlaceCallback &op) {
    nb::handle tp = h.type();

    const ArraySupplement &s = supp(tp);

    if (s.is_class)
        transform_in_place_dr_class(h, op);
    else
        transform_in_place_dr_flat(h, op);
}

static void transform_in_place(nb::handle h, TransformInPlaceCallback &op) {
    nb::handle tp = h.type();

    nb::print(tp);

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if (s.is_tensor) {
            transform_in_place_dr(h, op);
        } else if (s.ndim > 1) {
            Py_ssize_t len = s.shape[0];
            if (len == DRJIT_DYNAMIC)
                len = s.len(inst_ptr(h));

            for (Py_ssize_t i = 0; i < len; ++i) {
                transform_in_place(h[i], op);
            }
        } else {
            transform_in_place_dr(h, op);
        }
    } else if (tp.is(&PyTuple_Type)) {
        nb::tuple tuple = nb::borrow<nb::tuple>(h);
        for (uint32_t i = 0; i < tuple.size(); ++i) {
            transform_in_place(tuple[i], op);
        }
    } else if (tp.is(&PyList_Type)) {
        nb::list list = nb::borrow<nb::list>(h);
        for (uint32_t i = 0; i < list.size(); ++i) {
            transform_in_place(list[i], op);
        }
    } else if (tp.is(&PyDict_Type)) {
        nb::dict dict = nb::borrow<nb::dict>(h);
        for (auto v : dict.values()) {
            transform_in_place(v, op);
        }
    } else {
        if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {
            for (auto k: ds.keys()) {
                transform_in_place(nb::getattr(h, k), op);
            }
        } else if (nb::object df = get_dataclass_fields(tp)) {
            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                transform_in_place(nb::getattr(h, k), op);
            }
        } else if (nb::object cb = get_traverse_cb_rw(tp); cb.is_valid()) {
            // We want to transfer ownership, so we have to drop references
            // afterwards.
            // This is accomplished by storing them.
            std::vector<uint64_t> tmp;
            cb(h, nb::cpp_function([&](uint64_t index) {
                   uint64_t new_index = op(index);
                   tmp.push_back(new_index);
                   return new_index;
               }));
            for (uint64_t index : tmp) {
                ad_var_dec_ref(index);
            }
        } else {
        }
    }
}

static void make_opaque(nb::handle h) {
    jit_log(LogLevel::Debug, "make_opaque");

    struct ScheduleForceCallback : TransformInPlaceCallback {
        bool result = false;

        uint64_t operator()(uint64_t index) override {
            int rv = 0;
            uint64_t new_index = ad_var_schedule_force(index, &rv);
            jit_log(LogLevel::Debug, "    scheduled %zu -> %zu", index,
                    new_index);
            if (rv)
                result = true;

            return new_index;
        }
    };

    ScheduleForceCallback op;
    transform_in_place(h, op);

    if (op.result) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
}

struct FunctionRecording {
    Recording *recording = nullptr;
    FlatVariables out_variables;

    FunctionRecording() : out_variables(false) {
    }
    FunctionRecording(const FunctionRecording &) = delete;
    FunctionRecording &operator=(const FunctionRecording &) = delete;
    FunctionRecording(FunctionRecording &&) = default;
    FunctionRecording &operator=(FunctionRecording &&) = default;

    ~FunctionRecording() {
        if (this->recording) {
            jit_record_destroy(this->recording);
        }
        this->recording = nullptr;
    }

    /*
     * Record a function, given it's python input and flattened input.
     */
    nb::object record(nb::callable func, nb::list input,
                      const FlatVariables &in_variables) {
        JitBackend backend = in_variables.backend;

        jit_log(LogLevel::Info,
                "Recording (n_inputs=%u):", in_variables.variables.size());
        jit_record_start(backend, in_variables.variables.data(),
                         in_variables.variables.size());

        // Record the function
        bool tmp = jit_flag(JitFlag::KernelFreezing);
        jit_set_flag(JitFlag::KernelFreezing, false);
        auto result = func(*input[0], **input[1]);
        jit_set_flag(JitFlag::KernelFreezing, tmp);

        nb::list output;
        output.append(result);
        output.append(input);

        // NOTE: might want to make result opaque as well?
        make_opaque(input);
        eval(output);

        jit_log(LogLevel::Info, "Traversing output");

        // Pause recording before traversal as to not accedentally record
        // unwanted operations.
        jit_record_pause(backend);

        out_variables.traverse(output);

        if ((out_variables.variables.size() > 0 &&
             in_variables.variables.size() > 0) &&
            out_variables.backend != backend) {
            Recording *recording = jit_record_stop(backend, nullptr, 0);
            jit_record_destroy(recording);

            nb::raise("freeze(): backend missmatch error (backend %u of "
                      "output "
                      "variables did not match backend %u of input "
                      "variables)",
                      (uint32_t)out_variables.backend, (uint32_t)backend);
        }

        recording = jit_record_stop(backend, out_variables.variables.data(),
                                    out_variables.variables.size());

        jit_log(LogLevel::Info, "Recording done (n_outputs=%u)",
                out_variables.variables.size());

        return output[0];
    }
    /*
     * Replays the recording.
     *
     * This constructs the output and re-assigns the input.
     */
    nb::object replay(const FlatVariables &in_variables, nb::list input) {

        jit_log(LogLevel::Info, "Replaying:");
        jit_record_replay(recording, in_variables.variables.data(),
                          out_variables.variables.data());
        jit_log(LogLevel::Info, "Replaying done:");

        out_variables.layout_index = 0;
        auto output = nb::borrow<nb::list>(out_variables.construct());
        auto result = output[0];
        auto new_input = output[1];

        assign(input, new_input);

        // out_variables is assigned by jit_record_replay, which transfers
        // ownership to this array. Therefore, we have to drop the variables
        // afterwards.
        out_variables.drop_variables();

        return result;
    }
};

inline size_t py_object_hash(nb::handle h) {
    Py_hash_t hash = PyObject_Hash(h.ptr());
    if (hash == -1 && PyErr_Occurred())
        nb::raise_python_error();
    return (ssize_t)hash;
}

inline void hash_combine(size_t &seed, size_t value) {
    /// From CityHash (https://github.com/google/cityhash)
    const size_t mult = 0x9ddfea08eb382d69ull;
    size_t a = (value ^ seed) * mult;
    a ^= (a >> 47);
    size_t b = (seed ^ a) * mult;
    b ^= (b >> 47);
    seed = b * mult;
}

struct RecordingKey {
    std::vector<Layout> layout;
    uint32_t flags;

    RecordingKey() {
    }
    RecordingKey(std::vector<Layout> layout, uint32_t flags)
        : layout(std::move(layout)), flags(flags) {
    }

    bool operator==(const RecordingKey &rhs) const {
        return this->layout == rhs.layout && this->flags == rhs.flags;
    }

    void log_diff(const RecordingKey *rhs) const {
        jit_log(LogLevel::Debug, "Key difference:");
        if (this->flags != rhs->flags)
            jit_log(LogLevel::Debug, "    flags: %u != %u", this->flags,
                    rhs->flags);

        if (this->layout.size() != rhs->layout.size()) {
            jit_log(LogLevel::Debug, "    n_layout: %u != %u",
                    this->layout.size(), rhs->layout.size());
            return;
        }

        for (uint32_t i = 0; i < this->layout.size(); ++i) {
            const Layout &lhs_layout = this->layout[i];
            const Layout &rhs_layout = rhs->layout[i];

            // if (lhs_layout == rhs_layout)
            //     continue;

            jit_log(LogLevel::Debug, "    layout %u:", i);
            if (!lhs_layout.type.is_none() && !rhs_layout.type.is_none() &&
                !lhs_layout.type.equal(rhs_layout.type))
                jit_log(LogLevel::Debug, "    type: %s != %s",
                        nb::type_name(lhs_layout.type).c_str(),
                        nb::type_name(rhs_layout.type).c_str());
            if (lhs_layout.num != rhs_layout.num)
                jit_log(LogLevel::Debug, "    num: %u != %u", lhs_layout.num,
                        rhs_layout.num);
            if (lhs_layout.vt != rhs_layout.vt)
                jit_log(LogLevel::Debug, "    vt: %u != %u", lhs_layout.vt,
                        rhs_layout.vt);
            if (lhs_layout.vs != rhs_layout.vs)
                jit_log(LogLevel::Debug, "    vs: %u != %u", lhs_layout.vs,
                        rhs_layout.vs);
            if (lhs_layout.singleton_array != rhs_layout.singleton_array)
                jit_log(LogLevel::Debug, "    singleton_array: %u != %u",
                        lhs_layout.singleton_array, rhs_layout.singleton_array);
            if (lhs_layout.unaligned != rhs_layout.unaligned)
                jit_log(LogLevel::Debug, "    unaligned: %u != %u",
                        lhs_layout.unaligned, rhs_layout.unaligned);
            if (lhs_layout.literal != rhs_layout.literal)
                jit_log(LogLevel::Debug, "    literal: %u != %u",
                        lhs_layout.literal, rhs_layout.literal);
            if (lhs_layout.index != rhs_layout.index)
                jit_log(LogLevel::Debug, "    index: %u != %u",
                        lhs_layout.index, rhs_layout.index);
            if (!(lhs_layout.py_object.equal(rhs_layout.py_object)))
                jit_log(LogLevel::Debug, "    py_object: %s != %s",
                        nb::str(lhs_layout.py_object).c_str(),
                        nb::str(rhs_layout.py_object).c_str());
        }
    }
    void log() {
        jit_log(LogLevel::Debug, "RecordingKey{");
        jit_log(LogLevel::Debug, "    flags = %u", this->flags);

        jit_log(LogLevel::Debug, "    layout = [");
        for (Layout &layout : layout) {
            jit_log(LogLevel::Debug, "        Layout{");
            if (!layout.type.is_none())
                jit_log(LogLevel::Debug, "            type = %s,",
                        nb::type_name(layout.type).c_str());
            jit_log(LogLevel::Debug, "            num = %u,", layout.num);
            jit_log(LogLevel::Debug, "            vt = %u,",
                    (uint32_t)layout.vt);
            jit_log(LogLevel::Debug, "            vs = %u,",
                    (uint32_t)layout.vs);
            jit_log(LogLevel::Debug, "            singleton_array = %u,",
                    layout.singleton_array);
            jit_log(LogLevel::Debug, "        },");
        }
        jit_log(LogLevel::Debug, "    ]");

        jit_log(LogLevel::Debug, "}");
    }
};

struct RecordingKeyHasher {
    size_t operator()(const RecordingKey &key) const {
        // Hash the layout
        size_t hash = key.layout.size();
        for (const Layout &layout : key.layout) {
            hash_combine(hash, py_object_hash(layout.type));
            hash_combine(hash, layout.num);
            hash_combine(hash, layout.fields.size());
            for (auto &field : layout.fields) {
                hash_combine(hash, py_object_hash(field));
            }
            hash_combine(hash, (size_t)layout.vt);
            hash_combine(hash, (size_t)layout.vs);
            hash_combine(hash, (size_t)layout.singleton_array);
            hash_combine(hash, (size_t)layout.unaligned);
            hash_combine(hash, (size_t)layout.literal);
            hash_combine(hash, (size_t)layout.index);
            hash_combine(hash, py_object_hash(layout.py_object));
        }

        hash_combine(hash, (size_t)key.flags);

        return hash;
    }
};

using RecordingMap =
    tsl::robin_map<RecordingKey, std::unique_ptr<FunctionRecording>,
                   RecordingKeyHasher>;

struct FrozenFunction {
    nb::callable func;

    RecordingMap recordings;
    RecordingKey prev_key;

    FrozenFunction(nb::callable func) : func(func) {
    }
    ~FrozenFunction() {
    }

    FrozenFunction(const FrozenFunction &) = delete;
    FrozenFunction &operator=(const FrozenFunction &) = delete;
    FrozenFunction(FrozenFunction &&) = default;
    FrozenFunction &operator=(FrozenFunction &&) = default;

    uint32_t n_recordings() {
        return this->recordings.size();
    }

    nb::object operator()(nb::args args, nb::kwargs kwargs) {

        if (!jit_flag(JitFlag::KernelFreezing)) {
            return func(*args, **kwargs);
        }

        nb::list input;
        input.append(args);
        input.append(kwargs);

        // Evaluate input variables, forcing evaluation of undefined variables
        // NOTE: not sure, why both are necessary
        make_opaque(input);
        eval(input);

        // Traverse input variables
        jit_log(LogLevel::Debug, "freeze(): Traversing input.");
        FlatVariables in_variables(true);
        in_variables.traverse(input);

        raise_if(in_variables.backend == JitBackend::None,
                 "freeze(): Cannot infer backend without providing input "
                 "variable to frozen function!");

        uint32_t flags = jit_flags();
        auto key = RecordingKey(std::move(in_variables.layout), flags);
        auto it = this->recordings.find(key);

        if (it == this->recordings.end()) {
            if (this->recordings.size() >= 1) {
                jit_log(LogLevel::Info,
                        "Function input missmatch! Function will be retraced.");
                key.log_diff(&prev_key);
            }
            // FunctionRecording recording;
            auto recording = std::make_unique<FunctionRecording>();

            nb::object result;
            try {
                result = recording->record(func, input, in_variables);
            } catch (const std::exception &e) {
                in_variables.drop_variables();
                jit_record_abort(in_variables.backend);

                nb::chain_error(PyExc_RuntimeError, "%s", e.what());
                nb::raise_python_error();
            };

            in_variables.drop_variables();

            this->prev_key = key;
            this->recordings.insert({std::move(key), std::move(recording)});

            return result;
        } else {
            // Drop references to variables
            in_variables.drop_variables();

            FunctionRecording *recording = it.value().get();

            auto result = recording->replay(in_variables, input);

            return result;
        }
    }
};

FrozenFunction freeze(nb::callable func) {
    return FrozenFunction(func);
}

void export_freeze(nb::module_ &m) {
    m.def("freeze", &freeze, doc_freeze);
    nb::class_<FrozenFunction>(m, "FrozenFunction")
        .def("__get__",
             [](nb::object self, nb::object instance, nb::object) {
                 if (instance.is_none()) {
                     return self;
                 } else {
                     return nb::cpp_function(
                         [self, instance](nb::args args, nb::kwargs kwargs) {
                             return self(instance, *args, **kwargs);
                         },
                         nb::rv_policy::move);
                 }
             })
        .def_prop_ro("n_recordings",
                     [](FrozenFunction &self) { return self.n_recordings(); })
        .def("__call__", &FrozenFunction::operator());
}
