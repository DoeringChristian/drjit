
#include "freeze.h"
#include "base.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "drjit/extra.h"
#include "eval.h"
#include "listobject.h"
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
    VarState vs = VarState::Undefined;
    /// Weather the variable is an array with a single entry.
    /// Such arrays are handled differently by the compiler.
    bool singleton_array = false;
    /// The literal data
    void *literal = nullptr;
    /// The index in the flat_variables array of this variable.
    /// This can be used to determine aliasing.
    uint32_t index = 0;

    bool operator==(const Layout &rhs) const {
        if (!(this->type.equal(rhs.type)))
            return false;
        if (this->num != rhs.num)
            return false;
        if (this->fields.size() != rhs.fields.size())
            return false;
        for (uint32_t i = 0; i < this->fields.size(); ++i) {
            if (!(this->fields[i].equal(rhs.fields[i])))
                return false;
        }
        if (this->vt != rhs.vt)
            return false;
        if (this->vs != rhs.vs)
            return false;
        if (this->singleton_array != rhs.singleton_array)
            return false;
        if (this->index != rhs.index)
            return false;
        if (this->literal != rhs.literal)
            return false;
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

    // Wether variables should be copied when their reference count is > 1
    bool copy_on_write = true;

    FlatVariables() {
    }
    FlatVariables(bool copy_on_write) : copy_on_write(copy_on_write) {
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
            jit_log(LogLevel::Info,
                    "collect(): Adding new variable var(%u) -> flat_var(%u)",
                    variable_index, flat_index);

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

    void collect(nb::handle h) {
        auto s = supp(h.type());

        JitBackend var_backend = (JitBackend)s.backend;

        if (this->backend == var_backend || this->backend == JitBackend::None) {
            this->backend = var_backend;
        } else {
            nb::raise("freeze(): backend missmatch error (backend of this "
                      "variable %u does not match backend of others %u)!",
                      (uint32_t)var_backend, (uint32_t)this->backend);
        }
        raise_if(s.index == nullptr, "ArraySupplement index function "
                                     "pointer is nullptr.");

        uint64_t index = s.index(inst_ptr(h));

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

        jit_log(LogLevel::Info, "collect(): collecting var(%u)", index);
        uint32_t rc = jit_var_ref(index);
        jit_log(LogLevel::Info, "\trc=%u", rc);
        if (copy_on_write && rc > 1) {
            index = jit_var_copy(index);
            s.reset_index(index, inst_ptr(h));
            jit_var_dec_ref(index);
        }

        Layout layout;
        VarState vs = jit_var_state(index);
        layout.type = nb::borrow<nb::type_object>(h.type());
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
            jit_log(LogLevel::Info, "    vs=%u", (uint32_t)vs);
            layout.index = this->add_variable(index);
            layout.singleton_array = jit_var_size(index) == 1;
        } else {
            nb::raise("collect(): found variable %zu in unsupported state %u!",
                      index, (uint32_t)vs);
        }
        this->layout.push_back(layout);
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
                    collect(h);
                } else if (s.ndim > 1) {
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
                    collect(h);
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
                // TODO: traverse callback
            } else if (tp.is(&_PyNone_Type)) {
                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                this->layout.push_back(layout);
            } else {
                nb::raise("Encountered unsupported type! Only drjit types or "
                          "composite types containing drjit variables are "
                          "supported.");
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

    nb::object construct_dr_var(Layout &layout) {
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
            } else if (s.ndim > 1) {
                nb::raise("FlatVariables::construct(): dynamic sized Dr.Jit "
                          "variables are not yet supported.");

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
        } else {
            if (nb::dict ds = get_drjit_struct(layout.type); ds.is_valid()) {
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
            }
        }
        nb::raise("FlatVariables::construct(): could not reconstruct "
                  "output variable of type %s",
                  nb::type_name(layout.type).c_str());
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
    raise_if(!src_tp.equal(dtp), "");

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
        } else {
        }
    }
}

struct FrozenFunction {
    Recording *recording = nullptr;
    FlatVariables out_variables;
    std::vector<Layout> in_layout;
    nb::callable func;

    FrozenFunction(nb::callable func) : out_variables(false), func(func) {
    }
    ~FrozenFunction() {
        if (this->recording) {
            jit_record_destroy(this->recording);
        }
    }

    nb::object operator()(nb::args args, nb::kwargs kwargs) {

        nb::list input;
        input.append(args);
        input.append(kwargs);

        // Collect input variables and evaluate them
        eval(input);
        FlatVariables in_variables(true);
        in_variables.traverse(input);

        raise_if(in_variables.variables.size() == 0,
                 "freeze(): Cannot infer backend without providing input "
                 "variable to frozen functin!");

        JitBackend backend = in_variables.backend;

        if (recording == nullptr) {
            jit_log(LogLevel::Info,
                    "Recording (n_inputs=%u):", in_variables.variables.size());
            jit_record_start(backend, in_variables.variables.data(),
                             in_variables.variables.size());

            // Record the function
            auto result = func(*args, **kwargs);

            nb::list output;
            output.append(result);
            output.append(args);

            eval(output);

            // Pause recording before traversal as to not accedentally record
            // unwanted operations.
            jit_record_pause(backend);

            out_variables.traverse(output);

            raise_if((out_variables.variables.size() > 0 &&
                      in_variables.variables.size() > 0) &&
                         out_variables.backend != backend,
                     "freeze(): backend missmatch error (backend %u of "
                     "output "
                     "variables did not match backend %u of input "
                     "variables)",
                     (uint32_t)out_variables.backend, (uint32_t)backend);

            recording = jit_record_stop(backend, out_variables.variables.data(),
                                        out_variables.variables.size());

            jit_log(LogLevel::Info, "Recording done (n_outputs=%u)",
                    out_variables.variables.size());

            this->in_layout = std::move(in_variables.layout);
            return result;
        } else {
            // TODO: report missmatch
            raise_if(this->in_layout != in_variables.layout,
                     "freeze(): Layout mismatch!");
            jit_log(LogLevel::Info, "Replaying:");
            jit_record_replay(recording, in_variables.variables.data(),
                              out_variables.variables.data());
            jit_log(LogLevel::Info, "Replaying done:");

            out_variables.layout_index = 0;
            auto output = nb::borrow<nb::list>(out_variables.construct());
            auto result = output[0];
            auto new_args = output[1];
            assign(args, new_args);

            // out_variables is assigned by jit_record_replay, which transfers
            // ownership to this array. Therefore, we have to dop the variables
            // afterwards.
            out_variables.drop_variables();

            return output[0];
        }
    }
};

FrozenFunction freeze(nb::callable func) {
    FrozenFunction frozen(func);
    return frozen;
}

void export_freeze(nb::module_ &m) {
    m.def("freeze", &freeze, doc_freeze);
    nb::class_<FrozenFunction>(m, "FrozenFunction")
        .def("__call__", &FrozenFunction::operator());
}
