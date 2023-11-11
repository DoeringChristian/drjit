#include "while.h"
#include "eval.h"
#include "base.h"
#include "reduce.h"
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <functional>
#include <string>

/**
 * \brief This data structure is responsible for capturing and updating the
 * state variables of a dr.while_loop() call and ensuring that they stay
 * consistent over time.
 * */
struct LoopState {
    /// State tuple
    nb::tuple state;
    /// Loop condition
    nb::callable cond;
    /// Function to evolve the loop state
    nb::callable body;
    /// Variable labels to provide nicer error messages
    std::vector<std::string> state_labels;

    /// Holds a temporary reference to the loop condition
    nb::object active;

    struct Entry {
        std::string name;
        nb::object value;
        Entry(const std::string &name, nb::handle value)
            : name(name), value(nb::borrow(value)) {}
    };

    // Post-processed version of 'state'
    std::vector<Entry> entries;
    /// Temporary stack to avoid infinite recursion
    std::vector<PyObject*> stack;
    /// Temporary to assemble a per-variable name
    std::string name;
    /// This variable is 'true' when traverse() is called for the first time
    bool first_time = true;
    /// Index into the 'entries' array when traverse() is called in later iterations
    size_t entry_pos = 0;
    /// Index into the 'indices' array when traverse() is called with <Write>
    size_t indices_pos = 0;

    LoopState(nb::tuple &&state, nb::callable &&cond, nb::callable &&body,
              std::vector<std::string> &&state_labels)
        : state(std::move(state)), cond(std::move(cond)), body(std::move(body)),
          state_labels(std::move(state_labels)), first_time(true) { }

    /// Read or write the set of loop state variables
    template <bool Write, typename OutArray> void traverse(OutArray &indices) {
        size_t l1 = nb::len(state), l2 = state_labels.size();

        if (l2 && l1 != l2)
            nb::raise("the 'state' and 'state_labels' arguments have an inconsistent size.");

        if constexpr (Write)
            indices_pos = 0;

        entry_pos = 0;
        for (size_t i = 0; i < l1; ++i) {
            name = l2 ? state_labels[i] : ("arg" + std::to_string(i));
            traverse<Write>(state[i], indices);
        }

        if constexpr (Write) {
            if (indices_pos != indices.size())
                nb::raise("traverse(): internal error, did not consume all indices.");
        }


        first_time = false;
    }

private:
    template <bool Write, typename OutArray> void traverse(nb::handle h, OutArray &indices) {
        // Avoid infinite recursion
        if (std::find(stack.begin(), stack.end(), h.ptr()) != stack.end())
            return;
        stack.push_back(h.ptr());

        nb::object prev;
        if (first_time) {
            entries.emplace_back(name, h);
        } else {
            if (entry_pos >= entries.size())
                nb::raise("the number of loop state variables must stay "
                          "constant across iterations. However, Dr.Jit "
                          "detected a previously unobserved variable '%s' of "
                          "type '%s', which is not permitted. Please review "
                          "the interface and assumptions of dr.while_loop() as "
                          "explained in the Dr.Jit documentation.",
                          name.c_str(), nb::inst_name(h).c_str());

            Entry &e = entries[entry_pos++];

            if (name != e.name)
                nb::raise(
                    "loop state variable '%s' of type '%s' created in a "
                    "previous iteration cannot be found anymore. "
                    "Instead, another variable '%s' of type '%s' was "
                    "found in its place, which is not permitted. Please"
                    "review the interface and assumptions of dr.while_loop()"
                    "as explained in the Dr.Jit documentation.", e.name.c_str(),
                    nb::inst_name(e.value).c_str(), name.c_str(),
                    nb::inst_name(h).c_str());

            if (!h.type().is(e.value.type()))
                nb::raise(
                    "the body of this loop changed the type of loop state "
                    "variable '%s' from '%s' to '%s', which is not "
                    "permitted. Please review the interface and assumptions "
                    "of dr.while_loop() as explained in the Dr.Jit "
                    "documentation.",
                    name.c_str(), nb::inst_name(e.value).c_str(),
                    nb::inst_name(h).c_str());

            prev = std::move(e.value);
            e.value = nb::borrow(h);
        }

        size_t name_size = name.size();
        nb::handle tp = h.type();
        if (is_drjit_type(tp)) {
            const ArraySupplement &s = supp(tp);
            if (s.is_tensor) {
                name += ".array";
                traverse<Write>(nb::steal(s.tensor_array(h.ptr())), indices);
                name.resize(name_size);
            } else if (s.ndim > 1) {
                Py_ssize_t len = s.shape[0];
                if (len == DRJIT_DYNAMIC)
                    len = s.len(inst_ptr(h));

                for (Py_ssize_t i = 0; i < len; ++i) {
                    name += "[" + std::to_string(i) + "]";
                    traverse<Write>(nb::steal(s.item(h.ptr(), i)), indices);
                    name.resize(name_size);
                }
            } else if (s.index) {
                uint64_t i1 = s.index(inst_ptr(h));

                if (!first_time) {
                    uint64_t i2 = s.index(inst_ptr(prev));

                    size_t s1 = jit_var_size((uint32_t) i1),
                           s2 = jit_var_size((uint32_t) i2);

                    if (s1 != s2 && s1 != 1 && s2 != 1)
                        nb::raise("the body of this loop changed the size of loop "
                                  "state variable '%s' (which is of type '%s') from "
                                  "%zu to %zu. These sizes aren't compatible, and such "
                                  "a change is therefore not permitted. Please review "
                                  "the interface and assumptions of dr.while_loop() as "
                                  "explained in the Dr.Jit documentation.",
                                  name.c_str(), nb::inst_name(h).c_str(), s2, s1);
                }

                if constexpr (Write) {
                    if (indices_pos >= indices.size())
                        nb::raise("traverse(): internal error, ran out of indices.");

                    nb::handle tp = h.type();
                    nb::object tmp = nb::inst_alloc(tp);
                    supp(tp).init_index(indices[indices_pos++], inst_ptr(tmp));
                    nb::inst_mark_ready(tmp);
                    nb::inst_replace_move(h, tmp);
                } else {
                    ad_var_inc_ref(i1);
                    indices.push_back(i1);
                }
            }
        } else if (tp.is(&PyList_Type)) {
            size_t ctr = 0;
            for (nb::handle v: nb::borrow<nb::list>(h)) {
                name += "[" + std::to_string(ctr++) + "]";
                traverse<Write>(v, indices);
                name.resize(name_size);
            }
        } else if (tp.is(&PyTuple_Type)) {
            size_t ctr = 0;
            for (nb::handle v: nb::borrow<nb::tuple>(h)) {
                name += "[" + std::to_string(ctr++) + "]";
                traverse<Write>(v, indices);
                name.resize(name_size);
            }
        } else if (tp.is(&PyDict_Type)) {
            for (nb::handle kv: nb::borrow<nb::dict>(h).items()) {
                nb::handle k = kv[0], v = kv[1];
                if (!nb::isinstance<nb::str>(k))
                    continue;
                if (stack.size() == 1)
                    name = nb::borrow<nb::str>(k).c_str();
                else
                    name += "['" + std::string(nb::borrow<nb::str>(k).c_str()) + "']";
                traverse<Write>(v, indices);
                name.resize(name_size);
            }
        } else {
            nb::object dstruct = nb::getattr(tp, "DRJIT_STRUCT", nb::handle());
            if (dstruct.is_valid() && dstruct.type().is(&PyDict_Type)) {
                for (auto [k, v] : nb::borrow<nb::dict>(dstruct)) {
                    name += "."; name += nb::str(k).c_str();
                    traverse<Write>(nb::getattr(h, k), indices);
                    name.resize(name_size);
                }
            }
        }
        stack.pop_back();
    }
};

/// Helper fucntion to perform a tuple-based function call directly using the
/// CPython API. nanobind lacks a nice abstraction for this.
static nb::object tuple_call(nb::handle callable, nb::handle tuple) {
    nb::object result = nb::steal(PyObject_CallObject(callable.ptr(), tuple.ptr()));
    if (!result.is_valid())
        nb::raise_python_error();
    return result;
}

/// Helper function to check that the type+size of the state variable returned
/// by 'body()' is sensible
static nb::tuple check_state(const char *name, nb::object &&o, const nb::tuple &old_state) {
    if (!o.type().is(&PyTuple_Type))
        nb::raise("the '%s' function must return a tuple.", name);
    nb::tuple o_t = nb::borrow<nb::tuple>(o);
    if (nb::len(o_t) != nb::len(old_state))
        nb::raise("the '%s' function returned a tuple with an inconsistent size.", name);
    return o_t;
}

/// Helper fucntion to check that the return value of the loop conditional is sensible
static const ArraySupplement &check_cond(nb::handle h) {
    nb::handle tp = h.type();
    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if ((VarType) s.type == VarType::Bool && s.ndim == 1)
            return s;
    }

    nb::raise("the type of the loop condition ('%s') is not supported. "
              "You must either provide a 1D Dr.Jit boolean array or a "
              "Python 'bool' value.", nb::type_name(tp).c_str());
}

/// Callback functions that will be invoked by ad_loop()
static uint32_t while_loop_cond_cb(void *p) {
    LoopState *lp = (LoopState *) p;
    lp->active = tuple_call(lp->cond, lp->state);
    return (uint32_t) check_cond(lp->active).index(inst_ptr(lp->active));
}

static void while_loop_body_cb(void *p) {
    LoopState *lp = (LoopState *) p;
    lp->state = check_state("body", tuple_call(lp->body, lp->state), lp->state);
};

static void while_loop_read_cb(void *p, dr::dr_vector<uint64_t> &indices) {
    ((LoopState *) p)->traverse<false>(indices);
}

static void while_loop_write_cb(void *p,
                                const dr::dr_vector<uint64_t> &indices) {
    ((LoopState *) p)->traverse<true>(indices);
}

nb::tuple while_loop(nb::tuple state, nb::callable cond, nb::callable body,
                     std::vector<std::string> &&state_labels,
                     const std::string &name,
                     const std::string &method) {
    try {
        JitBackend backend = JitBackend::None;

        bool scalar_loop = method == "scalar",
             auto_loop = method == "auto";

        nb::object cond_val = tuple_call(cond, state);
        if (auto_loop)
            scalar_loop = cond_val.type().is(&PyBool_Type);

        if (scalar_loop) {
            // If so, process it directly
            while (nb::cast<bool>(cond_val)) {
                state = check_state("body", tuple_call(body, state), state);
                cond_val = tuple_call(cond, state);
            }

            return state;
        }

        backend = (JitBackend) check_cond(cond_val).backend;
        cond_val.reset();

        // General case: call ad_loop() with a number of callbacks
        // that implement an interface to Python
        dr::dr_unique_ptr<LoopState> payload(
            new LoopState(std::move(state), std::move(cond), std::move(body),
                          std::move(state_labels)));

        int symbolic = -1;
        if (auto_loop)
            symbolic = -1;
        else if (method == "symbolic")
            symbolic = 1;
        else if (method == "evaluated")
            symbolic = 0;
        else
            nb::raise("invalid 'method' parameter (must be one of \"auto\", "
                      "\"scalar\", \"symbolic\", or \"evaluated\").");

        ad_loop(backend, symbolic, name.c_str(), payload.get(),
                while_loop_read_cb, while_loop_write_cb, while_loop_cond_cb,
                while_loop_body_cb);

        return nb::borrow<nb::tuple>(payload->state);
    } catch (nb::python_error &e) {
        nb::raise_from(
            e, PyExc_RuntimeError,
            "dr.while_loop(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "dr.while_loop(): %s", e.what());
        throw nb::python_error();
    }
}

#if 0
nb::tuple if_stmt(nb::tuple state, nb::callable cond,
                  nb::callable true_fn,
                  nb::callable false_fn,
                  const std::vector<std::string> &,
                  const std::string &name,
                  const std::string &method) {
    try {
        // First, check if this is perhaps a scalar loop
        nb::object cond_val = tuple_call(cond, state);
        if (cond_val.type().is(&PyBool_Type)) {
            if (nb::cast<bool>(cond_val))
                return check_state("true_fn", tuple_call(true_fn, state), state);
            else
                return check_state("false_fn", tuple_call(false_fn, state), state);
        }
        nb::raise("Vectorial if statements arent' supported yet.");
    } catch (nb::python_error &e) {
        nb::raise_from(
            e, PyExc_RuntimeError,
            "dr.if_stmt(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "dr.if_stmt(): %s", e.what());
        throw nb::python_error();
    }
}
#endif

void export_while(nb::module_ &m) {
    m.def("while_loop", &while_loop, "state"_a, "cond"_a, "body"_a,
          "state_labels"_a = nb::make_tuple(), "label"_a = "unnamed",
          "method"_a = "auto", doc_while_loop);
    //
    // m.def("if_stmt", &if_stmt, "state"_a, "cond"_a, "true_fn"_a, "false_fn"_a,
    //       "state_labels"_a = nb::make_tuple(), "label"_a = "unnamed", method_a="auto");
}
