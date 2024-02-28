/*
    if_stmt.cpp -- Python implementation of drjit.if_stmt() based on the
    abstract interface ad_cond() provided by the drjit-extra library

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "if_stmt.h"
#include "pystate.h"
#include "detail.h"
#include "apply.h"
#include "base.h"
#include "shape.h"
#include <nanobind/stl/optional.h>

/// State object passed to callbacks that implement the Python interface around ad_cond().
struct IfState {
    nb::object args;
    nb::callable true_fn, false_fn;
    nb::object rv;
    dr::vector<dr::string> rv_labels;
    CopyMap copy_map;
    dr::vector<StashRef> sr;

    IfState(nb::object &&args, nb::callable &&true_fn, nb::callable &&false_fn,
            dr::vector<dr::string> &&rv_labels)
        : args(std::move(args)), true_fn(std::move(true_fn)),
          false_fn(std::move(false_fn)), rv_labels(std::move(rv_labels)) { }

    void cleanup() {
        copy_map.clear();
        sr.clear();
    }
};

static void if_stmt_body_cb(void *p, bool value,
                            const vector<uint64_t> &args_i,
                            vector<uint64_t> &rv_i) {
    IfState *is = (IfState *) p;
    nb::gil_scoped_acquire guard;

    // Rewrite the inputs with new argument indices given in 'args_i'
    is->args = update_indices(is->args, args_i, &is->copy_map,
                              /* preserve_dirty == */ !value);

    // Run the operation
    nb::object rv = tuple_call(value ? is->true_fn : is->false_fn, is->args);

    // Stash the reference count of any new variables created by side effects
    if (value)
        stash_ref(is->args, is->sr);

    // Ensure that the output of 'true_fn' and 'false_fn' is consistent
    if (is->rv.is_valid()) {
        size_t l1 = is->rv_labels.size(), l2 = (size_t) -1, l3 = (size_t) -1;

        try {
           l2 = nb::len(is->rv);
           l3 = nb::len(rv);
        } catch (...) { }

        try {
            if (l1 == l2 && l2 == l3 && l3 > 0) {
                for (size_t i = 0; i < l1; ++i)
                    check_compatibility(is->rv[i], rv[i],
                                        is->rv_labels[i].c_str());
            } else {
                check_compatibility(is->rv, rv, "result");
            }
        } catch (const std::exception &e) {
            nb::raise("detected an inconsistency when comparing the return "
                      "values of 'true_fn' and 'false_fn':\n%s\n\nPlease review "
                      "the interface and assumptions of dr.if_stmt() as "
                      "explained in the Dr.Jit documentation.", e.what());
        }
    }

    collect_indices(rv, rv_i, true);
    is->rv = std::move(rv);
}

static void if_stmt_delete_cb(void *p) {
    if (!nb::is_alive())
        return;
    nb::gil_scoped_acquire guard;
    delete (IfState *) p;
}

nb::object if_stmt(nb::tuple args, nb::handle cond, nb::callable true_fn,
                   nb::callable false_fn, dr::vector<dr::string> &&rv_labels,
                   std::optional<dr::string> name, std::optional<dr::string> mode) {
    try {
        (void) rv_labels;
        JitBackend backend = JitBackend::None;
        uint32_t cond_index = 0;

        bool is_scalar;
        if (mode.has_value())
            is_scalar = mode == "scalar";
        else
            is_scalar = cond.type().is(&PyBool_Type);

        if (!is_scalar) {
            nb::handle tp = cond.type();

            if (is_drjit_type(tp)) {
                const ArraySupplement &s = supp(tp);
                if ((VarType) s.type == VarType::Bool && s.ndim == 1 &&
                    (JitBackend) s.backend != JitBackend::None) {
                    backend = (JitBackend) s.backend;
                    cond_index = (uint32_t) s.index(inst_ptr(cond));
                    if (!cond_index)
                        nb::raise("'cond' cannot be empty.");
                }
            }

            if (!cond_index)
                nb::raise("'cond' must either be a Jit-compiled 1D Boolean "
                          "array or a Python 'bool'.");
        }

        if (is_scalar) {
            // If so, process it directly
            if (nb::cast<bool>(cond))
                return tuple_call(true_fn, args);
            else
                return tuple_call(false_fn, args);
        }

        // General case: call ad_cond() with a number of callbacks that
        // implement an interface to Python
        int symbolic = -1;
        if (!mode.has_value())
            symbolic = -1;
        else if (mode == "symbolic")
            symbolic = 1;
        else if (mode == "evaluated")
            symbolic = 0;
        else
            nb::raise("invalid 'mode' argument (must equal None, "
                      "\"scalar\", \"symbolic\", or \"evaluated\").");

        const char *name_cstr =
            name.has_value() ? name.value().c_str() : "unnamed";

        IfState *is =
            new IfState(std::move(args), std::move(true_fn),
                        std::move(false_fn), std::move(rv_labels));

        // Temporarily stash the reference counts of inputs. This influences the
        // behavior of copy-on-write (COW) operations like dr.scatter performed
        // within the symbolic region
        stash_ref(is->args, is->sr);

        dr_index_vector args_i, rv_i;
        collect_indices(is->args, args_i, true);

        bool all_done =
            ad_cond(backend, symbolic, name_cstr, is, cond_index, args_i,
                    rv_i, if_stmt_body_cb, if_stmt_delete_cb, true);

        // Construct the final set of return values
        nb::object rv = update_indices(is->rv, rv_i, &is->copy_map);

        // Undo copy operations for unchanged elements of 'args'
        rv = uncopy(rv, is->copy_map);

        if (all_done) {
            delete is;
        } else {
            is->rv.reset();
            is->cleanup();
        }

        return rv;
    } catch (nb::python_error &e) {
        nb::raise_from(
            e, PyExc_RuntimeError,
            "dr.if_stmt(): encountered an exception (see above).");
    } catch (const std::exception &e) {
        nb::chain_error(PyExc_RuntimeError, "dr.if_stmt(): %s", e.what());
        throw nb::python_error();
    }
}

void export_if_stmt(nb::module_ &m) {
    m.def("if_stmt", &if_stmt, "args"_a, "cond"_a, "true_fn"_a, "false_fn"_a,
          "rv_labels"_a = nb::make_tuple(), "label"_a = nb::none(),
          "mode"_a = nb::none(), doc_if_stmt,
          // Complicated signature to type-check if_stmt via TypeVarTuple
          nb::sig(
            "def if_stmt(args: tuple[*_Ts], "
                        "cond: ArrayBase | bool, "
                        "true_fn: Callable[[*_Ts], _T], "
                        "false_fn: Callable[[*_Ts], _T], "
                        "rv_labels: Sequence[str] = (), "
                        "label: str | None = None, "
                        "mode: str | None = None) "
            "-> _T")
    );
}
