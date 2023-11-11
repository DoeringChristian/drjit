/*
    eval.cpp -- Bindings for drjit.eval() and drjit.schedule()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "eval.h"
#include "apply.h"

bool schedule(nb::handle h) {
    bool result_ = false;

    struct ScheduleCallback : TraverseCallback {
        bool &result;
        ScheduleCallback(bool &result) : result(result) { }

        void operator()(nb::handle h) const override {
            const ArraySupplement &s = supp(h.type());
            if (s.index)
                result |= jit_var_schedule(s.index(inst_ptr(h))) != 0;
        }
    };

    traverse("drjit.schedule", ScheduleCallback{ result_ }, h);
    return result_;
}

static bool schedule_2(nb::args args) { return schedule(args); }

static void make_opaque(nb::handle h) {
    struct ScheduleForceCallback : TraverseCallback {
        bool &result;
        ScheduleForceCallback(bool &result) : result(result) { }

        void operator()(nb::handle h) const override {
            nb::handle tp = h.type();
            const ArraySupplement &s = supp(tp);
            if (!s.index)
                return;

            int rv = 0;
            uint64_t index = s.index(inst_ptr(h));
            uint32_t index_new = ad_var_schedule_force(index, &rv);
            if (rv)
                result = true;

            if (index != index_new) {
                nb::object tmp = nb::inst_alloc(tp);
                s.init_index(index_new, inst_ptr(tmp));
                nb::inst_mark_ready(tmp);
                nb::inst_replace_move(h, tmp);
            }

            ad_var_dec_ref(index_new);
        }
    };

    bool result = false;
    traverse("drjit.make_opaque", ScheduleForceCallback{ result }, h);
    if (result) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
}

static void make_opaque_2(nb::args args) { return make_opaque(args); }

bool eval(nb::handle h) {
    if (schedule(h)) {
        nb::gil_scoped_release guard;
        jit_eval();
        return true;
    }
    return false;
}

static bool eval_2(nb::args args) {
    bool rv = schedule(args);
    if (rv || nb::len(args) == 0) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
    return rv;
}

void export_eval(nb::module_ &m) {
    m.def("schedule", &schedule, doc_schedule)
     .def("schedule", &schedule_2)
     .def("eval", &eval, doc_eval)
     .def("eval", &eval_2)
     .def("make_opaque", &make_opaque, doc_make_opaque)
     .def("make_opaque", &make_opaque_2);
}
