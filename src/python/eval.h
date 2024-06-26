/*
    eval.h -- Bindings for drjit.eval() and drjit.schedule()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern bool schedule(nb::handle);
extern bool eval(nb::handle h);
extern void export_eval(nb::module_ &);
