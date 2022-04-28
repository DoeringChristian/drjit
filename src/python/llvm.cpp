/*
    llvm.cpp -- instantiates the drjit.llvm.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"
#include "random.h"

void bind_llvm(nb::module_ &m) {
    dr::bind_all_types<dr::LLVMArray<float>>();
    bind_pcg32<dr::LLVMArray<float>>(m);
}
