/*
    llvm.cpp -- instantiates the drjit.llvm.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "llvm.h"
#include "random.h"
#include "texture.h"

#if defined(DRJIT_ENABLE_LLVM)
void export_llvm(nb::module_ &m) {
    using Guide = dr::LLVMArray<float>;

    ArrayBinding b;
    dr::bind_all<Guide>(b);
    bind_pcg32<Guide>(m);
    bind_texture_all<Guide>(m);

    m.attr("Float32") = m.attr("Float");
    m.attr("Int32") = m.attr("Int");
    m.attr("UInt32") = m.attr("UInt");
}
#endif
