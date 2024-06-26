/*
    dlpack.h -- Data exchange with other tensor frameworks

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern void export_dlpack(nb::module_ &);

extern nb::dlpack::dtype drjit_type_to_dlpack(VarType vt);
extern VarType dlpack_type_to_drjit(nb::dlpack::dtype vt);
