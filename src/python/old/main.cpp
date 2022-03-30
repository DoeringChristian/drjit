#include "bind.h"
#include <drjit/autodiff.h>
#include <drjit/idiv.h>
#include <drjit/loop.h>
#include <drjit/texture.h>
#include <nanobind/stl.h>
#include <nanobind/operators.h>

extern void export_scalar(nb::module_ &m);

#if defined(DRJIT_ENABLE_PYTHON_PACKET)
extern void export_packet(nb::module_ &m);
#endif

#if defined(DRJIT_ENABLE_JIT)
#if defined(DRJIT_ENABLE_CUDA)
extern void export_cuda(nb::module_ &m);
#endif
extern void export_llvm(nb::module_ &m);
#if defined(DRJIT_ENABLE_AUTODIFF)
#if defined(DRJIT_ENABLE_CUDA)
extern void export_cuda_ad(nb::module_ &m);
#endif
extern void export_llvm_ad(nb::module_ &m);
#endif
#endif

const uint32_t var_type_size[(int) VarType::Count] {
    (uint32_t) -1, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 2, 4, 8
};

const bool var_type_is_float[(int) VarType::Count] {
    false, false, false, false, false, false, false, false,
    false, false, false, true, true, true
};

const bool var_type_is_unsigned[(int) VarType::Count] {
    false, false, false, true, false, true, false, true,
    false, true, true, false, false, false
};

const char* var_type_numpy[(int) VarType::Count] {
    "", "b1", "i1", "u1", "i2", "u2", "i4", "u4",
    "i8", "u8", "u8", "f2", "f4", "f8"
};

nb::handle array_base, array_name, array_init, tensor_init, array_configure;

/// Placeholder base of all DrJit arrays in the Python domain
struct ArrayBase { };

#if defined(DRJIT_ENABLE_JIT)
static void log_callback(LogLevel /* level */, const char *msg) {
    /* Try to print to the Python console if possible, but *never* risk
       deadlock over this. Calling nb::print() with an almost-finalized
       CPython interpreter can also fail. */
    if (PyGILState_Check() && !_Py_IsFinalizing())
        nb::print(msg);
    else
        fprintf(stderr, "%s\n", msg);
}
#endif

NB_MODULE(drjit_ext, m_) {
#if defined(DRJIT_ENABLE_JIT)
    jit_set_log_level_stderr(LogLevel::Disable);
    jit_set_log_level_callback(LogLevel::Warn, log_callback);
    jit_init_async((uint32_t) JitBackend::CUDA | (uint32_t) JitBackend::LLVM);
#endif

    (void) m_;
    nb::module_ m = nb::module_::import("drjit");

    // Look up some variables from the detail namespace
    nb::module_ array_detail = (nb::module_) m.attr("detail");
    array_name = array_detail.attr("array_name");
    array_init = array_detail.attr("array_init");
    tensor_init = array_detail.attr("tensor_init");
    array_configure = array_detail.attr("array_configure");

    m.attr("Dynamic") = dr::Dynamic;

    nb::enum_<VarType>(m, "VarType", nb::arithmetic())
        .value("Void", VarType::Void)
        .value("Bool", VarType::Bool)
        .value("Int8", VarType::Int8)
        .value("UInt8", VarType::UInt8)
        .value("Int16", VarType::Int16)
        .value("UInt16", VarType::UInt16)
        .value("Int32", VarType::Int32)
        .value("UInt32", VarType::UInt32)
        .value("Int64", VarType::Int64)
        .value("UInt64", VarType::UInt64)
        .value("Pointer", VarType::Pointer)
        .value("Float16", VarType::Float16)
        .value("Float32", VarType::Float32)
        .value("Float64", VarType::Float64)
        .def_property_readonly(
            "NumPy", [](VarType v) { return var_type_numpy[(int) v]; })
        .def_property_readonly(
            "Size", [](VarType v) { return var_type_size[(int) v]; });

    nb::enum_<dr::ADFlag>(m, "ADFlag", nb::arithmetic())
        .value("ClearNone", dr::ADFlag::ClearNone)
        .value("ClearEdges", dr::ADFlag::ClearEdges)
        .value("ClearInput", dr::ADFlag::ClearInput)
        .value("ClearInterior", dr::ADFlag::ClearInterior)
        .value("ClearVertices", dr::ADFlag::ClearVertices)
        .value("Default", dr::ADFlag::Default)
        .def(nb::self == nb::self)
        .def(nb::self | nb::self)
        .def(int() | nb::self)
        .def(nb::self & nb::self)
        .def(int() & nb::self)
        .def(+nb::self)
        .def(~nb::self);

    nb::enum_<dr::FilterMode>(m, "FilterMode")
        .value("Nearest", dr::FilterMode::Nearest)
        .value("Linear", dr::FilterMode::Linear);

    nb::enum_<dr::WrapMode>(m, "WrapMode")
        .value("Repeat", dr::WrapMode::Repeat)
        .value("Clamp", dr::WrapMode::Clamp)
        .value("Mirror", dr::WrapMode::Mirror);

    nb::class_<dr::detail::reinterpret_flag>(array_detail, "reinterpret_flag")
        .def(nb::init<>());

    array_base = nb::class_<ArrayBase>(m, "ArrayBase");

    nb::register_exception<drjit::Exception>(m, "Exception");
    array_detail.def("reinterpret_scalar", &reinterpret_scalar);
    array_detail.def("fmadd_scalar", [](double a, double b, double c) {
        return std::fma(a, b, c);
    });

#if defined(DRJIT_ENABLE_CUDA)
    array_detail.def("cuda_context", []() { return reinterpret_cast<std::uintptr_t>(jit_cuda_context()); });
    array_detail.def("cuda_device", &jit_cuda_device_raw);
    array_detail.def("cuda_stream", []() { return reinterpret_cast<std::uintptr_t>(jit_cuda_stream()); });
#endif

    export_scalar(m);

#if defined(DRJIT_ENABLE_PYTHON_PACKET)
    export_packet(m);
#endif

#if defined(DRJIT_ENABLE_JIT)
#if defined(DRJIT_ENABLE_CUDA)
    export_cuda(m);
#endif
    export_llvm(m);
#if defined(DRJIT_ENABLE_AUTODIFF)
    nb::enum_<dr::ADMode>(m, "ADMode")
        .value("Primal", dr::ADMode::Primal)
        .value("Forward", dr::ADMode::Forward)
        .value("Backward", dr::ADMode::Backward);

    nb::enum_<dr::detail::ADScope>(array_detail, "ADScope")
        .value("Suspend", dr::detail::ADScope::Suspend)
        .value("Resume", dr::detail::ADScope::Resume)
        .value("Isolate", dr::detail::ADScope::Isolate);

#if defined(DRJIT_ENABLE_CUDA)
    export_cuda_ad(m);
#endif
    export_llvm_ad(m);
    m.def("ad_whos_str", &dr::ad_whos);
    m.def("ad_whos", []() { nb::print(dr::ad_whos()); });
#endif

    struct Scope {
        Scope(const std::string &name) : name(name) { }

        void enter() {
            #if defined(DRJIT_ENABLE_JIT)
                #if defined(DRJIT_ENABLE_CUDA)
                    if (jit_has_backend(JitBackend::CUDA)) {
                        jit_prefix_push(JitBackend::CUDA, name.c_str());
                        pushed_cuda = true;
                    }
                #endif
                if (jit_has_backend(JitBackend::LLVM)) {
                    jit_prefix_push(JitBackend::LLVM, name.c_str());
                    pushed_llvm = true;
                }
            #endif
            #if defined(DRJIT_ENABLE_AUTODIFF)
                dr::ad_prefix_push(name.c_str());
            #endif
        }

        void exit(nb::handle, nb::handle, nb::handle) {
            #if defined(DRJIT_ENABLE_JIT)
                #if defined(DRJIT_ENABLE_CUDA)
                    if (pushed_cuda)
                        jit_prefix_pop(JitBackend::CUDA);
                #endif
                if (pushed_llvm)
                    jit_prefix_pop(JitBackend::LLVM);
            #endif
            #if defined(DRJIT_ENABLE_AUTODIFF)
                dr::ad_prefix_pop();
            #endif
        }

        std::string name;
        #if defined(DRJIT_ENABLE_JIT)
        #if defined(DRJIT_ENABLE_CUDA)
            bool pushed_cuda = false;
        #endif
            bool pushed_llvm = false;
        #endif
    };

    nb::class_<Scope>(m, "Scope")
        .def(nb::init<const std::string &>())
        .def("__enter__", &Scope::enter)
        .def("__exit__", &Scope::exit);

    nb::enum_<AllocType>(m, "AllocType")
        .value("Host", AllocType::Host)
        .value("HostAsync", AllocType::HostAsync)
        .value("HostPinned", AllocType::HostPinned)
        .value("Device", AllocType::Device)
        .value("Managed", AllocType::Managed)
        .value("ManagedReadMostly", AllocType::ManagedReadMostly);

    nb::enum_<JitBackend>(m, "JitBackend")
        .value("CUDA", JitBackend::CUDA)
        .value("LLVM", JitBackend::LLVM);

    nb::enum_<ReduceOp>(m, "ReduceOp")
        .value("None", ReduceOp::None)
        .value("Add",  ReduceOp::Add)
        .value("Mul",  ReduceOp::Mul)
        .value("Min",  ReduceOp::Min)
        .value("Max",  ReduceOp::Max)
        .value("And",  ReduceOp::And)
        .value("Or",   ReduceOp::Or);

    nb::enum_<JitFlag>(m, "JitFlag", nb::arithmetic())
        .value("ConstProp",        JitFlag::ConstProp)
        .value("ValueNumbering",   JitFlag::ValueNumbering)
        .value("LoopRecord",       JitFlag::LoopRecord)
        .value("LoopOptimize",     JitFlag::LoopOptimize)
        .value("VCallRecord",      JitFlag::VCallRecord)
        .value("VCallDeduplicate", JitFlag::VCallDeduplicate)
        .value("VCallOptimize",    JitFlag::VCallOptimize)
        .value("VCallInline",      JitFlag::VCallInline)
        .value("ForceOptiX",       JitFlag::ForceOptiX)
        .value("Recording",        JitFlag::Recording)
        .value("PrintIR",          JitFlag::PrintIR)
        .value("LaunchBlocking",   JitFlag::LaunchBlocking)
        .value("ADOptimize",       JitFlag::ADOptimize)
        .value("KernelHistory",    JitFlag::KernelHistory)
        .value("Default",          JitFlag::Default);

    nb::enum_<KernelType>(m, "KernelType")
        .value("JIT", KernelType::JIT)
        .value("Reduce", KernelType::Reduce)
        .value("VCallReduce", KernelType::VCallReduce)
        .value("Other", KernelType::Other);

#if defined(DRJIT_ENABLE_CUDA)
    m.def("device_count", &jit_cuda_device_count);
    m.def("set_device", &jit_cuda_set_device, "device"_a);
    m.def("device", &jit_cuda_device);
#endif

    m.def("has_backend", &jit_has_backend);
    m.def("sync_thread", &jit_sync_thread);
    m.def("sync_device", &jit_sync_device);
    m.def("sync_all_devices", &jit_sync_all_devices);
    m.def("whos_str", &jit_var_whos);
    m.def("whos", []() { nb::print(jit_var_whos()); });
    m.def("flush_kernel_cache", &jit_flush_kernel_cache);
    m.def("flush_malloc_cache", &jit_flush_malloc_cache);
    m.def("malloc_clear_statistics", &jit_malloc_clear_statistics);
    m.def("registry_trim", &jit_registry_trim);
    m.def("set_thread_count", &jit_llvm_set_thread_count);

    nb::object io = nb::module_::import("io");
    m.def(
        "kernel_history",
        [io](nb::list types) {
            KernelHistoryEntry *data  = jit_kernel_history();
            KernelHistoryEntry *entry = data;
            nb::list history;
            while (entry && (uint32_t) entry->backend) {
                nb::dict dict;
                dict["backend"] = entry->backend;
                dict["type"]    = entry->type;
                if (entry->type == KernelType::JIT) {
                    char kernel_hash[33];
                    snprintf(kernel_hash, sizeof(kernel_hash), "%016llx%016llx",
                             (unsigned long long) entry->hash[1],
                             (unsigned long long) entry->hash[0]);
                    dict["hash"] = kernel_hash;
                    dict["ir"]   = io.attr("StringIO")(entry->ir);
                    free(entry->ir);
                    dict["uses_optix"] = entry->uses_optix;
                    dict["cache_hit"]  = entry->cache_hit;
                    dict["cache_disk"] = entry->cache_disk;
                }
                dict["size"]         = entry->size;
                dict["input_count"]  = entry->input_count;
                dict["output_count"] = entry->output_count;
                if (entry->type == KernelType::JIT)
                    dict["operation_count"] = entry->operation_count;
                dict["codegen_time"]   = entry->codegen_time;
                dict["backend_time"]   = entry->backend_time;
                dict["execution_time"] = entry->execution_time;

                bool queried_type = types.empty();
                for (size_t i = 0; i < types.size(); ++i) {
                    KernelType t = types[i].template cast<KernelType>();
                    queried_type |= (t == entry->type);
                }

                if (queried_type)
                    history.append(dict);

                entry++;
            }
            free(data);
            return history;
        },
        "types"_a = nb::list());
    m.def("kernel_history_clear", &jit_kernel_history_clear);

    array_detail.def("graphviz", &jit_var_graphviz);
    array_detail.def("schedule", &jit_var_schedule);
    array_detail.def("eval", &jit_var_eval, nb::call_guard<nb::gil_scoped_release>());
    array_detail.def("eval", &jit_eval, nb::call_guard<nb::gil_scoped_release>());
    array_detail.def("to_dlpack", &to_dlpack, "owner"_a, "data"_a,
                     "type"_a, "device"_a, "shape"_a, "strides"_a);
    array_detail.def("from_dlpack", &from_dlpack);

#if defined(DRJIT_ENABLE_CUDA)
    array_detail.def("device", &jit_cuda_device);
#endif
    array_detail.def("device", &jit_var_device);

    array_detail.def("printf_async", [](bool cuda, uint32_t mask_index,
                                        const char *fmt,
                                        const std::vector<uint32_t> &indices) {
        jit_var_printf(cuda ? JitBackend::CUDA : JitBackend::LLVM, mask_index,
                       fmt, (uint32_t) indices.size(), indices.data());
    });

    m.def("set_flags", &jit_set_flags);
    m.def("set_flag", [](JitFlag f, bool v) { jit_set_flag(f, v); });
    m.def("flags", &jit_flags);
    m.def("flag", [](JitFlag f) { return jit_flag(f); });

    /* Register a cleanup callback function that is invoked when
       the 'drjit::ArrayBase' Python type is garbage collected */
    nb::cpp_function cleanup_callback([](nb::handle /* weakref */) {
        nb::gil_scoped_release gsr;
        jit_set_log_level_stderr(LogLevel::Warn);
        jit_set_log_level_callback(LogLevel::Disable, nullptr);
        jit_shutdown(false);
    });

    (void) nb::weakref(m.attr("ArrayBase"), cleanup_callback).release();
#else
    array_detail.def("schedule", [](uint32_t) { return false; });
    array_detail.def("eval", [](uint32_t) { return false; });
    array_detail.def("eval", []() { });
#endif

    array_detail.def("idiv", [](VarType t, nb::object value) -> nb::object {
        switch (t) {
            case VarType::Int32: {
                    drjit::divisor<int32_t> div(nb::cast<int32_t>(value));
                    return nb::make_tuple((int32_t) div.multiplier, div.shift);
                }
                break;

            case VarType::UInt32: {
                    drjit::divisor<uint32_t> div(nb::cast<uint32_t>(value));
                    return nb::make_tuple((uint32_t) div.multiplier, div.shift);
                }
                break;

            case VarType::Int64: {
                    drjit::divisor<int64_t> div(nb::cast<int64_t>(value));
                    return nb::make_tuple((int64_t) div.multiplier, div.shift);
                }
                break;

            case VarType::UInt64: {
                    drjit::divisor<uint64_t> div(nb::cast<uint64_t>(value));
                    return nb::make_tuple((uint64_t) div.multiplier, div.shift);
                }
                break;

            default:
                throw drjit::Exception("Unsupported integer type!");
        }
    });
}
