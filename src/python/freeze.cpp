
#include "freeze.h"
#include "apply.h"
#include "drjit-core/jit.h"

static const char *doc_freeze = R"(
    
)";

nb::object freeze(nb::callable func){


    struct FlatVariables: TraverseCallback{
        std::vector<uint32_t> variables;
        JitBackend backend;
        
        bool inc_ref;

        FlatVariables() : inc_ref(false) {
        }

        void operator()(nb::handle h) override {
            auto s = supp(h.type());
            if (s.index)
                operator()(s.index(inst_ptr(h)));

            JitBackend var_backend = (JitBackend)s.backend;
            if(this->backend == var_backend || this->backend == JitBackend::None){
                this->backend = var_backend;
            }else{
                jit_fail("freeze(): backend missmatch error (backend %u does "
                         "not match %u)!",
                         (uint32_t)this->backend, (uint32_t)var_backend);
            }
        }

        void operator()(uint64_t index) override {
            if (inc_ref)
                ad_var_inc_ref(index);
            variables.push_back(index);
        }
    };
    
    auto new_func = nb::cpp_function([func](nb::args args) {

        FlatVariables in_variables;
        traverse("freeze", in_variables, args);
        
        JitBackend backend = in_variables.backend;

        jit_record_start(backend, in_variables.variables.data(), in_variables.variables.size());

        auto result = func(*args);
        
        FlatVariables out_variables;
        traverse("freeze", out_variables, result);
        if(out_variables.backend != backend){
            jit_fail("freeze(): backend missmatch error (backend %u of output "
                     "variables did not match backend %u of input variables)",
                     (uint32_t)out_variables.backend, (uint32_t)backend);
        }

        Recording *record = jit_record_stop(backend, out_variables.variables.data(), out_variables.variables.size());

        jit_record_destroy(record);
        
        return result;
    });

    return new_func;
}

void export_freeze(nb::module_ &m){
    m.def("freeze", &freeze, doc_freeze);
}
