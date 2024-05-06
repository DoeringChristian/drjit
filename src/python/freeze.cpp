
#include "freeze.h"
#include "apply.h"
#include "base.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "listobject.h"
#include "tupleobject.h"
#include<vector>

static const char *doc_freeze = R"(
    
)";

struct Layout{
    nb::object type;
    // TODO: unionize
    size_t num = 0;
    std::vector<nb::object> fields;
};

struct FlatVariables{
    
    std::vector<uint32_t> variables;
    std::vector<Layout> layout;
    JitBackend backend = JitBackend::None;

    void collect(nb::handle h){
        auto s = supp(h.type());

        JitBackend var_backend = (JitBackend)s.backend;

        if (this->backend == var_backend || this->backend == JitBackend::None) {
            jit_log(LogLevel::Info, "backend: %u", (uint32_t)var_backend);
            this->backend = var_backend;
        } else {
            jit_fail("freeze(): backend missmatch error (backend of this "
                     "variable %u does not match backend of others %u)!",
                     (uint32_t)var_backend, (uint32_t)this->backend);
        }

        if (s.index)
            variables.push_back(s.index(inst_ptr(h)));
    }

    void traverse(nb::handle h){
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

                    for (Py_ssize_t i = 0; i < len; ++i)
                        traverse(nb::steal(s.item(h.ptr(), i)));
                } else{
                    collect(h);
                }
            } else if (tp.is(&PyTuple_Type)){
                nb::tuple tuple = nb::borrow<nb::tuple>(h);
                
                Layout layout;
                layout.type = nb::borrow(tp);
                layout.num = tuple.size();
                this->layout.push_back(layout);
                
                for (nb::handle h2: tuple){
                    traverse(h2);
                }
            } else if (tp.is(&PyList_Type)){
                nb::list list = nb::borrow<nb::list>(h);
                
                Layout layout;
                layout.type = nb::borrow(tp);
                layout.num = list.size();
                this->layout.push_back(layout);
                
                for (nb::handle h2: list){
                    traverse(h2);
                }
            } else if (tp.is(&PyDict_Type)) {
                nb::dict dict = nb::borrow<nb::dict>(h);

                Layout layout;
                layout.type = nb::borrow(tp);
                layout.num = dict.size();
                layout.fields.reserve(layout.num);
                for (auto k: dict.keys()){
                    layout.fields.push_back(nb::borrow(k));
                }
                this->layout.push_back(layout);
                
                for (auto [k, v] : dict){
                    traverse(v);
                }
            } else{
                if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()){
                    
                    Layout layout;
                    layout.type = nb::borrow(tp);
                    layout.num = ds.size();
                    layout.fields.reserve(layout.num);
                    for (auto k: ds.keys()){
                        layout.fields.push_back(nb::borrow(k));
                    }
                    this->layout.push_back(layout);
                    
                    for (auto [k, v] : ds){
                        traverse(nb::getattr(h, k));
                    }
                } else if (nb::object df = get_dataclass_fields(tp); df.is_valid()){
                    
                    Layout layout;
                    layout.type = nb::borrow(tp);
                    for (auto field: df){
                        nb::object k = field.attr(DR_STR(name));
                        layout.fields.push_back(nb::borrow(k));
                    }
                    layout.num = layout.fields.size();
                    this->layout.push_back(layout);
                    
                    for (nb::handle field: df){
                        nb::object k = field.attr(DR_STR(name));
                        traverse(nb::getattr(h, k));
                    }
                } else if (nb::object cb = get_traverse_cb_ro(tp); cb.is_valid()){
                    // TODO: traverse callback
                }
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
};

nb::object freeze(nb::callable func){
    
    auto new_func = nb::cpp_function([func](nb::args args) {

        FlatVariables in_variables;
        in_variables.traverse(args);
        
        JitBackend backend = in_variables.backend;

        jit_record_start(backend, in_variables.variables.data(), in_variables.variables.size());

        auto result = func(*args);
        
        FlatVariables out_variables;
        out_variables.traverse(result);
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
