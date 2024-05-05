
#include "freeze.h"

static const char *doc_freeze = R"(
    
)";

nb::object freeze(nb::callable func){
    
    auto new_func = nb::cpp_function([=](nb::args args) {
        
        return func(*args);
    });

    return new_func;
}

void export_freeze(nb::module_ &m){
    m.def("freeze", &freeze, doc_freeze);
}
