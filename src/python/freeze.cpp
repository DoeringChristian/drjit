
#include "freeze.h"

static const char *doc_freeze = R"(
    
)";

void freeze(nb::handle h){
    
}

void export_freeze(nb::module_ &m){
    m.def("freeze", &freeze, doc_freeze);
}
