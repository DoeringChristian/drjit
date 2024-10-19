#include "freeze.h"
#include "autodiff.h"
#include "base.h"
#include "common.h"
#include "drjit-core/hash.h"
#include "drjit-core/jit.h"
#include "drjit/array_router.h"
#include "drjit/autodiff.h"
#include "drjit/extra.h"
#include "drjit/fwd.h"
#include "drjit/traversable_base.h"
#include "listobject.h"
#include "nanobind/intrusive/counter.h"
#include "nanobind/nanobind.h"
#include "object.h"
#include "pyerrors.h"
#include "shape.h"
#include "tupleobject.h"
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <vector>

struct ProfilerPhase {
    const char* message;
    ProfilerPhase(const char *message): message(message) {
        jit_log(LogLevel::Debug, "profiler start: %s", message);
        jit_profile_range_push(message);
    }

    ~ProfilerPhase() {
        jit_profile_range_pop();
        jit_log(LogLevel::Debug, "profiler end: %s", message);
    }
};

struct ADScopeContext{
    bool process_postponed;
    ADScopeContext(drjit::ADScope type, size_t size, const uint64_t *indices,
            int symbolic, bool process_postponed)
        : process_postponed(process_postponed) {
        ad_scope_enter(type, size, indices, symbolic);
    }
    ~ADScopeContext(){
        ad_scope_leave(process_postponed);
    }
};

static const char *doc_freeze = R"(
    
)";

enum class LayoutFlag : uint32_t {
    SingletonArray = (1 << 0),
    Unaligned = (1 << 1),
    GradEnabled = (1 << 2),
    Postponed = (1 << 3),
    Registry = (1 << 4),
};

/// Stores information about python objects, such as their type, their number of
/// sub-elements or their field keys. This can be used to reconstruct a pytree
/// from a flattened variable array.
struct Layout {
    /// Nanobind type of the container/variable
    nb::type_object type;
    /// Number of members in this container
    size_t num = 0;
    /// Optional field identifiers of the container
    /// for example: keys in dictionary
    std::vector<nb::object> fields;
    /// Optional drjit type of the variable
    VarType vt = VarType::Void;
    /// Optional evaluation state of the variable
    VarState vs = VarState::Invalid;
    /// Weather the variable is an array with a single entry.
    /// Such arrays are handled differently by the compiler.
    // bool singleton_array = false;
    // bool unaligned = false;
    // /// Weather this variable represents a value and it's gradient
    // /// The actual value and gradient layout is handled by the children.
    // bool grad_enabled = false;
    uint32_t flags = 0;
    /// The literal data
    uint64_t literal = 0;
    /// The index in the flat_variables array of this variable.
    /// This can be used to determine aliasing.
    uint32_t index = 0;
    /// We have to track the condition, where two variables have the same size
    /// during recording but don't when replaying.
    /// Therefore we de-duplicate the size.
    uint32_t size_index = 0;

    /// If a non drjit type is passed as function arguments or result, we simply
    /// cache it here.
    /// TODO: possibly do the same for literals?
    nb::object py_object = nb::none();

    bool operator==(const Layout &rhs) const {
        if (!(this->type.equal(rhs.type))) {
            jit_log(LogLevel::Warn, "    type");
            return false;
        }
        if (this->num != rhs.num) {
            jit_log(LogLevel::Warn, "    num: %u != %u", this->num, rhs.num);
            return false;
        }
        if (this->fields.size() != rhs.fields.size()) {
            jit_log(LogLevel::Warn, "    fields.size");
            return false;
        }
        for (uint32_t i = 0; i < this->fields.size(); ++i) {
            if (!(this->fields[i].equal(rhs.fields[i]))) {
                jit_log(LogLevel::Warn, "    fields[%u]", i);
                return false;
            }
        }
        if (this->vt != rhs.vt) {
            jit_log(LogLevel::Warn, "    vt");
            return false;
        }
        if (this->vs != rhs.vs) {
            jit_log(LogLevel::Warn, "    vs: %u != %u", (uint32_t)this->vs,
                    (uint32_t)rhs.vs);
            return false;
        }
        if (this->flags != rhs.flags) {
            jit_log(LogLevel::Warn, "    flags");
            return false;
        }
        if (this->index != rhs.index) {
            jit_log(LogLevel::Warn, "    index");
            return false;
        }
        if (this->size_index != rhs.size_index) {
            jit_log(LogLevel::Warn, "    size_index");
            return false;
        }
        if (this->literal != rhs.literal) {
            jit_log(LogLevel::Warn, "    literal");
            return false;
        }
        if (!this->py_object.equal(rhs.py_object)) {
            jit_log(LogLevel::Warn, "    object");
            return false;
        }
        return true;
    }
};

// Additional context required when traversing the inputs
struct TraverseContext{
    /// set of postponed ad nodes, used to mark inputs to functions.
    const tsl::robin_set<uint32_t, UInt32Hasher> *postponed = nullptr;
};

/**
 * A flattened representation of the PyTree.
 */
struct FlatVariables {

    // Variable, used to iterate over the variables/layouts when constructing
    // python objects
    uint32_t layout_index = 0;

    /// The flattened variable indices of the input/output to a frozen function
    std::vector<uint32_t> variables;
    /// Mapping from drjit variable index to index in flat variables
    tsl::robin_map<uint32_t, uint32_t> index_to_slot;

    /// We have to track the condition, where two variables have the same size
    /// during recording but don't when replaying.
    /// Therefore we de-duplicate the size.
    std::vector<uint32_t> sizes;
    tsl::robin_map<uint32_t, uint32_t> size_to_slot;

    /// This saves information about the type, size and fields of pytree
    /// objects. The information is stored in DFS order.
    std::vector<Layout> layout;
    JitBackend backend = JitBackend::None;

    // Wether variables should be borrowed, instead of stealing them
    bool borrow = true;

    FlatVariables() {
    }
    FlatVariables(bool borrow) : borrow(borrow) {
    }

    void clear() {
        this->layout_index = 0;
        this->variables.clear();
        this->index_to_slot.clear();
        this->layout.clear();
        this->backend = JitBackend::None;
    }
    void release() {
        for (uint32_t &index : this->variables) {
            jit_var_dec_ref(index);
        }
    }

    /**
     * Adds a variable to the flattened array, deduplicating it.
     * This allows for checking for aliasing conditions, as aliasing inputs map
     * to the same flat variable index.
     */
    uint32_t add_variable_index(uint32_t variable_index) {
        auto it = this->index_to_slot.find(variable_index);

        if (it == this->index_to_slot.end()) {
            uint32_t slot = this->variables.size();
            jit_log(LogLevel::Info, "    aliasing var r%u -> slot s%u",
                    variable_index, slot);

            // NOTE: an alternative to borrowing here would be to make `refcount
            // > 1` part of the layout, which would allow us to selectively
            // enable COW if it is necessary.
            if (borrow)
                jit_var_inc_ref(variable_index);
            this->variables.push_back(variable_index);

            this->index_to_slot.insert({variable_index, slot});
            return slot;
        } else {
            uint32_t slot = it.value();
            jit_log(LogLevel::Info,
                    "collect(): Found aliasing condition var r%u -> slot s%u",
                    variable_index, slot);
            return slot;
        }
    }

    /**
     * This function returns an index of an equivalence class for the variable
     * size in the flattened variables.
     * It uses a hashmap and vector to deduplicate sizes.
     *
     * This is necessary, to catch cases, where two variables had the same size
     * when freezing a function and two different sizes when replaying.
     * In that case one kernel would be recorded, that evaluates both variables.
     * However, when replaying two kernels would have to be launched since the
     * now differently sized variables can not be evaluated by the same kernel.
     */
    uint32_t add_size(uint32_t size) {
        auto it = this->size_to_slot.find(size);

        if (it == this->size_to_slot.end()) {
            uint32_t slot = this->sizes.size();
            jit_log(LogLevel::Info, "    aliasing size %u -> slot %u", size,
                    slot);

            this->sizes.push_back(size);

            this->size_to_slot.insert({size, slot});
            return slot;
        } else {
            uint32_t slot = it.value();
            jit_log(LogLevel::Info,
                    "collect(): Found aliasing condition size %u -> slot %u",
                    size, slot);
            return slot;
        }
    }

    /**
     * Traverse the jit index and add it to the flat variables.
     * An optional type python type can be supplied if it is known.
     */
    void traverse_jit_index(uint32_t index, TraverseContext &ctx, nb::handle tp = nb::none()) {
        VarInfo info = jit_set_backend(index);
        JitBackend var_backend = info.backend;

        if (this->backend == var_backend || this->backend == JitBackend::None) {
            this->backend = var_backend;
        } else {
            nb::raise("freeze(): backend missmatch error (backend of this "
                      "variable %u does not match backend of others %u)!",
                      (uint32_t)var_backend, (uint32_t)this->backend);
        }

        jit_log(LogLevel::Info,
                "collect(): collecting r%u, backend=%u, type=%u", index,
                var_backend, jit_var_type(index));

        if (jit_var_type(index) == VarType::Pointer) {
            // In order to support pointer inputs,
            // we would have to get the source variable, handle the case when
            // it's rc > 1 and potentially create a new pointer pointing to the
            // new source variable. Then we could add the new variable to the
            // flat variables.
            nb::raise("Pointer inputs not yet supported!");
        }

        Layout layout;
        VarState vs = jit_var_state(index);
        layout.type = nb::borrow<nb::type_object>(tp);
        layout.vs = vs;
        layout.vt = jit_var_type(index);
        layout.size_index = this->add_size(jit_var_size(index));

        if (vs == VarState::Literal) {
            jit_log(LogLevel::Debug, "    vs=Literal");
            jit_var_read(index, 0, &layout.literal);
            // Store size in index variable, as this is not used for literals
            layout.index = jit_var_size(index);
        } else if (vs == VarState::Evaluated) {
            jit_log(LogLevel::Debug, "    handling evaluate case");
            
            void *data = nullptr;
            uint32_t tmp = jit_var_data(index, &data);
            if(tmp != index)
                jit_fail("traverse(): An evaluated variable changed during evaluation!");
            jit_var_dec_ref(tmp);
            
            jit_log(LogLevel::Debug, "    vs=%s, data=%p", jit_var_kind_name(index), data);
            layout.index = this->add_variable_index(index);
            // bool singleton_array = jit_var_size(index) == 1;
            bool unaligned = jit_var_is_unaligned(index);

            layout.flags |=
                (jit_var_size(index) == 1 ? (uint32_t)LayoutFlag::SingletonArray
                                          : 0);
            layout.flags |=
                (jit_var_is_unaligned(index) ? (uint32_t)LayoutFlag::Unaligned
                                             : 0);

        } else {
            jit_raise("collect(): found variable %u in unsupported state %u!",
                      index, (uint32_t) vs);
        }
        this->layout.push_back(layout);
    }
    /**
     * Add an ad variable by it's index.
     * Both the value and gradient is added to the flattened variables.
     * If the ad index has been marked as postponed in the
     * \TraverseContext.postponed field, we mark the resulting layout with that
     * flag. 
     * The function takes an optional python-type if that is known.
     */
    void traverse_ad_index(uint64_t index, TraverseContext &ctx, nb::handle tp = nb::none()) {
        ProfilerPhase profiler("traverse_ad_index");
        int grad_enabled = ad_grad_enabled(index);
        jit_log(LogLevel::Debug,
                "traverse(): a%u, r%u",
                (uint32_t)(index >> 32), (uint32_t)index, grad_enabled);
        if (grad_enabled) {
            uint32_t ad_index = (uint32_t)(index >> 32);
            
            jit_log(LogLevel::Debug, " => collecting ad var");
            Layout layout;
            layout.type = nb::borrow<nb::type_object>(tp);
            layout.num = 2;
            layout.vt = jit_var_type(index);
            
            // Set flags
            layout.flags |= (uint32_t)LayoutFlag::GradEnabled;
            // If the edge with this node as it's target has been postponed by
            // the isolate gradient scope, it has been enqueued and we mark the
            // ad variable as such.
            if(ctx.postponed && ctx.postponed->contains(ad_index)){
                layout.flags |= (uint32_t)LayoutFlag::Postponed;
                jit_log(LogLevel::Debug, "traverse(): found postponed ad_variable a%u", ad_index);
            }else
                jit_log(LogLevel::Debug,
                        "traverse(): found ad variable a%u that has not been "
                        "postponed",
                        ad_index);
            
            this->layout.push_back(layout);

            traverse_jit_index(index, ctx, tp);
            uint32_t grad = ad_grad(index);
            traverse_jit_index(grad, ctx, tp);
            jit_var_dec_ref(grad);
        } else {
            jit_log(LogLevel::Debug, " => collecting jit var");
            traverse_jit_index(index, ctx, tp);
        }
    }

    /**
     * Wrapper arround traverse_ad_index for a python variable handle.
     */
    void traverse_ad_var(nb::handle h, TraverseContext &ctx) {
        auto s = supp(h.type());

        raise_if(s.index == nullptr, "freeze(): ArraySupplement index function "
                                     "pointer is nullptr.");

        uint64_t index = s.index(inst_ptr(h));

        this->traverse_ad_index(index, ctx, h.type());
    }

    /**
     * Traverse a c++ tree using it's `traverse_1_cb_ro` callback.
     */
    void traverse_cb(const drjit::TraversableBase *traversable, TraverseContext &ctx,
                     nb::object type = nb::none()) {
        Layout layout;
        layout.type = nb::borrow<nb::type_object>(type);
        size_t layout_index = this->layout.size();
        this->layout.push_back(layout);

        uint32_t num_fileds = 0;

        struct Payload {
            FlatVariables *flat_vars;
            uint32_t num_fields;
            TraverseContext *ctx;
        };
        Payload payload{this, 0, &ctx};
        traversable->traverse_1_cb_ro(
            (void *)&payload, [](void *p, uint64_t index) {
                if(!index)
                    return;
                Payload *payload = (Payload *)p;
                payload->num_fields++;
                payload->flat_vars->traverse_ad_index(index, *payload->ctx);
            });

        this->layout[layout_index].num = payload.num_fields;
    }

    /**
     * Traverse a polymorphic variable, and it's subtypes.
     */
    void traverse_class_var(nb::handle h, TraverseContext &ctx) {
        const ArraySupplement &s = supp(h.type());

        jit_log(LogLevel::Debug,
                "traverse(): Found class variable on backend %u", backend);

        // Add the base type (hopefully)
        size_t layout_index = this->layout.size();
        traverse_ad_var(h, ctx);

        JitBackend backend = (JitBackend)s.backend;
        nb::str domain = nb::borrow<nb::str>(nb::getattr(h.type(), "Domain"));

        uint32_t id_bound = jit_registry_id_bound(backend, domain.c_str());

        jit_log(LogLevel::Debug,
                "traverse(): Traversing %u instances of domain \"%s\".",
                id_bound, domain.c_str());

        // We use types of the subtypes as fields.
        std::vector<nb::object> fields;
        for (uint32_t id = 0; id < id_bound; ++id) {
            void *ptr = jit_registry_ptr(backend, domain.c_str(), id + 1);
            if (!ptr)
                continue;

            // WARN: very unsafe cast!
            nb::intrusive_base *base = (nb::intrusive_base *)ptr;
            const drjit::TraversableBase *traversable =
                (drjit::TraversableBase *)base;
            nb::handle inst_obj = base->self_py();

            if (inst_obj.ptr()) {
                fields.push_back(nb::borrow(inst_obj.type()));
                traverse(inst_obj, ctx);
            } else if (traversable) {
                traverse_cb(traversable, ctx);
            } else {
                nb::raise("Could not traverse non-python sub-type!");
            }
        }

        this->layout[layout_index].num = id_bound;
        this->layout[layout_index].fields = std::move(fields);
    }

    /**
     * Traverse andy drjit variable.
     * It can be polimorphic or an ad variable.
     */
    void traverse_dr_var(nb::handle h, TraverseContext &ctx) {
        nb::handle tp = h.type();

        auto s = supp(h.type());

        // Handle class var
        if (s.is_class) {
            traverse_class_var(h, ctx);
        } else {
            traverse_ad_var(h, ctx);
        }
    }

    /**
     * Traverses a PyTree in DFS order, and records it's layout in the
     * `layout` vector.
     *
     * When hitting a drjit primitive type, it calls the
     * `traverse_dr_var` method, which will add their indices to the `flat_variables`
     * vector.
     * The collect method will also record metadata about the drjit variable in
     * the layout.
     * Therefore, the layout can be used as an identifier to the recording of
     * the frozen function.
     */
    void traverse(nb::handle h, TraverseContext &ctx) {
        ProfilerPhase profiler("traverse");
        nb::handle tp = h.type();

        nb::print(tp);
        nb::print("{");

        try {
            if (is_drjit_type(tp)) {
                const ArraySupplement &s = supp(tp);
                if (s.is_tensor) {
                    nb::handle array = s.tensor_array(h.ptr());

                    Layout layout;
                    layout.type = nb::borrow<nb::type_object>(tp);
                    layout.py_object = shape(h);
                    layout.num = width(array);
                    this->layout.push_back(layout);

                    traverse(nb::steal(array), ctx);
                } else if (s.ndim != 1) {
                    Py_ssize_t len = s.shape[0];
                    if (len == DRJIT_DYNAMIC)
                        len = s.len(inst_ptr(h));

                    Layout layout;
                    layout.type = nb::borrow<nb::type_object>(tp);
                    layout.num = len;
                    this->layout.push_back(layout);

                    for (Py_ssize_t i = 0; i < len; ++i)
                        traverse(nb::steal(s.item(h.ptr(), i)), ctx);
                } else {
                    traverse_dr_var(h, ctx);
                }
            } else if (tp.is(&PyTuple_Type)) {
                nb::tuple tuple = nb::borrow<nb::tuple>(h);

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.num = tuple.size();
                this->layout.push_back(layout);

                for (nb::handle h2 : tuple) {
                    traverse(h2, ctx);
                }
            } else if (tp.is(&PyList_Type)) {
                nb::list list = nb::borrow<nb::list>(h);

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.num = list.size();
                this->layout.push_back(layout);

                for (nb::handle h2 : list) {
                    traverse(h2, ctx);
                }
            } else if (tp.is(&PyDict_Type)) {
                nb::dict dict = nb::borrow<nb::dict>(h);

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.num = dict.size();
                layout.fields.reserve(layout.num);
                for (auto k : dict.keys()) {
                    layout.fields.push_back(nb::borrow(k));
                }
                this->layout.push_back(layout);

                for (auto [k, v] : dict) {
                    traverse(v, ctx);
                }
            } else if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.num = ds.size();
                layout.fields.reserve(layout.num);
                for (auto k : ds.keys()) {
                    layout.fields.push_back(nb::borrow(k));
                }
                this->layout.push_back(layout);

                for (auto [k, v] : ds) {
                    traverse(nb::getattr(h, k), ctx);
                }
            } else if (nb::object df = get_dataclass_fields(tp);
                       df.is_valid()) {

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                for (auto field : df) {
                    nb::object k = field.attr(DR_STR(name));
                    layout.fields.push_back(nb::borrow(k));
                }
                layout.num = layout.fields.size();
                this->layout.push_back(layout);

                for (nb::handle field : df) {
                    nb::object k = field.attr(DR_STR(name));
                    traverse(nb::getattr(h, k), ctx);
                }
            } else if (nb::object cb = get_traverse_cb_ro(tp); cb.is_valid()) {
                ProfilerPhase profiler("traverse cb");
                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                size_t layout_index = this->layout.size();
                this->layout.push_back(layout);

                uint32_t num_fields = 0;

                // Traverse the opaque C++ object
                cb(h, nb::cpp_function([&](uint64_t index) {
                       if (!index)
                           return;
                       jit_log(LogLevel::Debug,
                               "traverse(): traverse_cb[%u] = a%u r%u",
                               num_fields, (uint32_t) (index >> 32),
                               (uint32_t) index);
                       num_fields++;
                       this->traverse_ad_index(index, ctx, nb::none());
                       return;
                   }));

                // Update layout number of fields
                this->layout[layout_index].num = num_fields;
            } else if (tp.is(&_PyNone_Type)) {
                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                this->layout.push_back(layout);
            } else {
                jit_log(LogLevel::Warn,
                        "traverse(): You passed a value to a frozen function, "
                        "that could not be converted to Dr.Jit types. This is "
                        "not recommended and the value will be cached.",
                        nb::type_name(tp).c_str());

                Layout layout;
                layout.type = nb::borrow<nb::type_object>(tp);
                layout.py_object = nb::borrow<nb::object>(h);
                this->layout.push_back(layout);
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

        nb::print("}");
    }

    /**
     * First traverses the whole registry and then the handle provided.
     */
    void traverse_with_registry(nb::handle h, TraverseContext &ctx){
        // Traverse the registry
        {
            Layout layout;
            layout.type = nb::borrow<nb::type_object>(nb::none());
            size_t layout_index = this->layout.size();
            this->layout.push_back(layout);

            uint32_t num_fields = 0;
            
            jit_log(LogLevel::Debug, "registry{");
            uint32_t registry_bound = jit_registry_id_bound(JitBackend::None, nullptr);
            std::vector<void*> registry_pointers;
            registry_pointers.resize(registry_bound);
            jit_registry_fill_ptrs(registry_pointers.data());

            jit_log(LogLevel::Debug, "registry_bound=%u", registry_bound);
            jit_log(LogLevel::Debug, "layout_index=%u", this->layout.size());
            for (void *ptr : registry_pointers) {
                jit_log(LogLevel::Debug, "ptr=%p", ptr);
                if(!ptr)
                    continue;

                const drjit::TraversableBase *traversable =
                    (drjit::TraversableBase *) ptr;

                traverse_cb(traversable, ctx);
                num_fields++;
            }
            jit_log(LogLevel::Debug, "}");
            
            this->layout[layout_index].num = num_fields;
        }

        // Traverse the rest
        traverse(h, ctx);
    }

    /**
     * Construct a variable, given it's layout.
     * This is the counterpart to `traverse_jit_index`.
     */
    uint32_t construct_jit_index(const Layout &layout) {
        if (layout.vs == VarState::Literal) {
            uint32_t index = jit_var_literal(this->backend, layout.vt,
                                             &layout.literal, layout.index);

            return index;
        } else {
            uint32_t index = this->variables[layout.index];
            jit_log(LogLevel::Debug, "    uses output[%u] = r%u", layout.index, index);

            jit_var_inc_ref(index);

            return index;
        }
    }

    /**
     * Construct/assign the variable index given a layout.
     * This corresponds to `traverse_ad_index`>
     *
     * This function is also used for assignment to ad-variables.
     * If a `prev_index` is provided, and it is an ad-variable the gradient and
     * value of the flat variables will be applied to the ad variable,
     * preserving the ad_idnex.
     *
     * It returns a owning reference.
     */
    uint64_t construct_ad_index(const Layout &layout, uint32_t shrink = 0, uint64_t prev_index = 0) {
        uint64_t index;
        if ((layout.flags & (uint32_t)LayoutFlag::GradEnabled) != 0) {
            bool postponed = (layout.flags & (uint32_t)LayoutFlag::Postponed);
                
            Layout &val_layout = this->layout[layout_index++];
            uint32_t val = construct_jit_index(val_layout);

            Layout &grad_layout = this->layout[layout_index++];
            uint32_t grad = construct_jit_index(grad_layout);

            // Resize the gradient if it is a literal
            if ((VarState)jit_var_state(grad) == VarState::Literal) {
                uint32_t new_grad = jit_var_resize(grad, jit_var_size(val));
                jit_var_dec_ref(grad);
                grad = new_grad;
            }

            // If the prev_index variable is provided we assign the new value
            // and gradient to the ad variable of that index instead of creating
            // a new one.
            uint32_t ad_index = (uint32_t) (prev_index >> 32);
            if(ad_index){
                index = (((uint64_t) ad_index) << 32) | ((uint64_t) val);
                ad_var_inc_ref(index);
            } else
                index = ad_var_new(val);

            jit_log(LogLevel::Debug, " -> ad_var r%zu", index);
            jit_var_dec_ref(val);

            // Equivalent to set_grad
            ad_clear_grad(index);
            ad_accum_grad(index, grad);
            jit_var_dec_ref(grad);

            // Variables, that have been postponed by the isolate gradient scope
            // will be enqueued, which propagates their gradeint to previous
            // functions.
            if (ad_index && postponed) {
                ad_enqueue(drjit::ADMode::Backward, index);
            }
        } else {
            index = construct_jit_index(layout);
        }

        if (shrink > 0)
            index = ad_var_shrink(index, shrink);
        return index;
    }

    /**
     * Construct an ad variable given it's layout.
     * This corresponds to `traverse_ad_var`
     */
    nb::object construct_ad_var(const Layout &layout,
                                     uint32_t shrink = 0) {
        uint64_t index = construct_ad_index(layout, shrink);

        auto result = nb::inst_alloc_zero(layout.type);
        const ArraySupplement &s = supp(result.type());
        s.init_index(index, inst_ptr(result));

        // Have to decrement reference, as it is not part of `variables` and
        // will not be freed
        ad_var_dec_ref(index);

        return result;
    }

    /**
     * Construct a drjit variable.
     * Corresponds to `traverse_dr_var`.
     */
    nb::object construct_dr_var(const Layout &layout, uint32_t shrink = 0) {
        const ArraySupplement &s = supp(layout.type);

        if (s.is_class) {
            nb::raise("Tried to construct a polymorphic drjit variable");
        } else {
            return construct_ad_var(layout, shrink);
        }
    }

    /**
     * This is the counterpart to the traverse method.
     * Given a layout vector and flat_variables, it re-constructs the PyTree.
     */
    nb::object construct() {
        if (this->layout.size() == 0) {
            return nb::none();
        }

        const Layout &layout = this->layout[layout_index++];
        jit_log(LogLevel::Debug, "construct(): type=%s",
                nb::type_name(layout.type).c_str());
        if (layout.type.is(nb::none().type())) {
            return nb::none();
        }
        if (is_drjit_type(layout.type)) {
            const ArraySupplement &s = supp(layout.type);
            if (s.is_tensor) {
                const Layout &array_layout = this->layout[layout_index++];
                nb::object array = construct_dr_var(array_layout, layout.num);

                return layout.type(array, layout.py_object);
            } else if (s.ndim != 1) {
                auto result = nb::inst_alloc_zero(layout.type);
                dr::ArrayBase *p = inst_ptr(result);
                size_t size = s.shape[0];
                if (size == DRJIT_DYNAMIC) {
                    size = s.len(p);
                    s.init(size, p);
                }
                for (size_t i = 0; i < size; ++i) {
                    result[i] = construct();
                }
                return result;
            } else {
                return construct_dr_var(layout);
            }
        } else if (layout.type.is(&PyTuple_Type)) {
            nb::list list;
            for (uint32_t i = 0; i < layout.num; ++i) {
                list.append(construct());
            }
            return nb::tuple(list);
        } else if (layout.type.is(&PyList_Type)) {
            nb::list list;
            for (uint32_t i = 0; i < layout.num; ++i) {
                list.append(construct());
            }
            return list;
        } else if (layout.type.is(&PyDict_Type)) {
            nb::dict dict;
            for (auto k : layout.fields) {
                dict[k] = construct();
            }
            return dict;
        } else if (nb::dict ds = get_drjit_struct(layout.type); ds.is_valid()) {
            nb::object tmp = layout.type();
            // TODO: validation against `ds`
            for (auto k : layout.fields) {
                nb::setattr(tmp, k, construct());
            }
            return tmp;
        } else if (nb::object df = get_dataclass_fields(layout.type);
                   df.is_valid()) {
            nb::dict dict;
            for (auto k : layout.fields) {
                dict[k] = construct();
            }
            return layout.type(**dict);
        } else {
            if(layout.py_object.is_none()){
                nb::raise("Tried to construct a variable that is not constructable!");
            }
            return layout.py_object;
        }
    }

    /**
     * Assigns an ad variable.
     * Corresponds to `traverse_ad_var`.
     * This uses `construct_ad_index` to either construct a new ad variable or
     * assign the value and gradient to an already existing one.
     */
    void assign_ad_var(Layout &layout, nb::handle dst) {
        const ArraySupplement &s = supp(layout.type);

        uint64_t index;
        if(s.index){
            index = construct_ad_index(layout, 0, s.index(inst_ptr(dst)));
        } else
            index = construct_ad_index(layout);

        s.reset_index(index, inst_ptr(dst));
        jit_log(LogLevel::Debug,
                "index=%zu, grad_enabled=%u, ad_grad_enabled=%u", index,
                grad_enabled(dst), ad_grad_enabled(index));

        ad_var_dec_ref(index);
    }

    /**
     * Helper function, used to assign a callback variable.
     */
    uint64_t assign_cb_internal(uint64_t index, std::vector<uint64_t> &tmp){
        if(!index)
            return index;
        Layout &layout = this->layout[layout_index++];

        uint64_t new_index = this->construct_ad_index(layout, 0, index);

        if (layout.vt != (VarType)jit_var_type(index))
            jit_fail("VarType missmatch %u != %u while assigning (a%u, r%u) -> (a%u, r%u)!",
                     (uint32_t)layout.vt,
                     (uint32_t)jit_var_type(index),
                     (uint32_t)(index >> 32), (uint32_t)index, (uint32_t)(new_index >> 32), (uint32_t)new_index
                     );
    
        tmp.push_back(new_index);
        return new_index;
    }

    /**
     * Assigns variables using it's `traverse_cb_rw` callback.
     * This corresponds to `traverse_cb`.
     */
    void assign_cb(drjit::TraversableBase *traversable) {
        Layout &layout = this->layout[layout_index++];
        
        struct Payload {
            FlatVariables *flat_vars;
            std::vector<uint64_t> tmp;
            uint32_t num_fields;
            uint32_t field_counter;
        };
        jit_log(LogLevel::Debug, "    layout.num=%u", layout.num);
        Payload payload{ this, std::vector<uint64_t>(), (uint32_t) layout.num,
                         0 };
        traversable->traverse_1_cb_rw((void *) &payload, [](void *p,
                                                            uint64_t index) {
            if (!index)
                return index;
            Payload *payload = (Payload *) p;
            jit_log(LogLevel::Debug, "    field_counter=%u", payload->field_counter);
            if (payload->field_counter >= payload->num_fields)
                jit_raise("While traversing an object "
                          "for assigning the inputs, the number of "
                          "variables to assign did not match the "
                          "number of variables traversed when recording!");
            payload->field_counter++;

            return payload->flat_vars->assign_cb_internal(index, payload->tmp);
        });
        if (payload.field_counter != layout.num)
            jit_raise("While traversing and object "
                      "for assigning the inputs, the number of "
                      "variables to assign did not match the "
                      "number of variables traversed when recording!");
        for (uint64_t index : payload.tmp)
            ad_var_dec_ref(index);
    }

    
    /**
     * Assign a polymorphic variable and it's subtypes.
     * Corresponds to `traverse_class_var`
     */
    void assign_class_var(Layout &layout, nb::handle dst) {
        const ArraySupplement &s = supp(layout.type);

        JitBackend backend = (JitBackend)s.backend;
        nb::str domain =
            nb::borrow<nb::str>(nb::getattr(layout.type, "Domain"));

        uint32_t id_bound = jit_registry_id_bound(backend, domain.c_str());
        if (id_bound != layout.num) {
            jit_fail("Number of sub-types registered with backend changed "
                     "during recording!");
        }

        assign_ad_var(layout, dst);

        for (uint32_t id = 0; id < id_bound; ++id) {
            void *ptr = jit_registry_ptr(backend, domain.c_str(), id + 1);
            if (!ptr)
                continue;

            // WARN: very unsafe cast!
            nb::intrusive_base *base = (nb::intrusive_base *)ptr;
            drjit::TraversableBase *traversable =
                dynamic_cast<drjit::TraversableBase *>(base);
            nb::handle inst_obj = base->self_py();

            if (inst_obj.ptr()) {
                assign(inst_obj);
            } else if (traversable) {
                assign_cb(traversable);
            } else {
                nb::raise("Could not traverse non-python sub-type!");
            }
        }
    }

    /**
     * Assign a drjit variable.
     * Corresponds to `traverse_dr_var`.
     */
    void assign_dr_var(Layout &layout, nb::handle dst) {
        const ArraySupplement &s = supp(layout.type);
        if (s.is_class)
            assign_class_var(layout, dst);
        else
            assign_ad_var(layout, dst);
    }

    /**
     * Assigns the flattened variables to an already existing PyTree.
     * This is used when input variables are changed.
     */
    void assign(nb::handle dst) {
        nb::handle tp = dst.type();
        Layout &layout = this->layout[layout_index++];

        nb::print(tp);
        nb::print("{");

        if (!layout.type.equal(tp))
            nb::raise("Type missmatch! Type of original object %s does not "
                      "match type of new object %s.",
                      nb::type_name(tp).c_str(),
                      nb::type_name(layout.type).c_str());

        try {
            if (is_drjit_type(tp)) {
                const ArraySupplement &s = supp(tp);

                if (s.is_tensor) {
                    nb::handle array = s.tensor_array(dst.ptr());

                    Layout &array_layout = this->layout[layout_index++];
                    
                    assign_dr_var(array_layout, array);
                } else if (s.ndim != 1) {
                    Py_ssize_t len = s.shape[0];
                    if (len == DRJIT_DYNAMIC)
                        len = s.len(inst_ptr(dst));

                    for (Py_ssize_t i = 0; i < len; ++i)
                        assign(dst[i]);
                } else {
                    assign_dr_var(layout, dst);
                }
            } else if (tp.is(&PyTuple_Type)) {
                nb::tuple tuple = nb::borrow<nb::tuple>(dst);
                raise_if(tuple.size() != layout.num, "");

                for (nb::handle h2 : tuple)
                    assign(h2);
            } else if (tp.is(&PyList_Type)) {
                nb::list list = nb::borrow<nb::list>(dst);
                raise_if(list.size() != layout.num, "");

                for (nb::handle h2 : list)
                    assign(h2);
            } else if (tp.is(&PyDict_Type)) {
                nb::dict dict = nb::borrow<nb::dict>(dst);
                for (auto &k : layout.fields) {
                    if (dict.contains(&k))
                        assign(dict[k]);
                    else
                        dst[k] = construct();
                }
            } else if (nb::dict ds = get_drjit_struct(dst); ds.is_valid()) {
                for (auto &k : layout.fields) {
                    if (nb::hasattr(dst, k))
                        assign(nb::getattr(dst, k));
                    else
                        nb::setattr(dst, k, construct());
                }
            } else if (nb::object df = get_dataclass_fields(tp);
                       df.is_valid()) {
                for (auto k : layout.fields) {
                    if (nb::hasattr(dst, k))
                        assign(nb::getattr(dst, k));
                    else
                        nb::setattr(dst, k, construct());
                }
            } else if (nb::object cb = get_traverse_cb_rw(tp); cb.is_valid()) {
                std::vector<uint64_t> tmp;
                uint32_t num_fields = 0;
                
                cb(dst, nb::cpp_function([&](uint64_t index) {
                       if (!index)
                           return index;
                       jit_log(LogLevel::Debug,
                               "assign(): traverse_cb[%u] was a%u r%u",
                               num_fields, (uint32_t) (index >> 32),
                               (uint32_t) index);
                       num_fields++;
                       if (num_fields > layout.num)
                           jit_raise(
                               "While traversing the object of type %s "
                               "for assigning the inputs, the number of "
                               "variables to assign did not match the "
                               "number of variables traversed when recording!",
                               nb::str(tp).c_str());
                       return assign_cb_internal(index, tmp);
                   }));
                if (num_fields != layout.num)
                    jit_raise("While traversing the object of type %s "
                              "for assigning the inputs, the number of "
                              "variables to assign did not match the "
                              "number of variables traversed when recording!",
                              nb::str(tp).c_str());
                for (uint64_t index : tmp)
                    ad_var_dec_ref(index);
            } else {
            }
        } catch (nb::python_error &e) {
            nb::raise_from(e, PyExc_RuntimeError,
                           "FlatVariables::assign(): error encountered while "
                           "processing an argument "
                           "of type '%U' (see above).",
                           nb::type_name(tp).ptr());
        } catch (const std::exception &e) {
            nb::chain_error(PyExc_RuntimeError,
                            "FlatVariables::assign(): error encountered "
                            "while processing an argument "
                            "of type '%U': %s",
                            nb::type_name(tp).ptr(), e.what());
            nb::raise_python_error();
        }

        nb::print("}");
    }

    /**
     * First assigns the registry and then the PyTree.
     * Corresponds to `traverse_with_registry`.
     */
    void assign_with_registry(nb::handle dst){

        // Assign registry
        Layout &layout = this->layout[layout_index++];
        uint32_t num_fields = 0;
        jit_log(LogLevel::Debug, "registry{");
        uint32_t registry_bound = jit_registry_id_bound(JitBackend::None, nullptr);
        std::vector<void*> registry_pointers;
        registry_pointers.resize(registry_bound);
        jit_registry_fill_ptrs(registry_pointers.data());
        
        jit_log(LogLevel::Debug, "registry_bound=%u", registry_bound);
        jit_log(LogLevel::Debug, "layout_index=%u", this->layout_index);
        for (void *ptr : registry_pointers) {
            jit_log(LogLevel::Debug, "ptr=%p", ptr);
            if(!ptr)
                continue;
            
            drjit::TraversableBase *traversable =
                (drjit::TraversableBase *) ptr;

            assign_cb(traversable);
            num_fields++;
        }
        jit_log(LogLevel::Debug, "}");

        // Assign rest
        assign(dst);
    }

    void log_layout() const {
        for (uint32_t i = this->layout_index; i < this->layout.size(); ++i) {
            const Layout &layout = this->layout[i];
            jit_log(LogLevel::Debug, "layout.type=%s, layout.num=%u",
                    nb::type_name(layout.type).c_str(), layout.num);
        }
    }
};

struct TransformInPlaceCallback {
    // The transform operation, applied to each index.
    // Should return an owning reference.
    virtual uint64_t operator()(uint64_t index) {
        return index;
    };
};

static void transform_in_place(nb::handle h, TransformInPlaceCallback &op);

static void transform_in_place_dr_flat(nb::handle h,
                                       TransformInPlaceCallback &op) {
    nb::handle tp = h.type();
    nb::print(tp);

    const ArraySupplement &s = supp(tp);

    auto index_fn = s.index;
    if (!index_fn)
        jit_fail("Index function not set!");
    uint64_t index = index_fn(inst_ptr(h));
    uint64_t new_index = op(index);
    s.reset_index(new_index, inst_ptr(h));
    ad_var_dec_ref(new_index);
}

void transform_in_place_traversable(drjit::TraversableBase *traversable,
                                    TransformInPlaceCallback &cb) {
    struct Payload {
        TransformInPlaceCallback &cb;
        std::vector<uint64_t> tmp;
    };
    Payload payload{cb, std::vector<uint64_t>()};
    traversable->traverse_1_cb_rw((void *)&payload,
                                  [](void *p, uint64_t index) {
                                      Payload *payload = (Payload *)p;

                                      uint64_t new_index = payload->cb(index);
                                      payload->tmp.push_back(new_index);
                                      return new_index;
                                  });

    for (uint64_t index : payload.tmp) {
        ad_var_dec_ref(index);
    }
}

static void transform_in_place_dr_class(nb::handle h,
                                        TransformInPlaceCallback &op) {
    nb::handle tp = h.type();
    const ArraySupplement &s = supp(tp);

    transform_in_place_dr_flat(h, op);

    JitBackend backend = (JitBackend)s.backend;
    nb::str domain = nb::borrow<nb::str>(nb::getattr(tp, "Domain"));

    uint32_t id_bound = jit_registry_id_bound(backend, domain.c_str());

    jit_log(LogLevel::Debug, "transforming %u subtypes of domain %s", id_bound,
            domain.c_str());
    for (uint32_t id = 0; id < id_bound; ++id) {
        void *ptr = jit_registry_ptr(backend, domain.c_str(), id + 1);
        jit_log(LogLevel::Debug, "traversing subtype with instance at %p", ptr);
        if (!ptr)
            continue;

        // WARN: very unsafe cast!
        nb::intrusive_base *base = (nb::intrusive_base *)ptr;
        drjit::TraversableBase *traversable = (drjit::TraversableBase *)base;
        nb::handle inst_obj = base->self_py();

        if (inst_obj.ptr()) {
            transform_in_place(inst_obj, op);
        } else if (traversable) {
            transform_in_place_traversable(traversable, op);
        } else {
            nb::raise("Could not traverse non-python sub-type!");
        }
    }
}
static void transform_in_place_dr(nb::handle h, TransformInPlaceCallback &op) {
    nb::handle tp = h.type();

    const ArraySupplement &s = supp(tp);

    if (s.is_class)
        transform_in_place_dr_class(h, op);
    else
        transform_in_place_dr_flat(h, op);
}

static void transform_in_place(nb::handle h, TransformInPlaceCallback &op) {
    nb::handle tp = h.type();

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);
        if (s.is_tensor) {
            nb::object array = nb::steal(s.tensor_array(h.ptr()));
            transform_in_place(array, op);
        } else if (s.ndim > 1) {
            Py_ssize_t len = s.shape[0];
            if (len == DRJIT_DYNAMIC)
                len = s.len(inst_ptr(h));

            for (Py_ssize_t i = 0; i < len; ++i) {
                transform_in_place(h[i], op);
            }
        } else {
            transform_in_place_dr(h, op);
        }
    } else if (tp.is(&PyTuple_Type)) {
        nb::tuple tuple = nb::borrow<nb::tuple>(h);
        for (uint32_t i = 0; i < tuple.size(); ++i) {
            transform_in_place(tuple[i], op);
        }
    } else if (tp.is(&PyList_Type)) {
        nb::list list = nb::borrow<nb::list>(h);
        for (uint32_t i = 0; i < list.size(); ++i) {
            transform_in_place(list[i], op);
        }
    } else if (tp.is(&PyDict_Type)) {
        nb::dict dict = nb::borrow<nb::dict>(h);
        for (auto v : dict.values()) {
            transform_in_place(v, op);
        }
    } else {
        if (nb::dict ds = get_drjit_struct(tp); ds.is_valid()) {
            for (auto k : ds.keys()) {
                transform_in_place(nb::getattr(h, k), op);
            }
        } else if (nb::object df = get_dataclass_fields(tp)) {
            for (nb::handle field : df) {
                nb::object k = field.attr(DR_STR(name));
                transform_in_place(nb::getattr(h, k), op);
            }
        } else if (nb::object cb = get_traverse_cb_rw(tp); cb.is_valid()) {
            // We want to transfer ownership, so we have to drop references
            // afterwards.
            // This is accomplished by storing them.
            
            // Scheduling the registry
            {
                uint32_t registry_bound = jit_registry_id_bound(JitBackend::None, nullptr);
                std::vector<void*> registry_pointers;
                registry_pointers.resize(registry_bound);
                jit_registry_fill_ptrs(registry_pointers.data());

                for (void *ptr : registry_pointers) {
                    if (!ptr)
                        continue;
                    
                    drjit::TraversableBase *traversable =
                        (drjit::TraversableBase *) ptr;

                    transform_in_place_traversable(traversable, op);
                }
            }
            
            
            
            std::vector<uint64_t> tmp;
            cb(h, nb::cpp_function([&](uint64_t index) {
                   if (!index)
                       return index;
                   uint64_t new_index = op(index);
                   tmp.push_back(new_index);
                   return new_index;
               }));
            for (uint64_t index : tmp) {
                ad_var_dec_ref(index);
            }
        } else {
        }
    }
}

static void deep_make_opaque(nb::handle h, bool eval = true) {
    jit_log(LogLevel::Debug, "make_opaque");

    struct ScheduleForceCallback : TransformInPlaceCallback {
        bool result = false;

        uint64_t operator()(uint64_t index) override {
            uint64_t new_index;
            if (ad_grad_enabled(index)) {

                uint32_t grad = ad_grad(index);

                int rv = 0;
                new_index = ad_var_schedule_force(index, &rv);
                if (rv){
                    jit_log(LogLevel::Debug,
                            "   scheduled ad-variable a%u, r%u -> a%u, r%u",
                            (uint32_t) (index >> 32), (uint32_t) index,
                            (uint32_t) (new_index >> 32), (uint32_t) new_index);
                    jit_log(LogLevel::Debug, "    state=%u", jit_var_state(new_index));
                    result = true;
                }

                rv = 0;
                uint32_t new_grad = jit_var_schedule_force(grad, &rv);
                jit_var_dec_ref(grad);
                if (rv){
                    jit_log(LogLevel::Debug,
                            "    scheduled gradient r%u -> r%u", grad,
                            new_grad);
                    jit_log(LogLevel::Debug, "    state=%u", jit_var_state(new_grad));
                    result = true;
                }

                ad_clear_grad(new_index);
                ad_accum_grad(new_index, new_grad);
                jit_var_dec_ref(new_grad);
            } else {
                int rv = 0;
                new_index = ad_var_schedule_force(index, &rv);
                if (rv){
                    jit_log(LogLevel::Debug,
                            "   scheduled variable r%u, label=%s -> r%u",
                            (uint32_t) index, jit_var_label(index),
                            (uint32_t) new_index);
                    result = true;
                }
            }


            return new_index;
        }
    };

    ScheduleForceCallback op;
    transform_in_place(h, op);

    if (op.result && eval) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
}

static void deep_eval(nb::handle h, bool eval = true) {
    jit_log(LogLevel::Debug, "deep eval");

    struct ScheduleCallback : TransformInPlaceCallback {
        bool result = false;

        uint64_t operator()(uint64_t index) override {
            if (ad_grad_enabled(index)) {
                int rv = 0;

                if (jit_var_schedule(index)){
                    jit_log(LogLevel::Debug,
                            "   scheduled ad-variable a%u, r%u, label=%s",
                            (uint32_t) (index >> 32), (uint32_t) index,
                            jit_var_label(index));
                    result = true;
                }

                uint32_t grad = ad_grad(index);
                if (jit_var_schedule(grad)){
                    jit_log(LogLevel::Debug, "    scheduled gradient r%u, label=%s",
                            grad, jit_var_label(grad));
                    result = true;
                }
                jit_var_dec_ref(grad);

            } else {
                int rv = jit_var_schedule(index);
                if (rv){
                    jit_log(LogLevel::Debug,
                            "   scheduled variable r%u, label=%s",
                            (uint32_t) index, jit_var_label(index));
                    result = true;
                }
            }
            ad_var_inc_ref(index);

            jit_log(LogLevel::Debug, "    scheduled a%u r%u",
                    (uint32_t) (index >> 32), (uint32_t) index);

            return index;
        }
    };

    ScheduleCallback op;
    transform_in_place(h, op);

    if (op.result && eval) {
        nb::gil_scoped_release guard;
        jit_eval();
    }
}

inline size_t py_object_hash(nb::handle h) {
    Py_hash_t hash = PyObject_Hash(h.ptr());
    if (hash == -1 && PyErr_Occurred())
        nb::raise_python_error();
    return (ssize_t)hash;
}

inline void hash_combine(size_t &seed, size_t value) {
    /// From CityHash (https://github.com/google/cityhash)
    const size_t mult = 0x9ddfea08eb382d69ull;
    size_t a = (value ^ seed) * mult;
    a ^= (a >> 47);
    size_t b = (seed ^ a) * mult;
    b ^= (b >> 47);
    seed = b * mult;
}

struct RecordingKey {
    std::vector<Layout> layout;
    uint32_t flags;

    RecordingKey() {
    }
    RecordingKey(std::vector<Layout> layout, uint32_t flags)
        : layout(std::move(layout)), flags(flags) {
    }

    bool operator==(const RecordingKey &rhs) const {
        return this->layout == rhs.layout && this->flags == rhs.flags;
    }

    void log_diff(const RecordingKey *rhs, LogLevel log_level) const {
        jit_log(log_level, "Key difference:");
        if (this->flags != rhs->flags)
            jit_log(log_level, "    flags: %u != %u", this->flags, rhs->flags);

        if (this->layout.size() != rhs->layout.size()) {
            jit_log(log_level, "    n_layout: %u != %u", this->layout.size(),
                    rhs->layout.size());
            return;
        }

        for (uint32_t i = 0; i < this->layout.size(); ++i) {
            const Layout &lhs_layout = this->layout[i];
            const Layout &rhs_layout = rhs->layout[i];

            // if (lhs_layout == rhs_layout)
            //     continue;

            jit_log(log_level, "    layout %u:", i);
            if (!lhs_layout.type.is_none() && !rhs_layout.type.is_none() &&
                !lhs_layout.type.equal(rhs_layout.type))
                jit_log(log_level, "    type: %s != %s",
                        nb::type_name(lhs_layout.type).c_str(),
                        nb::type_name(rhs_layout.type).c_str());
            if (lhs_layout.num != rhs_layout.num)
                jit_log(log_level, "    num: %u != %u", lhs_layout.num,
                        rhs_layout.num);
            if (lhs_layout.vt != rhs_layout.vt)
                jit_log(log_level, "    vt: %u != %u", lhs_layout.vt,
                        rhs_layout.vt);
            if (lhs_layout.vs != rhs_layout.vs)
                jit_log(log_level, "    vs: %u != %u", lhs_layout.vs,
                        rhs_layout.vs);
            if (lhs_layout.flags != rhs_layout.flags)
                jit_log(log_level, "    singleton_array: %u != %u",
                        lhs_layout.flags, rhs_layout.flags);
            if (lhs_layout.literal != rhs_layout.literal)
                jit_log(log_level, "    literal: %u != %u", lhs_layout.literal,
                        rhs_layout.literal);
            if (lhs_layout.index != rhs_layout.index)
                jit_log(log_level, "    index: %u != %u", lhs_layout.index,
                        rhs_layout.index);
            if (lhs_layout.size_index != rhs_layout.size_index)
                jit_log(log_level, "    size_index: %u != %u",
                        lhs_layout.size_index, rhs_layout.size_index);
            if (!(lhs_layout.py_object.equal(rhs_layout.py_object)))
                jit_log(log_level, "    py_object: %s != %s",
                        nb::str(lhs_layout.py_object).c_str(),
                        nb::str(rhs_layout.py_object).c_str());
        }
    }
    void log() {
        jit_log(LogLevel::Debug, "RecordingKey{");
        jit_log(LogLevel::Debug, "    flags = %u", this->flags);

        jit_log(LogLevel::Debug, "    layout = [");
        for (Layout &layout : layout) {
            jit_log(LogLevel::Debug, "        Layout{");
            if (!layout.type.is_none())
                jit_log(LogLevel::Debug, "            type = %s,",
                        nb::type_name(layout.type).c_str());
            jit_log(LogLevel::Debug, "            num = %u,", layout.num);
            jit_log(LogLevel::Debug, "            vt = %u,",
                    (uint32_t)layout.vt);
            jit_log(LogLevel::Debug, "            vs = %u,",
                    (uint32_t)layout.vs);
            jit_log(LogLevel::Debug, "            flags = %u,", layout.flags);
            jit_log(LogLevel::Debug, "        },");
        }
        jit_log(LogLevel::Debug, "    ]");

        jit_log(LogLevel::Debug, "}");
    }
};

struct RecordingKeyHasher {
    size_t operator()(const RecordingKey &key) const {
        // Hash the layout
        size_t hash = key.layout.size();
        for (const Layout &layout : key.layout) {
            hash_combine(hash, py_object_hash(layout.type));
            hash_combine(hash, layout.num);
            hash_combine(hash, layout.fields.size());
            for (auto &field : layout.fields) {
                hash_combine(hash, py_object_hash(field));
            }
            hash_combine(hash, (size_t)layout.vt);
            hash_combine(hash, (size_t)layout.vs);
            hash_combine(hash, (size_t)layout.flags);
            hash_combine(hash, (size_t)layout.literal);
            hash_combine(hash, (size_t)layout.index);
            hash_combine(hash, (size_t)layout.size_index);
            hash_combine(hash, py_object_hash(layout.py_object));
        }

        hash_combine(hash, (size_t)key.flags);

        return hash;
    }
};

struct FunctionRecording;

using RecordingMap =
    tsl::robin_map<RecordingKey, std::unique_ptr<FunctionRecording>,
                   RecordingKeyHasher>;

struct FrozenFunction {
    nb::callable func;

    RecordingMap recordings;
    RecordingKey prev_key;
    uint32_t recording_counter = 0;

    FrozenFunction(nb::callable func) : func(func) {
    }
    ~FrozenFunction() {
    }

    FrozenFunction(const FrozenFunction &) = delete;
    FrozenFunction &operator=(const FrozenFunction &) = delete;
    FrozenFunction(FrozenFunction &&) = default;
    FrozenFunction &operator=(FrozenFunction &&) = default;

    uint32_t saved_recordings() {
        return this->recordings.size();
    }
    
    nb::object operator()(nb::args args, nb::kwargs kwargs);
};

struct FunctionRecording {
    Recording *recording = nullptr;
    FlatVariables out_variables;

    FunctionRecording() : out_variables(false) {
    }
    FunctionRecording(const FunctionRecording &) = delete;
    FunctionRecording &operator=(const FunctionRecording &) = delete;
    FunctionRecording(FunctionRecording &&) = default;
    FunctionRecording &operator=(FunctionRecording &&) = default;

    ~FunctionRecording() {
        if (this->recording) {
            jit_freeze_destroy(this->recording);
        }
        this->recording = nullptr;
    }

    void clear() {
        if (this->recording) {
            jit_freeze_destroy(this->recording);
        }
        this->recording = nullptr;
        this->out_variables = FlatVariables(false);
    }

    /*
     * Record a function, given it's python input and flattened input.
     */
    nb::object record(nb::callable func, FrozenFunction *frozen_func,
                      nb::list input, const FlatVariables &in_variables) {
        ProfilerPhase profiler("record");
        JitBackend backend = in_variables.backend;
        frozen_func->recording_counter++;

        jit_log(LogLevel::Info,
                "Recording (n_inputs=%u):", in_variables.variables.size());
        jit_freeze_start(backend, in_variables.variables.data(),
                         in_variables.variables.size());

        // Record the function
        // bool tmp = jit_flag(JitFlag::KernelFreezing);
        jit_set_flag(JitFlag::KernelFreezing, false);
        nb::object output;
        {
            ProfilerPhase profiler("function");
            output = func(*input[0], **input[1]);
        }
        jit_set_flag(JitFlag::KernelFreezing, true);

        // output.append(result);
        // output.append(input);

        // Eval the input and output and it's gradients.
        jit_log(LogLevel::Debug, "Evaluating output:");
        {
            ProfilerPhase profiler("evaluate input + output");
            // Enter Resume scope, so we can track gradients
            ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1,
                                    false);
            {
                ProfilerPhase profiler("schedule input");
                deep_make_opaque(input, false);
            }
            {
                ProfilerPhase profiler("schedule output");
                deep_eval(output, false);
                // throw std::runtime_error("test");
            }
            {
                nb::gil_scoped_release guard;
                jit_eval();
            }
        }

        // Pause recording before traversal as to not accedentally record
        // unwanted operations.
        jit_freeze_pause(backend);

        // TODO: validate, that gradients wheren't enabled for inputs inside the
        // frozen function.

        // Collect nodes, that have been postponed by the `Isolate` scope in a
        // hash set.
        // These are the targets of postponed edges, as the isolate gradient
        // scope only handles backward mode differentiation.
        // If they are, then we have to enqueue them when replaying the
        // recording.
        tsl::robin_set<uint32_t, UInt32Hasher> postponed;
        {
            drjit::vector<uint32_t> postponed_vec;
            ad_scope_postponed(postponed_vec);
            for(uint32_t index : postponed_vec)
            postponed.insert(index);

        }
            

        jit_log(LogLevel::Info, "Traversing output");
        {
            ProfilerPhase profiler("traverse output");
            // Enter Resume scope, so we can track gradients
            ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1,
                                    false);
            
            TraverseContext ctx;
            ctx.postponed = &postponed;
            out_variables.traverse(output, ctx);
            out_variables.traverse_with_registry(input, ctx);
        }

        if ((out_variables.variables.size() > 0 &&
             in_variables.variables.size() > 0) &&
            out_variables.backend != backend) {
            Recording *recording = jit_freeze_stop(backend, nullptr, 0);
            jit_freeze_destroy(recording);

            nb::raise("freeze(): backend missmatch error (backend %u of "
                      "output "
                      "variables did not match backend %u of input "
                      "variables)",
                      (uint32_t)out_variables.backend, (uint32_t)backend);
        }

        recording = jit_freeze_stop(backend, out_variables.variables.data(),
                                    out_variables.variables.size());

        jit_log(LogLevel::Info, "Recording done (n_outputs=%u)",
                out_variables.variables.size());

        // For catching input assignment missmatches, we asign the input and output
        {
            // Enter Resume scope, so we can track gradients
            ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1,
                                    false);

            out_variables.layout_index = 0;
            jit_log(LogLevel::Debug, "Construct:");
            output = nb::borrow<nb::object>(out_variables.construct());
            // NOTE: temporarily disable this to not enqueue twice
            // jit_log(LogLevel::Debug, "Assign:");
            // out_variables.assign(input);
            out_variables.layout_index = 0;
        }

        return output;
    }
    /*
     * Replays the recording.
     *
     * This constructs the output and re-assigns the input.
     */
    nb::object replay(nb::callable func, FrozenFunction *frozen_func,
                      nb::list input, const FlatVariables &in_variables) {
        ProfilerPhase profiler("replay");

        jit_log(LogLevel::Info, "Replaying:");
        int dryrun_success;
        {
            ProfilerPhase profiler("dry run");
            dryrun_success =
                jit_freeze_dry_run(recording, in_variables.variables.data(),
                                    out_variables.variables.data());
        }
        if(!dryrun_success){
            // Dry run has failed. Re-record the function.
            jit_log(LogLevel::Warn, "re-recording");
            this->clear();
            try {
                return this->record(func, frozen_func, input, in_variables);
            } catch (nb::python_error &e) {
                nb::raise_from(e, PyExc_RuntimeError,
                               "replay(): error encountered while re-recording a "
                               "function (see above).");
            } catch (const std::exception &e) {
                jit_freeze_abort(in_variables.backend);

                nb::chain_error(PyExc_RuntimeError, "record(): %s", e.what());
                nb::raise_python_error();
            }
        }else{
            ProfilerPhase profiler("jit replay");
            nb::gil_scoped_release guard;
            jit_freeze_replay(recording, in_variables.variables.data(),
                              out_variables.variables.data());
        }
        jit_log(LogLevel::Info, "Replaying done:");

        // Construct Output variables
        nb::object output;
        {
            // Enter Resume scope, so we can track gradients
            ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, -1,
                                    false);
            out_variables.layout_index = 0;
            {
                ProfilerPhase profiler("construct output");
                output = nb::borrow<nb::object>(out_variables.construct());
            }
            {
                ProfilerPhase profiler("assign input");
                out_variables.assign_with_registry(input);
            }
        }

        // out_variables is assigned by jit_record_replay, which transfers
        // ownership to this array. Therefore, we have to drop the variables
        // afterwards.
        out_variables.release();

        return output;
    }
};

nb::object FrozenFunction::operator()(nb::args args, nb::kwargs kwargs) {
    nb::object result;
    {
        // Enter Isolate grad scope, so that gradients don't traverse
        // outside of the function scope.
        ADScopeContext ad_scope(drjit::ADScope::Isolate, 0, nullptr, -1, true);

        if (!jit_flag(JitFlag::KernelFreezing)) {
            ProfilerPhase profiler("function");
            return func(*args, **kwargs);
        }

        nb::list input;
        input.append(args);
        input.append(kwargs);

        FlatVariables in_variables(true);
        // Evaluate and traverse input variables (args and kwargs)
        {
            // Enter Resume scope, so we can track gradients
            ADScopeContext ad_scope(drjit::ADScope::Resume, 0, nullptr, 0,
                                    true);
            // Evaluate input variables, forcing evaluation of undefined
            // variables
            {
                ProfilerPhase profiler("evaluate input");
                deep_make_opaque(input);
            }

            // Traverse input variables
            ProfilerPhase profiler("traverse input");
            jit_log(LogLevel::Debug, "freeze(): Traversing input.");
            TraverseContext ctx;
            in_variables.traverse_with_registry(input, ctx);
        }

        raise_if(in_variables.backend == JitBackend::None,
                 "freeze(): Cannot infer backend without providing input "
                 "variable to frozen function!");

        uint32_t flags = jit_flags();
        auto key       = RecordingKey(std::move(in_variables.layout), flags);
        auto it        = this->recordings.find(key);

        if (it == this->recordings.end()) {
            if (this->recordings.size() >= 1) {
                jit_log(LogLevel::Info,
                        "Function input missmatch! Function will be retraced.");
                key.log_diff(&prev_key, LogLevel::Debug);
            }
            // FunctionRecording recording;
            auto recording = std::make_unique<FunctionRecording>();

            try {
                result = recording->record(func, this, input, in_variables);
            } catch (nb::python_error &e) {
                jit_log(LogLevel::Debug, "failed recording!");
                in_variables.release();
                jit_freeze_abort(in_variables.backend);
                jit_set_flag(JitFlag::KernelFreezing, true);
                nb::raise_from(e, PyExc_RuntimeError,
                               "record(): error encountered while recording a "
                               "function (see above).");
            } catch (const std::exception &e) {
                jit_log(LogLevel::Debug, "failed recording!");
                in_variables.release();
                jit_freeze_abort(in_variables.backend);
                jit_set_flag(JitFlag::KernelFreezing, true);

                nb::chain_error(PyExc_RuntimeError, "record(): %s", e.what());
                nb::raise_python_error();
            };

            in_variables.release();

            this->prev_key = key;
            this->recordings.insert({ std::move(key), std::move(recording) });

        } else {
            // Drop references to variables

            FunctionRecording *recording = it.value().get();

            { 
                result = recording->replay(func, this, input, in_variables); 
            }

            in_variables.release();
        }
    }
    // WARN: should track which variables where enqueud
    ad_traverse(drjit::ADMode::Backward,
                (uint32_t) drjit::ADFlag::ClearVertices);
    return result;
}

FrozenFunction freeze(nb::callable func) {
    return FrozenFunction(func);
}

void export_freeze(nb::module_ &m) {
    m.def("freeze", &freeze, doc_freeze);
    nb::class_<FrozenFunction>(m, "FrozenFunction")
        .def("__get__",
             [](nb::object self, nb::object instance, nb::object) {
                 if (instance.is_none()) {
                     return self;
                 } else {
                     return nb::cpp_function(
                         [self, instance](nb::args args, nb::kwargs kwargs) {
                             return self(instance, *args, **kwargs);
                         },
                         nb::rv_policy::move);
                 }
             })
        .def_prop_ro("n_cached_recordings",
                     [](FrozenFunction &self) { return self.saved_recordings(); })
        .def_ro("n_recordings", &FrozenFunction::recording_counter)
        .def("__call__", &FrozenFunction::operator());
}
