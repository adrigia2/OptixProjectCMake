#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "gdt/math/vec.h"

namespace pybind11 {
namespace detail {

template <>
struct type_caster<gdt::vec3f> {
public:
    PYBIND11_TYPE_CASTER(gdt::vec3f, const_name("vec3f"));

    // Python -> C++
    bool load(handle src, bool convert) {
        // 1. Already a wrapped vec3f
        if (isinstance<gdt::vec3f>(src)) {
            value = src.cast<gdt::vec3f>();
            return true;
        }

        // 2. numpy array (1-D, at least 3 elements)
        if (isinstance<array>(src)) {
            auto arr = reinterpret_borrow<array>(src);
            if (arr.ndim() != 1 || arr.shape(0) < 3)
                return false;
            try {
                auto arr_f = arr.cast<array_t<float, array::c_style | array::forcecast>>();
                auto buf = arr_f.unchecked<1>();
                value = gdt::vec3f(buf(0), buf(1), buf(2));
                return true;
            } catch (...) {
                return false;
            }
        }

        // 3. list or tuple of exactly 3 numeric elements
        if (isinstance<sequence>(src)) {
            auto seq = reinterpret_borrow<sequence>(src);
            if (seq.size() != 3)
                return false;
            try {
                value = gdt::vec3f(
                    seq[0].cast<float>(),
                    seq[1].cast<float>(),
                    seq[2].cast<float>()
                );
                return true;
            } catch (...) {
                return false;
            }
        }

        return false;
    }

    // C++ -> Python: return a registered vec3f wrapper object
    static handle cast(const gdt::vec3f& src, return_value_policy policy, handle parent) {
        (void)policy; (void)parent;
        return pybind11::cast(src, return_value_policy::copy).release();
    }
};

template <>
struct type_caster<gdt::vec2i> {
public:
    PYBIND11_TYPE_CASTER(gdt::vec2i, const_name("vec2i"));

    // Python -> C++
    bool load(handle src, bool convert) {
        // 1. Already a wrapped vec2i
        if (isinstance<gdt::vec2i>(src)) {
            value = src.cast<gdt::vec2i>();
            return true;
        }

        // 2. numpy array (1-D, at least 2 elements)
        if (isinstance<array>(src)) {
            auto arr = reinterpret_borrow<array>(src);
            if (arr.ndim() != 1 || arr.shape(0) < 2)
                return false;
            try {
                auto arr_i = arr.cast<array_t<int32_t, array::c_style | array::forcecast>>();
                auto buf = arr_i.unchecked<1>();
                value = gdt::vec2i(buf(0), buf(1));
                return true;
            } catch (...) {
                return false;
            }
        }

        // 3. list or tuple of exactly 2 numeric elements
        if (isinstance<sequence>(src)) {
            auto seq = reinterpret_borrow<sequence>(src);
            if (seq.size() != 2)
                return false;
            try {
                value = gdt::vec2i(
                    seq[0].cast<int32_t>(),
                    seq[1].cast<int32_t>()
                );
                return true;
            } catch (...) {
                return false;
            }
        }

        return false;
    }

    // C++ -> Python: return a registered vec2i wrapper object
    static handle cast(const gdt::vec2i& src, return_value_policy policy, handle parent) {
        (void)policy; (void)parent;
        return pybind11::cast(src, return_value_policy::copy).release();
    }
};

} // namespace detail
} // namespace pybind11
