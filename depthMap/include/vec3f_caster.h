#pragma once

// This header intentionally left minimal.
// vec3f and vec2i are registered as py::class_ in pybind11_wrapper.cpp.
// Implicit conversions from sequences/tuples are handled by py::init
// overloads registered there. No custom type_caster is needed.
