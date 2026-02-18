// ======================================================================== //
// Camera Structure - Shared camera definition                             //
// ======================================================================== //

#pragma once

#include "gdt/math/vec.h"

namespace osc {

    using namespace gdt;

    /*! Camera structure for defining viewpoint and orientation */
    struct Camera {
        /*! camera position - *from* where we are looking */
        vec3f pos;
        /*! which direction we are looking (forward vector) */
        vec3f forward;
        /*! general up-vector */
        vec3f up;
    };

} // namespace osc
