#pragma once

#include "Camera.h"
#include "gdt/math/vec.h"
#include <vector>

namespace osc {
    using namespace gdt;

    struct Frame {
        Camera             camera;
        float              peak;
        std::vector<vec3f> image; // flat: size = frameSize.x * frameSize.y
    };

} // namespace osc
