#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

namespace osc {
    using namespace gdt;

    // Hemispherical visibility oracle, camera-agnostic.
    //
    // Unlike LaunchParams_SpecCone this pass knows about neither cameras nor rings:
    // for every texel it traces a FIXED Fibonacci set of num_samples rays uniform in
    // solid angle over the hemisphere above the normal, and returns only the t_hit of
    // each. Directions, envmap, NeRF query and per-camera classification all live on
    // the Python (torch) side, where the same ray serves every camera that sees the
    // texel.
    //
    // The mirror rays R_j = reflect(v_j, n) stay per-camera (level 0 is a delta
    // direction, which cannot be shared) and are launched in a second pass with
    // mode = MODE_MIRROR.
    struct LaunchParams_HemiVis
    {
        enum Mode { MODE_SHARED = 0, MODE_MIRROR = 1 };

        // IUM data (mirrors LaunchParams_SpecCone.ium_data)
        struct {
            vec3f*   ium_positions;
            vec3f*   ium_normals;
            uint8_t* ium_masks;
            int      num_pixels;
        } ium_data;

        // Which of the two passes is running (see Mode). The launch dimensions are
        // (tile_size, num_samples) for MODE_SHARED and (tile_size, num_cams) for
        // MODE_MIRROR.
        int mode;

        // Shared samples per texel (S). The sequence is deterministic and is rebuilt
        // identically on the Python side: cos(theta_s) = 1 - (s + 0.5)/S,
        // phi_s = s*goldenAngle + rot(global_idx), in Frisvad's ONB around n.
        int num_samples;

        // World positions of the cameras, for the mirror rays
        vec3f* cam_pos;          // [num_cams]
        int    num_cams;

        // Current tile
        int tile_offset;
        int tile_size;

        // Dense outputs, one value per ray, with no compaction and no counters:
        //   > 0  -> hit, distance to the surface (the NeRF window needs it)
        //   = 0  -> miss (sky)
        //   < 0  -> ray not launched (texel outside the mask, degenerate normal,
        //          or camera behind the surface in MODE_MIRROR)
        // Every thread always writes its own slot, so there is nothing to clear.
        float* t_hit_shared;     // [tile_size * num_samples]
        float* t_hit_mirror;     // [tile_size * num_cams]

        // Directions actually traced, indexed exactly like the t_hits.
        // nullptr = disabled: only the kernel<->torch parity test needs them
        // (test_hemivis_shared.py), which is the only way to notice the two formulas
        // diverging -- the production pass never transfers them.
        vec3f* dbg_dirs;         // [tile_size * max(num_samples, num_cams)]

        float epsilon;           // self-intersection offset along the normal

        OptixTraversableHandle traversable;
    };

} // namespace osc
