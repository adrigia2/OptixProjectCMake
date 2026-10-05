#pragma once

#include "gdt/math/vec.h"
#include "gdt/math/AffineSpace.h"
#include "objLoader/tiny_obj_loader.h"

using namespace gdt;

struct TriangleMesh {
    /*! add a unit cube (subject to given xfm matrix) to the current
        triangleMesh */
    void addUnitCube(const affine3f& xfm);

    //! add aligned cube aith front-lower-left corner and size
    void addCube(const vec3f& center, const vec3f& size);
    void addFromObjFile(const std::string& filename);

    std::vector<vec3f> vertex;
    std::vector<vec3i> index;
    std::vector<vec2f> texcoord;
    // One per entry of `vertex` (the vertices are unrolled, one per triangle corner):
    // the OBJ `vn` of that corner, normalised, or (0,0,0) where the corner has none.
    // Empty when the file carries no `vn` at all. Blender writes one `vn` per corner
    // that already encodes smooth shading, flat faces and sharp edges, so
    // interpolating these reproduces its shading normal without reading `s` groups.
    std::vector<vec3f> normal;
    bool hasNormals() const { return !normal.empty() && normal.size() == vertex.size(); }
};