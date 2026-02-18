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
};