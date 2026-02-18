#define TINYOBJLOADER_IMPLEMENTATION

#include "TriangleMesh.h"
using namespace gdt;


//! add aligned cube with front-lower-left corner and size
void TriangleMesh::addCube(const vec3f& center, const vec3f& size)
{
    affine3f xfm;
    xfm.p = center - 0.5f * size;
    xfm.l.vx = vec3f(size.x, 0.f, 0.f);
    xfm.l.vy = vec3f(0.f, size.y, 0.f);
    xfm.l.vz = vec3f(0.f, 0.f, size.z);
    addUnitCube(xfm);
}

void TriangleMesh::addFromObjFile(const std::string& filename)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    /*
    bool LoadObj(attrib_t *attrib, std::vector<shape_t> *shapes,
           std::vector<material_t> *materials, std::string *err,
           const char *filename, const char *mtl_basedir = NULL,
           bool triangulate = true);*/

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());
    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to load/parse .obj file: " << filename << std::endl;
        if (!err.empty()) {
            std::cerr << "ERROR: " << err << std::endl;
        }
        throw std::runtime_error("Failed to load/parse .obj file");
        return;
    }

    bool hasTexCoords = !attrib.texcoords.empty();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            // Only process triangles
            if (fv != 3) {
                index_offset += fv;
                continue;
            }
            vec3i face_idx;
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                // Extract vertex position
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                vertex.push_back(vec3f(vx, vy, vz));

                // Extract texture coordinates if available
                if (hasTexCoords && idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                    texcoord.push_back(vec2f(tx, ty));
                }
                else {
                    // Default texture coordinates if not available
                    texcoord.push_back(vec2f(0.f, 0.f));
                }

                face_idx[v] = static_cast<int>(vertex.size() - 1);
            }
            index.push_back(face_idx);
            index_offset += fv;
        }
    }
}

/*! add a unit cube (subject to given xfm matrix) to the current
    triangleMesh */
void TriangleMesh::addUnitCube(const affine3f& xfm)
{
    int firstVertexID = (int)vertex.size();

    // Aggiungi i vertici del cubo
    vertex.push_back(xfmPoint(xfm, vec3f(0.f, 0.f, 0.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(1.f, 0.f, 0.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(0.f, 1.f, 0.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(1.f, 1.f, 0.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(0.f, 0.f, 1.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(1.f, 0.f, 1.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(0.f, 1.f, 1.f)));
    vertex.push_back(xfmPoint(xfm, vec3f(1.f, 1.f, 1.f)));

    // Aggiungi le coordinate texture per ogni vertice
    // Mapping UV semplice basato sulle coordinate x,y,z normalizzate
    texcoord.push_back(vec2f(0.f, 0.f)); // vertice 0
    texcoord.push_back(vec2f(1.f, 0.f)); // vertice 1
    texcoord.push_back(vec2f(0.f, 1.f)); // vertice 2
    texcoord.push_back(vec2f(1.f, 1.f)); // vertice 3
    texcoord.push_back(vec2f(0.f, 0.f)); // vertice 4
    texcoord.push_back(vec2f(1.f, 0.f)); // vertice 5
    texcoord.push_back(vec2f(0.f, 1.f)); // vertice 6
    texcoord.push_back(vec2f(1.f, 1.f)); // vertice 7

    // Definisci gli indici per i triangoli (12 triangoli, 2 per faccia)
    int indices[] = { 0,1,3, 2,0,3,
                     5,7,6, 5,6,4,
                     0,4,5, 0,5,1,
                     2,3,7, 2,7,6,
                     1,5,7, 1,7,3,
                     4,0,2, 4,2,6
    };

    for (int i = 0; i < 12; i++)
        index.push_back(firstVertexID + vec3i(indices[3 * i + 0],
            indices[3 * i + 1],
            indices[3 * i + 2]));
}