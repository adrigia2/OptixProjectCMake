#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SampleRenderer.h"

namespace py = pybind11;

PYBIND11_MODULE(depthMapModule, m) {
    m.doc() = "Python bindings for OptiX depth map renderer";

    // Binding minimo per vec3f (necessario per TriangleMesh)
    py::class_<gdt::vec3f>(m, "Vec3f")
        .def(py::init<float, float, float>());

    // Binding minimo per TriangleMesh (necessario per costruire SampleRenderer)
    py::class_<osc::TriangleMesh>(m, "TriangleMesh")
        .def(py::init<>())
        .def("addFromObjFile", &osc::TriangleMesh::addFromObjFile);

    // Binding per SampleRenderer con solo generateDepthMapsFromTransform
    py::class_<osc::SampleRenderer>(m, "SampleRenderer")
        .def(py::init<const osc::TriangleMesh&>())
        .def("generateDepthMapsFromTransform", 
             &osc::SampleRenderer::generateDepthMapsFromTransform,
             py::arg("transformFile"),
             py::arg("outputDir"),
             "Generate depth maps from transforms.json file");
}
