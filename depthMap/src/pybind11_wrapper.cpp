#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "TriangleMesh.h"
#include "OptixManager.h"
#include "IUM_Generator.h"
#include "Depth_Generator.h"
#include "ImageResultType.h"
#include "LogManager.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

// Classe wrapper per gestire il flusso completo di lavoro
class OptiXPipeline {

};

PYBIND11_MODULE(depthMapModule, m) {
	py::class_<TriangleMesh>(m, "TriangleMesh")
		.def(py::init<>())
		.def("add_from_obj_file",
			&TriangleMesh::addFromObjFile,
			py::arg("filename"));

	py::class_<Camera>(m, "Camera")
		.def(py::init<>())
		.def_readwrite("pos", &Camera::pos)
		.def_readwrite("forward", &Camera::forward)
		.def_readwrite("up", &Camera::up);

	py::class_<Depth_Generator>(m, "DepthGenerator")
		.def(py::init<>())
		.def("set_traversable", &Depth_Generator::setTraversable, py::arg("model"))
		.def("set_camera", &Depth_Generator::setCamera, py::arg("camera"), py::arg("fovY"), py::arg("frameSize"))
		.def("need_render_depth", &Depth_Generator::needRenderDepth, py::arg("isNeeded"))
		.def("need_render_position", &Depth_Generator::meedRenderPosition, py::arg("isNeeded"))
		.def("need_render_normal", &Depth_Generator::needRenderNormal, py::arg("isNeeded"))
		.def("render", &Depth_Generator::render)
		.def("get_result", &Depth_Generator::getResult);

	py::class_<IUM_Generator>(m, "IUMGenerator")
		.def(py::init<>())
		.def("set_traversable", &IUM_Generator::setTraversable, py::arg("model"))
		.def("set_texture_size", &IUM_Generator::setTextureSize, py::arg("size"))
		.def("render", &IUM_Generator::render)
		.def("get_result", &IUM_Generator::getResult);

	py::class_<OptixManager>(m, "OptixManager")
		.def_static("instance", &OptixManager::instance,
			py::return_value_policy::reference)
		.def("set_log_level", &OptixManager::setLogLevel, py::arg("level"));

	py::enum_<LogManager::Level>(m, "LogLevel")
		.value("Debug", LogManager::Level::Debug)
		.value("Info", LogManager::Level::Info)
		.value("Warning", LogManager::Level::Warning)
		.value("Error", LogManager::Level::Error)
		.value("Default", LogManager::Level::Default);

	py::class_<gdt::vec3f>(m, "vec3f")
		.def(py::init<>())
		.def(py::init<float, float, float>())
		.def_readwrite("x", &gdt::vec3f::x)
		.def_readwrite("y", &gdt::vec3f::y)
		.def_readwrite("z", &gdt::vec3f::z);

	py::class_<IUM_Generator::Result>(m, "IUMResult")
		.def("has_positions", &IUM_Generator::Result::hasPositions)
		.def("has_masks", &IUM_Generator::Result::hasMasks)

		.def_property_readonly("positions_np", [](IUM_Generator::Result& r) {
		using gdt::vec3f;
		py::ssize_t n = static_cast<py::ssize_t>(r.positions.size());

		// base: tiene vivo r finché l'array vive
		py::object base = py::cast(&r);

		// view zero-copy: shape (n,3), strides (sizeof(vec3f), sizeof(float))
		return py::array_t<float>(
			{ n, py::ssize_t(3) },
			{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
			reinterpret_cast<float*>(r.positions.data()),
			base
		);
			})
		.def_property_readonly("masks_np", [](IUM_Generator::Result& r) {
		py::ssize_t n = static_cast<py::ssize_t>(r.masks.size());
		py::object base = py::cast(&r);

		return py::array_t<uint8_t>(
			{ n },
			{ py::ssize_t(sizeof(uint8_t)) },
			r.masks.data(),
			base
		);
			});



	py::class_<Depth_Generator::Result>(m, "DepthResult")
		.def(py::init<>()) // opzionale
		.def("has_depth_data", &Depth_Generator::Result::hasDepthData)
		.def("has_positional_data", &Depth_Generator::Result::hasPositionalData)
		.def("has_normal_data", &Depth_Generator::Result::hasNormalData)

		// --- NumPy views ---
		.def_property_readonly("depths_np", [](Depth_Generator::Result& r) {
		py::ssize_t n = (py::ssize_t)r.depthData.size();
		py::object base = py::cast(&r);
		return py::array_t<float>(
			{ n },
			{ py::ssize_t(sizeof(float)) },
			r.depthData.data(),
			base
		);
			})
		.def_property_readonly("positions_np", [](Depth_Generator::Result& r) {
		using gdt::vec3f;
		py::ssize_t n = (py::ssize_t)r.positionalData.size();
		py::object base = py::cast(&r);
		return py::array_t<float>(
			{ n, py::ssize_t(3) },
			{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
			reinterpret_cast<float*>(r.positionalData.data()),
			base
		);
			})
		.def_property_readonly("normals_np", [](Depth_Generator::Result& r) {
		using gdt::vec3f;
		py::ssize_t n = (py::ssize_t)r.normalData.size();
		py::object base = py::cast(&r);
		return py::array_t<float>(
			{ n, py::ssize_t(3) },
			{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
			reinterpret_cast<float*>(r.normalData.data()),
			base
		);
			})
		.def_property_readonly("masks_np", [](Depth_Generator::Result& r) {
		py::ssize_t n = (py::ssize_t)r.maskData.size();
		py::object base = py::cast(&r);
		return py::array_t<uint8_t>(
			{ n },
			{ py::ssize_t(sizeof(uint8_t)) },
			r.maskData.data(),
			base
		);
			});
}
