#include <pybind11/pybind11.h>
#include <pybind11/native_enum.h> // Not already included with pybind11.h
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "vec3f_caster.h"
#include "TriangleMesh.h"
#include "OptixManager.h"
#include "IUM_Generator.h"
#include "Depth_Generator.h"
#include "Visibility_Generator.h"
#include "ColorTex_Generator.h"
#include "Frame.h"
#include "ImageResultType.h"
#include "LogManager.h"

namespace py = pybind11;

// Classe wrapper per gestire il flusso completo di lavoro
class OptiXPipeline {

};

PYBIND11_MODULE(OptixProgrammablePasses, m, py::mod_gil_not_used()) {
	m.doc() = "Module for generating stuff with OptiX";

	// --- Primitive types first: used in signatures of classes below ---

	py::native_enum<LogManager::Level>(m, "LogLevel", "enum.IntEnum")
		.value("Disabled", LogManager::Level::Disabled)
		.value("Fatal", LogManager::Level::Fatal)
		.value("Error", LogManager::Level::Error)
		.value("Warning", LogManager::Level::Warning)
		.value("Verbose", LogManager::Level::Verbose)
		.export_values()
		.finalize();

	py::class_<gdt::vec2i>(m, "vec2i")
		.def(py::init<>())
		.def(py::init<int32_t, int32_t>())
		.def(py::init([](py::sequence s) {
			if (s.size() != 2) throw py::value_error("vec2i requires exactly 2 elements");
			return gdt::vec2i(s[0].cast<int32_t>(), s[1].cast<int32_t>());
		}))
		.def_readwrite("x", &gdt::vec2i::x)
		.def_readwrite("y", &gdt::vec2i::y)
		.def("__repr__", [](const gdt::vec2i& v) {
			return "<vec2i x=" + std::to_string(v.x) + " y=" + std::to_string(v.y) + ">";
		})
		.def("__iter__", [](const gdt::vec2i& v) {
			return py::iter(py::make_tuple(v.x, v.y));
		});

	py::class_<gdt::vec3f>(m, "vec3f")
		.def(py::init<>())
		.def(py::init<float, float, float>())
		.def(py::init([](py::sequence s) {
			if (s.size() != 3) throw py::value_error("vec3f requires exactly 3 elements");
			return gdt::vec3f(s[0].cast<float>(), s[1].cast<float>(), s[2].cast<float>());
		}))
		.def_readwrite("x", &gdt::vec3f::x)
		.def_readwrite("y", &gdt::vec3f::y)
		.def_readwrite("z", &gdt::vec3f::z)
		.def("__repr__", [](const gdt::vec3f& v) {
			return "<vec3f x=" + std::to_string(v.x) + " y=" + std::to_string(v.y) + " z=" + std::to_string(v.z) + ">";
		})
		.def("__iter__", [](const gdt::vec3f& v) {
			return py::iter(py::make_tuple(v.x, v.y, v.z));
		});

	py::implicitly_convertible<py::sequence, gdt::vec2i>();
	py::implicitly_convertible<py::sequence, gdt::vec3f>();

	// --- Result types: must be registered before the generators that return them ---

	py::class_<Depth_Generator::Result>(m, "DepthResult")
		.def(py::init<>())
		.def("has_depth_data", &Depth_Generator::Result::hasDepthData)
		.def("has_positional_data", &Depth_Generator::Result::hasPositionalData)
		.def("has_normal_data", &Depth_Generator::Result::hasNormalData)
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

	py::class_<IUM_Generator::Result>(m, "IUMResult")
		.def("has_positions", &IUM_Generator::Result::hasPositions)
		.def("has_masks", &IUM_Generator::Result::hasMasks)
		.def_property_readonly("positions_np", [](IUM_Generator::Result& r) {
			using gdt::vec3f;
			py::ssize_t n = static_cast<py::ssize_t>(r.positions.size());
			py::object base = py::cast(&r);
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

	// --- Main classes ---

	py::class_<TriangleMesh>(m, "TriangleMesh")
		.def(py::init<>())
		.def("add_from_obj_file",
			&TriangleMesh::addFromObjFile,
			py::arg("filename"));

	py::class_<Camera>(m, "Camera")
		.def(py::init([](const gdt::vec3f& pos, const gdt::vec3f& forward, const gdt::vec3f& up,
			float fovY, const gdt::vec2i& frameSize) {
			return Camera(pos, forward, up, fovY, frameSize);
		}),
			py::arg("pos"), py::arg("forward"), py::arg("up"),
			py::arg("fovY") = 45.0f, py::arg("frameSize") = gdt::vec2i(1024, 1024))
		.def_property("pos", &Camera::getPos, &Camera::setPos)
		.def_property("forward", &Camera::getForward, &Camera::setForward)
		.def_property("up", &Camera::getUp, &Camera::setUp)
		.def_property("fovY", &Camera::getFovY, &Camera::setFovY)
		.def_property("frame_size", &Camera::getFrameSize, &Camera::setFrameSize)
		.def("__repr__", [](const Camera& cam) {
			return "<Camera pos=" + std::to_string(cam.getPos().x) + "," + std::to_string(cam.getPos().y) + "," + std::to_string(cam.getPos().z) +
				" forward=" + std::to_string(cam.getForward().x) + "," + std::to_string(cam.getForward().y) + "," + std::to_string(cam.getForward().z) +
				" up=" + std::to_string(cam.getUp().x) + "," + std::to_string(cam.getUp().y) + "," + std::to_string(cam.getUp().z) +
				" fovY=" + std::to_string(cam.getFovY()) +
				" frameSize=" + std::to_string(cam.getFrameSize().x) + "x" + std::to_string(cam.getFrameSize().y) + ">";
		});

	py::class_<Depth_Generator>(m, "DepthGenerator")
		.def(py::init<>())
		.def("set_traversable", &Depth_Generator::setTraversable, py::arg("model"))
		.def("set_camera", &Depth_Generator::setCamera, py::arg("camera"))
		.def("need_render_depth", &Depth_Generator::needRenderDepth, py::arg("isNeeded"))
		.def("need_render_position", &Depth_Generator::meedRenderPosition, py::arg("isNeeded"))
		.def("need_render_normal", &Depth_Generator::needRenderNormal, py::arg("isNeeded"))
		.def("render", &Depth_Generator::render)
		.def("get_result", [](Depth_Generator& self) -> Depth_Generator::Result {
			return self.getResult();
		}, py::return_value_policy::move);

	py::class_<IUM_Generator>(m, "IUMGenerator")
		.def(py::init<>())
		.def("set_traversable", &IUM_Generator::setTraversable, py::arg("model"))
		.def("set_texture_size", [](IUM_Generator& self, gdt::vec2i size) {
			self.setTextureSize(size);
		}, py::arg("size"))
		.def("render", &IUM_Generator::render)
		.def("get_result", [](IUM_Generator& self) -> IUM_Generator::Result {
			return self.getResult();
		}, py::return_value_policy::move);

	py::class_<Visibility_Generator>(m, "VisibilityGenerator")
		.def(py::init<>())
		.def("set_traversable", &Visibility_Generator::setTraversable, py::arg("model"))
		.def("check_visibility", [](Visibility_Generator& self, const IUM_Generator::Result& ium_res, int width, int height, py::list cameras_list) {
			std::vector<Camera> camDefs;
			for (auto item : cameras_list) {
				Camera cam = item.cast<Camera>();
				camDefs.push_back(cam);
			}

            std::vector<uint8_t> h_results = self.checkVisibility(ium_res, width, height, camDefs);
			
			py::ssize_t num_pixels = width * height;
			py::ssize_t num_cameras = camDefs.size();
			
			// Restituiamo una py::array_t di shape (num_pixels, num_cameras)
			return py::array_t<uint8_t>(
				{ num_pixels, num_cameras },
				{ py::ssize_t(sizeof(uint8_t) * num_cameras), py::ssize_t(sizeof(uint8_t)) },
				h_results.data()
			);
		}, py::arg("ium_result"), py::arg("width"), py::arg("height"), py::arg("cameras"));

	// --- Frame ---

	py::class_<osc::Frame>(m, "Frame")
		.def(py::init([](const osc::Camera& cam, float peak, py::array_t<float, py::array::c_style | py::array::forcecast> img) {
			py::buffer_info buf = img.request();
			if (buf.ndim != 2 || buf.shape[1] != 3)
				throw py::value_error("Frame: image must be a (N, 3) float32 array");
			osc::Frame f{ cam, peak, {} };
			f.image.resize(buf.shape[0]);
			const float* ptr = static_cast<const float*>(buf.ptr);
			for (py::ssize_t i = 0; i < buf.shape[0]; ++i)
				f.image[i] = gdt::vec3f(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]);
			return f;
		}), py::arg("camera"), py::arg("peak"), py::arg("image"))
		.def_readwrite("camera", &osc::Frame::camera)
		.def_readwrite("peak",   &osc::Frame::peak);

	// --- ColorTexResult ---

	py::class_<osc::ColorTex_Generator::Result>(m, "ColorTexResult")
		.def_property_readonly("colors_np", [](osc::ColorTex_Generator::Result& r) {
			using gdt::vec3f;
			py::ssize_t n = static_cast<py::ssize_t>(r.colors.size());
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, py::ssize_t(3) },
				{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.colors.data()),
				base
			);
		})
		.def_property_readonly("camera_colors_np", [](osc::ColorTex_Generator::Result& r) {
			using gdt::vec3f;
			py::ssize_t n  = static_cast<py::ssize_t>(r.num_cameras > 0
			                   ? r.camera_colors.size() / r.num_cameras : 0);
			py::ssize_t nc = static_cast<py::ssize_t>(r.num_cameras);
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, nc, py::ssize_t(3) },
				{ py::ssize_t(sizeof(vec3f)) * nc,
				  py::ssize_t(sizeof(vec3f)),
				  py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.camera_colors.data()),
				base
			);
		})
		.def_property_readonly("color_min_np", [](osc::ColorTex_Generator::Result& r) {
			using gdt::vec3f;
			py::ssize_t n = static_cast<py::ssize_t>(r.color_min.size());
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, py::ssize_t(3) },
				{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<const float*>(r.color_min.data()),
				base
			);
		})
		.def_property_readonly("color_max_np", [](osc::ColorTex_Generator::Result& r) {
			using gdt::vec3f;
			py::ssize_t n = static_cast<py::ssize_t>(r.color_max.size());
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, py::ssize_t(3) },
				{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<const float*>(r.color_max.data()),
				base
			);
		})
		.def_property_readonly("color_variance_np", [](osc::ColorTex_Generator::Result& r) {
			using gdt::vec3f;
			py::ssize_t n = static_cast<py::ssize_t>(r.color_variance.size());
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, py::ssize_t(3) },
				{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<const float*>(r.color_variance.data()),
				base
			);
		});

	// --- ColorTexGenerator ---

	py::class_<osc::ColorTex_Generator>(m, "ColorTexGenerator")
		.def(py::init<>())
		.def("set_inputs", [](osc::ColorTex_Generator& self,
				const IUM_Generator::Result& ium_res,
				py::array_t<uint8_t, py::array::c_style | py::array::forcecast> visibility,
				py::list frames_list) {
			py::buffer_info vbuf = visibility.request();
			std::vector<uint8_t> vis_vec(
				static_cast<uint8_t*>(vbuf.ptr),
				static_cast<uint8_t*>(vbuf.ptr) + vbuf.size);

			std::vector<osc::Frame> frames;
			frames.reserve(frames_list.size());
			for (auto item : frames_list)
				frames.push_back(item.cast<osc::Frame>());

			self.setInputs(ium_res, vis_vec, frames);
		}, py::arg("ium_result"), py::arg("visibility"), py::arg("frames"))
		.def("render", &osc::ColorTex_Generator::render)
		.def("get_result", [](osc::ColorTex_Generator& self) -> osc::ColorTex_Generator::Result {
			return self.getResult();
		}, py::return_value_policy::move);

	py::class_<OptixManager>(m, "OptixManager")
		.def_static("instance", &OptixManager::instance,
			py::return_value_policy::reference)
		.def("set_log_level", [](OptixManager& self, LogManager::Level level) {
			self.setLogLevel(level);
		}, py::arg("level"),
			"Set the log level for both OptiX callbacks and LogManager filter.");

	py::class_<LogManager>(m, "LogManager")
		.def_static("log", [](const std::string& msg) { LogManager::Log("%s", msg.c_str()); },
			py::arg("message"))
		.def_static("log_error", [](const std::string& msg) { LogManager::LogError("%s", msg.c_str()); },
			py::arg("message"))
		.def_static("log_warning", [](const std::string& msg) { LogManager::LogWarning("%s", msg.c_str()); },
			py::arg("message"))
		.def_static("log_fatal", [](const std::string& msg) { LogManager::LogFatal("%s", msg.c_str()); },
			py::arg("message"))
		.def_static("set_log_file", &LogManager::SetLogFile,
			py::arg("path"), py::arg("append") = true)
		.def_static("enable_console", &LogManager::EnableConsole,
			py::arg("enabled"))
		.def_static("enable_file", &LogManager::EnableFile,
			py::arg("enabled"))
		.def_static("set_min_level", &LogManager::SetMinLevel,
			py::arg("level"));
}
