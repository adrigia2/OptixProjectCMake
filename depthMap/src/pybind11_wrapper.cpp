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
#include "Irradiance_Generator.h"
#include "Indirect_Generator.h"
#include "SpecCone_Generator.h"
#include "HemiVis_Generator.h"
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
		.def("has_normals", &IUM_Generator::Result::hasNormals)
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
		.def_property_readonly("normals_np", [](IUM_Generator::Result& r) {
			using gdt::vec3f;
			py::ssize_t n = static_cast<py::ssize_t>(r.normals.size());
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, py::ssize_t(3) },
				{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.normals.data()),
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
		.def("need_render_position", &Depth_Generator::needRenderPosition, py::arg("isNeeded"))
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
			
			// Return a py::array_t of shape (num_pixels, num_cameras)
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
				py::list frames_list,
				float grazing_max_deg) {
			py::buffer_info vbuf = visibility.request();
			std::vector<uint8_t> vis_vec(
				static_cast<uint8_t*>(vbuf.ptr),
				static_cast<uint8_t*>(vbuf.ptr) + vbuf.size);

			std::vector<osc::Frame> frames;
			frames.reserve(frames_list.size());
			for (auto item : frames_list)
				frames.push_back(item.cast<osc::Frame>());

			self.setInputs(ium_res, vis_vec, frames, grazing_max_deg);
		}, py::arg("ium_result"), py::arg("visibility"), py::arg("frames"),
		   py::arg("grazing_max_deg") = 90.f)
		.def("render", &osc::ColorTex_Generator::render)
		// Per-camera slice, downloaded from the GPU on demand (the full host mirror,
		// num_pixels x num_cameras, does not scale); the download writes straight into
		// the buffer of the returned numpy array, with no intermediate copy.
		.def("download_camera_colors", [](osc::ColorTex_Generator& self, int cam) {
			py::ssize_t n = static_cast<py::ssize_t>(self.numPixels());
			py::array_t<float> out({ n, py::ssize_t(3) });
			self.downloadCameraColors(
				cam, reinterpret_cast<gdt::vec3f*>(out.mutable_data()));
			return out;
		}, py::arg("cam"))
		// Per-camera mask (uint8, shape (num_pixels,)): 1 = the camera contributes
		// (unoccluded AND in frustum AND not grazing; pre-peak, so source-independent).
		.def("download_camera_mask", [](osc::ColorTex_Generator& self, int cam) {
			py::ssize_t n = static_cast<py::ssize_t>(self.numPixels());
			py::array_t<uint8_t> out(n);
			self.downloadCameraMask(cam, out.mutable_data());
			return out;
		}, py::arg("cam"))
		.def("get_result", [](osc::ColorTex_Generator& self) -> osc::ColorTex_Generator::Result {
			return self.getResult();
		}, py::return_value_policy::move);

	// --- IrradianceResult / IrradianceGenerator ---

	py::class_<osc::Irradiance_Generator::Result>(m, "IrradianceResult")
		.def("has_irradiance", &osc::Irradiance_Generator::Result::hasIrradiance)
		.def_property_readonly("irradiance_np", [](osc::Irradiance_Generator::Result& r) {
			using gdt::vec3f;
			py::ssize_t n = static_cast<py::ssize_t>(r.irradiance.size());
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, py::ssize_t(3) },
				{ py::ssize_t(sizeof(vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.irradiance.data()),
				base
			);
		});

	py::class_<osc::Irradiance_Generator>(m, "IrradianceGenerator")
		.def(py::init<>())
		.def("set_traversable", &osc::Irradiance_Generator::setTraversable, py::arg("model"))
		.def("set_inputs", [](osc::Irradiance_Generator& self,
				const IUM_Generator::Result& ium_res,
				py::array_t<float, py::array::c_style | py::array::forcecast> skybox,
				gdt::vec2i skybox_size,
				int sample_side,
				float skybox_yaw_degrees) {
			py::buffer_info b = skybox.request();
			if (b.ndim != 2 || b.shape[1] != 3)
				throw py::value_error("skybox must be a (H*W, 3) float32 array");

			std::vector<gdt::vec3f> sky(b.shape[0]);
			const float* p = static_cast<const float*>(b.ptr);
			for (py::ssize_t i = 0; i < b.shape[0]; ++i)
				sky[i] = gdt::vec3f(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);

			self.setInputs(ium_res, sky, skybox_size, sample_side, skybox_yaw_degrees);
		}, py::arg("ium_result"), py::arg("skybox"), py::arg("skybox_size"),
		   py::arg("sample_side"), py::arg("skybox_yaw_degrees") = 0.0f)
		.def("render", &osc::Irradiance_Generator::render)
		.def("get_result", [](osc::Irradiance_Generator& self) -> osc::Irradiance_Generator::Result {
			return self.getResult();
		}, py::return_value_policy::move);

	// --- IndirectTileResult / IndirectGenerator ---

	py::class_<osc::Indirect_Generator::TileResult>(m, "IndirectTileResult")
		.def_readonly("count", &osc::Indirect_Generator::TileResult::count)
		.def_property_readonly("directions_np", [](osc::Indirect_Generator::TileResult& r) {
			py::ssize_t n = static_cast<py::ssize_t>(r.dirs.size());
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, py::ssize_t(3) },
				{ py::ssize_t(sizeof(gdt::vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.dirs.data()),
				base
			);
		})
		.def_property_readonly("cos_np", [](osc::Indirect_Generator::TileResult& r) {
			py::ssize_t n = static_cast<py::ssize_t>(r.cos_weights.size());
			py::object base = py::cast(&r);
			return py::array_t<float>({ n }, { sizeof(float) },
				r.cos_weights.data(), base);
		})
		.def_property_readonly("t_hit_np", [](osc::Indirect_Generator::TileResult& r) {
			py::ssize_t n = static_cast<py::ssize_t>(r.t_hits.size());
			py::object base = py::cast(&r);
			return py::array_t<float>({ n }, { sizeof(float) },
				r.t_hits.data(), base);
		})
		.def_property_readonly("local_idx_np", [](osc::Indirect_Generator::TileResult& r) {
			py::ssize_t n = static_cast<py::ssize_t>(r.local_indices.size());
			py::object base = py::cast(&r);
			return py::array_t<int>({ n }, { sizeof(int) },
				r.local_indices.data(), base);
		});

	py::class_<osc::Indirect_Generator>(m, "IndirectGenerator")
		.def(py::init<>())
		.def("set_traversable", &osc::Indirect_Generator::setTraversable, py::arg("model"))
		.def("set_inputs", [](osc::Indirect_Generator& self,
				const IUM_Generator::Result& ium_res,
				int sample_side,
				int tile_size) {
			self.setInputs(ium_res, sample_side, tile_size);
		}, py::arg("ium_result"), py::arg("sample_side"), py::arg("tile_size") = 1024)
		.def("num_tiles", &osc::Indirect_Generator::numTiles)
		.def("tile_size", &osc::Indirect_Generator::tileSize)
		.def("num_pixels", &osc::Indirect_Generator::numPixels)
		.def("render_tile", [](osc::Indirect_Generator& self, int tile_idx)
				-> osc::Indirect_Generator::TileResult {
			return self.renderTile(tile_idx);
		}, py::arg("tile_idx"), py::return_value_policy::move);

	// --- SpecConeTileResult / SpecConeGenerator ---

	py::class_<osc::SpecCone_Generator::TileResult>(m, "SpecConeTileResult")
		.def_readonly("count", &osc::SpecCone_Generator::TileResult::count)
		.def_readonly("tile_texels", &osc::SpecCone_Generator::TileResult::tile_texels)
		.def_readonly("num_levels", &osc::SpecCone_Generator::TileResult::num_levels)
		.def_readonly("overflow", &osc::SpecCone_Generator::TileResult::overflow)
		.def_readonly("requested", &osc::SpecCone_Generator::TileResult::requested)
		.def_property_readonly("directions_np", [](osc::SpecCone_Generator::TileResult& r) {
			py::ssize_t n = static_cast<py::ssize_t>(r.dirs.size());
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ n, py::ssize_t(3) },
				{ py::ssize_t(sizeof(gdt::vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.dirs.data()),
				base
			);
		})
		.def_property_readonly("t_hit_np", [](osc::SpecCone_Generator::TileResult& r) {
			py::ssize_t n = static_cast<py::ssize_t>(r.t_hits.size());
			py::object base = py::cast(&r);
			return py::array_t<float>({ n }, { sizeof(float) },
				r.t_hits.data(), base);
		})
		.def_property_readonly("local_idx_np", [](osc::SpecCone_Generator::TileResult& r) {
			py::ssize_t n = static_cast<py::ssize_t>(r.local_indices.size());
			py::object base = py::cast(&r);
			return py::array_t<int>({ n }, { sizeof(int) },
				r.local_indices.data(), base);
		})
		.def_property_readonly("ring_idx_np", [](osc::SpecCone_Generator::TileResult& r) {
			py::ssize_t n = static_cast<py::ssize_t>(r.ring_indices.size());
			py::object base = py::cast(&r);
			return py::array_t<int>({ n }, { sizeof(int) },
				r.ring_indices.data(), base);
		})
		.def_property_readonly("sky_sum_np", [](osc::SpecCone_Generator::TileResult& r) {
			py::ssize_t t = static_cast<py::ssize_t>(r.tile_texels);
			py::ssize_t l = static_cast<py::ssize_t>(r.num_levels);
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ t, l, py::ssize_t(3) },
				{ py::ssize_t(l * sizeof(gdt::vec3f)), py::ssize_t(sizeof(gdt::vec3f)), py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.sky_sum.data()),
				base
			);
		})
		.def_property_readonly("valid_count_np", [](osc::SpecCone_Generator::TileResult& r) {
			py::ssize_t t = static_cast<py::ssize_t>(r.tile_texels);
			py::ssize_t l = static_cast<py::ssize_t>(r.num_levels);
			py::object base = py::cast(&r);
			return py::array_t<int>(
				{ t, l },
				{ py::ssize_t(l * sizeof(int)), py::ssize_t(sizeof(int)) },
				r.valid_count.data(),
				base
			);
		});

	py::class_<osc::SpecCone_Generator>(m, "SpecConeGenerator")
		.def(py::init<>())
		.def("set_traversable", &osc::SpecCone_Generator::setTraversable, py::arg("model"))
		.def("set_inputs", [](osc::SpecCone_Generator& self,
				const IUM_Generator::Result& ium_res,
				const std::vector<float>& cone_apertures_deg,
				py::object samples_per_ring,
				int tile_size) {
			if (cone_apertures_deg.size() < 2)
				throw py::value_error("cone_apertures_deg needs at least 2 values");
			const size_t n_rings = cone_apertures_deg.size() - 1;

			// A scalar (numpy.int64 included) -> uniform; a sequence -> one per ring.
			std::vector<int> spr;
			try {
				spr.assign(n_rings, samples_per_ring.cast<int>());
			} catch (const py::cast_error&) {
				try {
					spr = samples_per_ring.cast<std::vector<int>>();
				} catch (const py::cast_error&) {
					throw py::type_error(
						"samples_per_ring must be an int (same count on every "
						"ring) or a sequence of int of length "
						"len(cone_apertures_deg)-1");
				}
				if (spr.size() != n_rings)
					throw py::value_error(
						"samples_per_ring: expected " + std::to_string(n_rings) +
						" values (len(cone_apertures_deg)-1), got " +
						std::to_string(spr.size()));
			}
			self.setInputs(ium_res, cone_apertures_deg, spr, tile_size);
		}, py::arg("ium_result"), py::arg("cone_apertures_deg"),
		   py::arg("samples_per_ring"), py::arg("tile_size") = 1024,
		   "samples_per_ring: int (same count on every ring) or list[int] of "
		   "length len(cone_apertures_deg)-1 (rings 1..K-1; level 0 = mirror "
		   "ray is always a single sample).")
		.def("set_envmap", [](osc::SpecCone_Generator& self,
				py::array_t<float, py::array::c_style | py::array::forcecast> skybox,
				gdt::vec2i skybox_size,
				float skybox_yaw_degrees) {
			py::buffer_info b = skybox.request();
			if (b.ndim != 2 || b.shape[1] != 3)
				throw py::value_error("skybox must be a (H*W, 3) float32 array");

			std::vector<gdt::vec3f> sky(b.shape[0]);
			const float* p = static_cast<const float*>(b.ptr);
			for (py::ssize_t i = 0; i < b.shape[0]; ++i)
				sky[i] = gdt::vec3f(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);

			self.setEnvmap(sky, skybox_size, skybox_yaw_degrees);
		}, py::arg("skybox"), py::arg("skybox_size"), py::arg("skybox_yaw_degrees") = 0.0f)
		.def("set_camera", [](osc::SpecCone_Generator& self,
				const gdt::vec3f& cam_pos,
				py::array_t<uint8_t, py::array::c_style | py::array::forcecast> visibility) {
			py::buffer_info vbuf = visibility.request();
			std::vector<uint8_t> vis_vec(
				static_cast<uint8_t*>(vbuf.ptr),
				static_cast<uint8_t*>(vbuf.ptr) + vbuf.size);
			self.setCamera(cam_pos, vis_vec);
		}, py::arg("cam_pos"), py::arg("visibility"))
		.def("num_tiles", &osc::SpecCone_Generator::numTiles)
		.def("tile_size", &osc::SpecCone_Generator::tileSize)
		.def("num_pixels", &osc::SpecCone_Generator::numPixels)
		.def("num_levels", &osc::SpecCone_Generator::numLevels)
		.def("render_tile", [](osc::SpecCone_Generator& self, int tile_idx)
				-> osc::SpecCone_Generator::TileResult {
			return self.renderTile(tile_idx);
		}, py::arg("tile_idx"), py::return_value_policy::move);

	// --- HemiVisTileResult / HemiVisGenerator ---

	py::class_<osc::HemiVis_Generator::TileResult>(m, "HemiVisTileResult")
		.def_readonly("tile_texels", &osc::HemiVis_Generator::TileResult::tile_texels)
		.def_readonly("num_samples", &osc::HemiVis_Generator::TileResult::num_samples)
		.def_readonly("num_cams", &osc::HemiVis_Generator::TileResult::num_cams)
		.def_property_readonly("t_hit_np", [](osc::HemiVis_Generator::TileResult& r) {
			py::ssize_t t = static_cast<py::ssize_t>(r.tile_texels);
			py::ssize_t s = static_cast<py::ssize_t>(r.num_samples);
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ t, s },
				{ py::ssize_t(s * sizeof(float)), py::ssize_t(sizeof(float)) },
				r.t_hit_shared.data(),
				base
			);
		}, "t_hit of the shared rays, shape (tile_texels, num_samples): "
		   ">0 hit, 0 miss, <0 ray not launched.")
		.def_property_readonly("t_hit_mirror_np", [](osc::HemiVis_Generator::TileResult& r) {
			py::ssize_t t = static_cast<py::ssize_t>(r.tile_texels);
			py::ssize_t c = static_cast<py::ssize_t>(r.num_cams);
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ t, c },
				{ py::ssize_t(c * sizeof(float)), py::ssize_t(sizeof(float)) },
				r.t_hit_mirror.empty() ? nullptr : r.t_hit_mirror.data(),
				base
			);
		}, "t_hit of the mirror rays, shape (tile_texels, num_cams).")
		.def_property_readonly("dirs_np", [](osc::HemiVis_Generator::TileResult& r) {
			py::ssize_t t = static_cast<py::ssize_t>(r.tile_texels);
			py::ssize_t s = static_cast<py::ssize_t>(r.num_samples);
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ t, s, py::ssize_t(3) },
				{ py::ssize_t(s * sizeof(gdt::vec3f)), py::ssize_t(sizeof(gdt::vec3f)),
				  py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.dirs_shared.empty() ? nullptr : r.dirs_shared.data()),
				base
			);
		}, "Shared directions traced (only with set_debug_directions(True)).")
		.def_property_readonly("dirs_mirror_np", [](osc::HemiVis_Generator::TileResult& r) {
			py::ssize_t t = static_cast<py::ssize_t>(r.tile_texels);
			py::ssize_t c = static_cast<py::ssize_t>(r.num_cams);
			py::object base = py::cast(&r);
			return py::array_t<float>(
				{ t, c, py::ssize_t(3) },
				{ py::ssize_t(c * sizeof(gdt::vec3f)), py::ssize_t(sizeof(gdt::vec3f)),
				  py::ssize_t(sizeof(float)) },
				reinterpret_cast<float*>(r.dirs_mirror.empty() ? nullptr : r.dirs_mirror.data()),
				base
			);
		}, "Mirror directions traced (only with set_debug_directions(True)).");

	py::class_<osc::HemiVis_Generator>(m, "HemiVisGenerator")
		.def(py::init<>())
		.def("set_traversable", &osc::HemiVis_Generator::setTraversable, py::arg("model"))
		.def("set_inputs", &osc::HemiVis_Generator::setInputs,
			py::arg("ium_result"), py::arg("num_samples"), py::arg("tile_size") = 1024,
			"num_samples = shared Fibonacci samples per texel (S). The directions are "
			"deterministic and have to be rebuilt on the Python side with the same "
			"formula as the kernel (see deviceProgramsHemiVis.cu).")
		.def("set_cameras", [](osc::HemiVis_Generator& self,
				const std::vector<gdt::vec3f>& cam_positions) {
			self.setCameras(cam_positions);
		}, py::arg("cam_positions"),
		   "World positions of the cameras for the mirror rays (empty list = none).")
		.def("set_debug_directions", &osc::HemiVis_Generator::setDebugDirections,
			py::arg("enabled"),
			"Also return the traced directions (dirs_np / dirs_mirror_np). "
			"Needed by the kernel<->torch parity test; costs 12 B/ray.")
		.def("num_tiles", &osc::HemiVis_Generator::numTiles)
		.def("tile_size", &osc::HemiVis_Generator::tileSize)
		.def("num_pixels", &osc::HemiVis_Generator::numPixels)
		.def("num_samples", &osc::HemiVis_Generator::numSamples)
		.def("num_cameras", &osc::HemiVis_Generator::numCameras)
		.def("render_tile", [](osc::HemiVis_Generator& self, int tile_idx)
				-> osc::HemiVis_Generator::TileResult {
			return self.renderTile(tile_idx);
		}, py::arg("tile_idx"), py::return_value_policy::move);

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
