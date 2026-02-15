#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SampleRenderer.h"
#include "TriangleMesh.h"
#include "OptixManager.h"
#include "IUM_Generator.h"
#include "Depth_Generator.h"
#include "ImageResultType.h"
#include "LogManager.h"

namespace py = pybind11;

// Classe wrapper per gestire il flusso completo di lavoro
class OptiXPipeline {
private:
    TriangleMesh model;
    bool modelLoaded = false;

public:
    // Carica il modello 3D
    void loadModel(const std::string& modelPath) {
        LogManager::LogInfo("Loading 3D model from: %s", modelPath.c_str());
        model.addFromObjFile(modelPath);
        modelLoaded = true;
        LogManager::LogInfo("3D model loaded: %zu vertices, %zu triangles", 
                          model.vertex.size(), model.index.size());
    }

    // Genera l'Inverse UV Mapping
    void generateIUM(const std::string& outputPath, 
                     const std::string& fileName,
                     ImageResultType type = ImageResultType::OpenEXR,
                     uint32_t width = 512,
                     uint32_t height = 512) {
        if (!modelLoaded) {
            throw std::runtime_error("Model not loaded! Call loadModel() first.");
        }

        LogManager::LogInfo("Creating IUM pipeline...");
        OptixManager optixManager;
        IUM_Generator iumGenerator(optixManager);

        LogManager::LogInfo("Setting up IUM_Generator...");
        iumGenerator.setTraversable(model);
        iumGenerator.setTextureSize(width, height);
        iumGenerator.printStatus();
        optixManager.printStatus();

        optixManager.createPipeline();
        optixManager.buildSBT();

        LogManager::LogInfo("Rendering IUM...");
        iumGenerator.render();

        LogManager::LogInfo("Saving IUM texture...");
        if (type == ImageResultType::BMP) {
            iumGenerator.saveIUMTextureToBitmap(outputPath + fileName + ".bmp");
            LogManager::LogInfo("IUM saved as BMP: %s.bmp", fileName.c_str());
        } else if (type == ImageResultType::OpenEXR) {
            iumGenerator.saveIUMTextureToOpenExr(outputPath + fileName + ".exr");
            LogManager::LogInfo("IUM saved as OpenEXR: %s.exr", fileName.c_str());
        }
    }

    // Genera le depth maps da file transforms
    void generateDepthMaps(const std::string& transformFile,
                          const std::string& outputDir,
                          ImageResultType type = ImageResultType::OpenEXR) {
        if (!modelLoaded) {
            throw std::runtime_error("Model not loaded! Call loadModel() first.");
        }

        LogManager::LogInfo("Creating depth map pipeline...");
        OptixManager optixManager;
        Depth_Generator depthGenerator(optixManager);

        LogManager::LogInfo("Setting up Depth_Generator...");
        depthGenerator.setTraversable(model);

        optixManager.createPipeline();
        optixManager.buildSBT();

        LogManager::LogInfo("Rendering depth maps from transforms...");
        depthGenerator.renderTransforms(transformFile, outputDir);

        LogManager::LogInfo("Saving depth maps...");
        if (type == ImageResultType::BMP) {
            depthGenerator.saveIUMTextureToBitmapAll(outputDir);
            LogManager::LogInfo("Depth maps saved as BMP");
        } else if (type == ImageResultType::OpenEXR) {
            depthGenerator.saveDepthMapsToOpenExrAll(outputDir);
            LogManager::LogInfo("Depth maps saved as OpenEXR");
        }
    }

    // Metodo combinato per eseguire tutto il flusso
    void processAll(const std::string& modelPath,
                   const std::string& transformFile,
                   const std::string& iumOutputPath,
                   const std::string& depthOutputDir,
                   const std::string& iumFileName = "ium_output",
                   ImageResultType type = ImageResultType::OpenEXR,
                   uint32_t iumWidth = 512,
                   uint32_t iumHeight = 512) {
        loadModel(modelPath);
        generateIUM(iumOutputPath, iumFileName, type, iumWidth, iumHeight);
        generateDepthMaps(transformFile, depthOutputDir, type);
        LogManager::LogInfo("All processing completed successfully!");
    }

    // Getter per informazioni sul modello
    size_t getVertexCount() const { 
        return modelLoaded ? model.vertex.size() : 0; 
    }
    
    size_t getTriangleCount() const { 
        return modelLoaded ? model.index.size() : 0; 
    }

    bool isModelLoaded() const { 
        return modelLoaded; 
    }
};

PYBIND11_MODULE(depthMapModule, m) {
    m.doc() = "Python bindings for OptiX depth map and IUM renderer";

    // Enum per il tipo di output
    py::enum_<ImageResultType>(m, "ImageResultType")
        .value("BMP", ImageResultType::BMP)
        .value("OpenEXR", ImageResultType::OpenEXR)
        .export_values();

    // Binding per vec3f (utile per debug/ispezione)
    py::class_<gdt::vec3f>(m, "Vec3f")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &gdt::vec3f::x)
        .def_readwrite("y", &gdt::vec3f::y)
        .def_readwrite("z", &gdt::vec3f::z)
        .def("__repr__", [](const gdt::vec3f &v) {
            return "Vec3f(" + std::to_string(v.x) + ", " + 
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    // Classe principale per il workflow Python
    py::class_<OptiXPipeline>(m, "OptiXPipeline")
        .def(py::init<>())
        .def("load_model", &OptiXPipeline::loadModel,
             py::arg("model_path"),
             "Load a 3D model from OBJ file")
        .def("generate_ium", &OptiXPipeline::generateIUM,
             py::arg("output_path"),
             py::arg("file_name"),
             py::arg("image_type") = ImageResultType::OpenEXR,
             py::arg("width") = 512,
             py::arg("height") = 512,
             "Generate Inverse UV Mapping texture")
        .def("generate_depth_maps", &OptiXPipeline::generateDepthMaps,
             py::arg("transform_file"),
             py::arg("output_dir"),
             py::arg("image_type") = ImageResultType::OpenEXR,
             "Generate depth maps from transforms.json file")
        .def("process_all", &OptiXPipeline::processAll,
             py::arg("model_path"),
             py::arg("transform_file"),
             py::arg("ium_output_path"),
             py::arg("depth_output_dir"),
             py::arg("ium_file_name") = "ium_output",
             py::arg("image_type") = ImageResultType::OpenEXR,
             py::arg("ium_width") = 512,
             py::arg("ium_height") = 512,
             "Execute complete pipeline: load model, generate IUM and depth maps")
        .def("get_vertex_count", &OptiXPipeline::getVertexCount,
             "Get number of vertices in loaded model")
        .def("get_triangle_count", &OptiXPipeline::getTriangleCount,
             "Get number of triangles in loaded model")
        .def("is_model_loaded", &OptiXPipeline::isModelLoaded,
             "Check if a model is currently loaded");

    // Funzioni di utilità per il logging
    m.def("set_log_level", [](int level) {
        // Placeholder per impostare il livello di log
        LogManager::LogInfo("Log level set to: %d", level);
    }, py::arg("level"), "Set logging level (0=Error, 1=Warning, 2=Info, 3=Debug)");
}
