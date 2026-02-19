#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "TriangleMesh.h"
#include "OptixManager.h"
#include "IUM_Generator.h"
#include "Depth_Generator.h"
#include "ImageResultType.h"
#include "LogManager.h"

namespace py = pybind11;

// Classe wrapper per gestire il flusso completo di lavoro
class OptiXPipeline {

};

PYBIND11_MODULE(depthMapModule, m) {
  
}
