#include "PipelineManager.h"


void PipelineManager::createPipelines()
{
}

void PipelineManager::destroyPipelines()
{
}

void PipelineManager::LoadModel(const std::string& modelPath)
{
	model.addFromObjFile(modelPath);
}

void PipelineManager::GenerateIUM(const std::string& outputPath, const std::string& fileName, uint32_t width, uint32_t height)
{
	OptixManager optixManager;

}