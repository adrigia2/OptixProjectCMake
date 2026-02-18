// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"
#include "TriangleMesh.h"
#include "OptixManager.h"
#include "IUM_Generator.h"
#include "Camera.h"
#include "LogManager.h"
#include "ImageResultType.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>
#include "Depth_Generator.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

	TriangleMesh model;

	extern "C" void loadModel(const std::string& modelPath)
	{
		LogManager::LogInfo("Loading 3D model...");
		model.addFromObjFile(modelPath);
		LogManager::LogInfo("3D model loaded: %zu vertices, %zu triangles", model.vertex.size(), model.index.size());
	}

	extern "C" void createIUM(const std::string& outputPath, const std::string& fileName, const ImageResultType& type)
	{
		LogManager::LogInfo("Creating OptixManager...");
		OptixManager optixManager;
		LogManager::LogInfo("OptixManager created successfully.");

		LogManager::LogInfo("Creating IUM_Generator...");
		IUM_Generator iumGenerator(optixManager);
		LogManager::LogInfo("IUM_Generator created successfully.");

		LogManager::LogInfo("Setting up IUM_Generator...");
		iumGenerator.setTraversable(model);
		iumGenerator.setTextureSize(512, 512); // Imposta la dimensione della texture per la depth map
		iumGenerator.printStatus();
		optixManager.printStatus();

		optixManager.createPipeline();
		optixManager.buildSBT();

		LogManager::LogInfo("Starting rendering...");
		iumGenerator.render();

		LogManager::LogInfo("Rendering completed");

		LogManager::LogInfo("Saving IUM texture to bitmap...");

		if (type == ImageResultType::BMP)
		{
			iumGenerator.saveIUMTextureToBitmap(outputPath + fileName + ".bmp");
			LogManager::LogInfo("Done! Check ium_output.bmp for results.");
		}

		if (type == ImageResultType::OpenEXR)
		{
			iumGenerator.saveIUMTextureToOpenExr(outputPath + fileName + ".exr");
			LogManager::LogInfo("Done! Check ium_output.exr for results.");
		}

	}

	extern "C" void createDepthMaps(const std::string& transformFile, const std::string& outputDir, const ImageResultType& type)
	{
		LogManager::LogInfo("Creating OptixManager...");
		OptixManager optixManager;
		LogManager::LogInfo("OptixManager created successfully.");

		LogManager::LogInfo("Creating Depth_Generator...");
		Depth_Generator depthGenerator(optixManager);
		LogManager::LogInfo("Depth_Generator created successfully.");

		LogManager::LogInfo("Creating depth maps from transforms...");
		depthGenerator.setTraversable(model);

		optixManager.createPipeline();
		optixManager.buildSBT();

		depthGenerator.renderTransforms(transformFile, outputDir);

		if (type == ImageResultType::BMP)
		{
			LogManager::LogInfo("Saving depth maps to bitmap...");
			depthGenerator.saveIUMTextureToBitmapAll(outputDir);
			LogManager::LogInfo("Done! Check the output directory for depth maps.");
		}

		if (type == ImageResultType::OpenEXR)
		{
			LogManager::LogInfo("Saving depth maps to OpenEXR...");
			depthGenerator.saveDepthMapsToOpenExrAll(outputDir);
			LogManager::LogInfo("Done! Check the output directory for depth maps in OpenEXR format.");
		}



	}

	/*! main entry point to this example - initially optix, print hello
	  world, then exit */
	extern "C" int main(int ac, char** av)
	{
		loadModel("C:/Users/adriano.cicco/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Models/SwordShield.obj");

		createIUM(
			"C:/Users/adriano.cicco/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/InverseUvMapping/", 
			"ium_output",
			ImageResultType::OpenEXR
		);
		createDepthMaps
		(
			"C:/Users/adriano.cicco/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Nerf/transforms.json", 
			"C:/Users/adriano.cicco/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Depth/",
			ImageResultType::OpenEXR
		);
	}

} // ::osc
