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



	}

	/*! main entry point to this example - initially optix, print hello
	  world, then exit */
	extern "C" int main(int ac, char** av)
	{
		// try {
		//   TriangleMesh model;
		//   
		//   // Carica il modello 3D
		   //std::cout << "==================================================" << std::endl;
		   //std::cout << "Caricamento modello 3D..." << std::endl;
		   //auto filePath = std::string("C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Models/SwordShield.obj");
		   //model.addFromObjFile(filePath);

		   //std::cout << "==================================================" << std::endl;

		//   std::cout << "Modello caricato: " << model.vertex.size() << " vertici, " 
		//             << model.index.size() << " triangoli" << std::endl;

		//   Camera camera = { /*pos*/vec3f(0,-10.f,1.f),
		//                     /* direction */vec3f(0.f,1.f,0.f),
		//                     /* up */vec3f(0.f,0.f,1.f) };

		//   // something approximating the scale of the world, so the
		//   // camera knows how much to move for any given user interaction:
		//   const float worldScale = 1.f;

		//   // Controlla se è richiesta la generazione di depth maps da transforms.json
		//   bool generateDepthMaps = false;
		//   std::string transformFile;
		//   std::string outputDir = "./depth_maps";
		//   
		//   for (int i = 1; i < ac; i++) {
		//     std::string arg = av[i];
		//     if (arg == "--transform" && i + 1 < ac) {
		//       transformFile = av[i + 1];
		//       generateDepthMaps = true;
		//       i++;
		//     } else if (arg == "--output" && i + 1 < ac) {
		//       outputDir = av[i + 1];
		//       i++;
		//     } else if (arg == "--help") {
		//       std::cout << "Uso:" << std::endl;
		//       std::cout << "  " << av[0] << " [opzioni]" << std::endl;
		//       std::cout << std::endl;
		//       std::cout << "Opzioni:" << std::endl;
		//       std::cout << "  --transform <file>  Percorso al file transforms.json" << std::endl;
		//       std::cout << "  --output <dir>      Directory di output per le depth maps (default: ./depth_maps)" << std::endl;
		//       std::cout << "  --help              Mostra questo messaggio" << std::endl;
		//       return 0;
		//     }
		//   }

		//   if (generateDepthMaps && !transformFile.empty()) {
		//     std::cout << std::endl;
		//     std::cout << "==================================================" << std::endl;
		//     std::cout << "Modalità generazione depth maps" << std::endl;
		//     std::cout << "Transform file: " << transformFile << std::endl;
		//     std::cout << "Output directory: " << outputDir << std::endl;
		//     std::cout << "==================================================" << std::endl;
		//     std::cout << std::endl;
		//     
		//     // Crea il renderer senza finestra
		//     SampleRenderer renderer(model);
		//     
		//     // Genera le depth maps
		//     renderer.generateDepthMapsFromTransform(transformFile, outputDir);
		//     
		//     std::cout << std::endl;
		//     std::cout << "Operazione completata!" << std::endl;
		//     
		//   } else {
		//     // Modalità finestra interattiva normale
		//     SampleWindow *window = new SampleWindow("Optix 7 Depth Map Generator",
		//                                             model,camera,worldScale);
		//     window->run();
		//   }
		//   
		// } catch (std::runtime_error& e) {
		//   std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
		//             << GDT_TERMINAL_DEFAULT << std::endl;
		//   exit(1);
		// }
		// return 0;

		loadModel("C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Models/SwordShield.obj");

		createIUM(
			"C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/InverseUvMapping/", 
			"ium_output",
			ImageResultType::BMP
		);
		createDepthMaps
		(
			"C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Nerf/transforms.json", 
			"C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Depth/",
			ImageResultType::BMP
		);
	}

} // ::osc
