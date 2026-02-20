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

	/*! main entry point to this example - initially optix, print hello
	  world, then exit */
	extern "C" int main(int ac, char** av)
	{
		TriangleMesh model;
		model.addFromObjFile("C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Models/SwordShield.obj");
		LogManager::LogInfo("Model loaded with %zu vertices and %zu triangles", model.vertex.size(), model.index.size());
	
		IUM_Generator iumGenerator;
		Depth_Generator depthGenerator;
		LogManager::LogInfo("Starting IUM generation...");
		iumGenerator.setTraversable(model);
		iumGenerator.setTextureSize(vec2i(1024, 1024)); // Imposta la dimensione della texture per l'IUM
		iumGenerator.render();


		Camera camera;
		camera.pos = vec3f(0.0f, 0.0f, 5.0f); // Posizione della camera
		camera.forward = vec3f(0.0f, 0.0f, -1.0f); // Direzione della camera
		camera.up = vec3f(0.0f, 1.0f, 0.0f); // Up vector della camera

		depthGenerator.setTraversable(model);
		depthGenerator.setCamera(camera, 45.0f, vec2i(1024, 1024)); // Imposta la camera e la dimensione del frame
		depthGenerator.needRenderDepth(true); // Abilita il rendering della depth map
		depthGenerator.meedRenderPosition(true);
		depthGenerator.needRenderNormal(true);
		depthGenerator.render();
	}

} // ::osc
