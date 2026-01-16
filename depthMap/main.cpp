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

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct SampleWindow : public GLFCameraWindow
  {
    SampleWindow(const std::string &title,
                 const TriangleMesh &model,
                 const Camera &camera,
                 const float worldScale)
      : GLFCameraWindow(title,camera.pos,camera.forward,camera.up,worldScale),
        sample(model)
    {
    }

    virtual void render() override
    {
      if (cameraFrame.modified) {
        sample.setCamera(Camera{ cameraFrame.get_position(),
                                 cameraFrame.get_forward(),
                                 cameraFrame.get_up() });
        cameraFrame.modified = false;
      }
      sample.render();
    }
    
    virtual void draw() override
    {
      sample.downloadPixels(pixels.data());
      if (fbTexture == 0)
        glGenTextures(1, &fbTexture);
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      GLenum texFormat = GL_RGBA;
      GLenum texelType = GL_UNSIGNED_BYTE;
      glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                   texelType, pixels.data());

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x*newSize.y);
    }

    vec2i                 fbSize;
    GLuint                fbTexture {0};
    SampleRenderer        sample;
    std::vector<uint32_t> pixels;
  };
  
  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      TriangleMesh model;
      
      // Carica il modello 3D
	  auto filePath = std::string("C:/Users/adria/Documents/GitHub/OptixProjectCMake/Scenes/SwordShield/Models/SwordShield.obj");
	  model.addFromObjFile(filePath);

      std::cout << "Modello caricato: " << model.vertex.size() << " vertici, " 
                << model.index.size() << " triangoli" << std::endl;

      Camera camera = { /*pos*/vec3f(0,-10.f,1.f),
                        /* direction */vec3f(0.f,1.f,0.f),
                        /* up */vec3f(0.f,0.f,1.f) };

      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = 1.f;

      // Controlla se è richiesta la generazione di depth maps da transforms.json
      bool generateDepthMaps = false;
      std::string transformFile;
      std::string outputDir = "./depth_maps";
      
      for (int i = 1; i < ac; i++) {
        std::string arg = av[i];
        if (arg == "--transform" && i + 1 < ac) {
          transformFile = av[i + 1];
          generateDepthMaps = true;
          i++;
        } else if (arg == "--output" && i + 1 < ac) {
          outputDir = av[i + 1];
          i++;
        } else if (arg == "--help") {
          std::cout << "Uso:" << std::endl;
          std::cout << "  " << av[0] << " [opzioni]" << std::endl;
          std::cout << std::endl;
          std::cout << "Opzioni:" << std::endl;
          std::cout << "  --transform <file>  Percorso al file transforms.json" << std::endl;
          std::cout << "  --output <dir>      Directory di output per le depth maps (default: ./depth_maps)" << std::endl;
          std::cout << "  --help              Mostra questo messaggio" << std::endl;
          return 0;
        }
      }

      if (generateDepthMaps && !transformFile.empty()) {
        std::cout << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << "Modalità generazione depth maps" << std::endl;
        std::cout << "Transform file: " << transformFile << std::endl;
        std::cout << "Output directory: " << outputDir << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << std::endl;
        
        // Crea il renderer senza finestra
        SampleRenderer renderer(model);
        
        // Genera le depth maps
        renderer.generateDepthMapsFromTransform(transformFile, outputDir);
        
        std::cout << std::endl;
        std::cout << "Operazione completata!" << std::endl;
        
      } else {
        // Modalità finestra interattiva normale
        SampleWindow *window = new SampleWindow("Optix 7 Depth Map Generator",
                                                model,camera,worldScale);
        window->run();
      }
      
    } catch (std::runtime_error& e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
      exit(1);
    }
    return 0;
  }
  
} // ::osc
