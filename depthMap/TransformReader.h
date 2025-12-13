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

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include "gdt/math/vec.h"

namespace osc {
  using namespace gdt;

  // Simple JSON parser per transforms.json (formato NeRF)
  struct CameraTransform {
    float transform_matrix[4][4];
    std::string file_path;
    
    vec3f getPosition() const {
      return vec3f(transform_matrix[0][3], 
                   transform_matrix[1][3], 
                   transform_matrix[2][3]);
    }
    
    vec3f getForward() const {
      // In NeRF convention, forward is -Z axis
      return -vec3f(transform_matrix[0][2], 
                    transform_matrix[1][2], 
                    transform_matrix[2][2]);
    }
    
    vec3f getUp() const {
      // Up is Y axis
      return vec3f(transform_matrix[0][1], 
                   transform_matrix[1][1], 
                   transform_matrix[2][1]);
    }
  };

  struct TransformData {
    float camera_angle_x;
    int w, h;
    std::vector<CameraTransform> frames;
    
    // Simple parser senza librerie esterne
    bool loadFromFile(const std::string& filename) {
      std::ifstream file(filename);
      if (!file.is_open()) {
        std::cerr << "Impossibile aprire il file: " << filename << std::endl;
        return false;
      }
      
      std::string line, content;
      while (std::getline(file, line)) {
        content += line;
      }
      file.close();
      
      try {
        // Parse camera_angle_x
        size_t pos = content.find("\"camera_angle_x\"");
        if (pos != std::string::npos) {
          size_t colon = content.find(":", pos);
          size_t comma = content.find(",", colon);
          std::string value = content.substr(colon + 1, comma - colon - 1);
          camera_angle_x = std::stof(value);
        }
        
        // Parse w
        pos = content.find("\"w\"");
        if (pos != std::string::npos) {
          size_t colon = content.find(":", pos);
          size_t comma = content.find(",", colon);
          std::string value = content.substr(colon + 1, comma - colon - 1);
          w = std::stoi(value);
        } else {
          w = 800; // default
        }
        
        // Parse h
        pos = content.find("\"h\"");
        if (pos != std::string::npos) {
          size_t colon = content.find(":", pos);
          size_t comma = content.find(",", colon);
          if (comma == std::string::npos) comma = content.find("}", colon);
          std::string value = content.substr(colon + 1, comma - colon - 1);
          h = std::stoi(value);
        } else {
          h = 800; // default
        }
        
        // Parse frames array
        pos = content.find("\"frames\"");
        if (pos != std::string::npos) {
          size_t arrayStart = content.find("[", pos);
          size_t arrayEnd = content.find("]", arrayStart);
          std::string framesContent = content.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
          
          // Parse ogni frame
          size_t frameStart = 0;
          while ((frameStart = framesContent.find("{", frameStart)) != std::string::npos) {
            size_t frameEnd = framesContent.find("}", frameStart);
            std::string frameContent = framesContent.substr(frameStart, frameEnd - frameStart + 1);
            
            CameraTransform cam;
            
            // Parse file_path
            size_t filePathPos = frameContent.find("\"file_path\"");
            if (filePathPos != std::string::npos) {
              size_t colonPos = frameContent.find(":", filePathPos);
              size_t quote1 = frameContent.find("\"", colonPos);
              size_t quote2 = frameContent.find("\"", quote1 + 1);
              cam.file_path = frameContent.substr(quote1 + 1, quote2 - quote1 - 1);
            }
            
            // Parse transform_matrix
            size_t matrixPos = frameContent.find("\"transform_matrix\"");
            if (matrixPos != std::string::npos) {
              size_t matrixStart = frameContent.find("[", matrixPos);
              size_t matrixEnd = frameContent.rfind("]");
              std::string matrixContent = frameContent.substr(matrixStart + 1, matrixEnd - matrixStart - 1);
              
              // Parse le 4 righe della matrice
              int row = 0;
              size_t rowStart = 0;
              while (row < 4 && (rowStart = matrixContent.find("[", rowStart)) != std::string::npos) {
                size_t rowEnd = matrixContent.find("]", rowStart);
                std::string rowContent = matrixContent.substr(rowStart + 1, rowEnd - rowStart - 1);
                
                std::istringstream iss(rowContent);
                std::string token;
                int col = 0;
                while (std::getline(iss, token, ',') && col < 4) {
                  cam.transform_matrix[row][col] = std::stof(token);
                  col++;
                }
                
                rowStart = rowEnd + 1;
                row++;
              }
            }
            
            frames.push_back(cam);
            frameStart = frameEnd + 1;
          }
        }
        
        std::cout << "Caricati " << frames.size() << " frames dal file transforms.json" << std::endl;
        std::cout << "Risoluzione: " << w << "x" << h << std::endl;
        std::cout << "Camera angle X: " << camera_angle_x << " radianti" << std::endl;
        return true;
        
      } catch (const std::exception& e) {
        std::cerr << "Errore durante il parsing del JSON: " << e.what() << std::endl;
        return false;
      }
    }
  };

} // ::osc