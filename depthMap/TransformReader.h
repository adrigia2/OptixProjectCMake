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
#include "gdt/math/vec.h"

// Include nlohmann/json
#include "nlohmann/json.hpp"

namespace osc {
	using namespace gdt;
	using json = nlohmann::json;

	// Parser JSON robusto per transforms.json (formato NeRF)
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

		std::string getFilePath() const {
			return file_path;
		}

		std::string getFolderPath() const {
			size_t lastSlash = file_path.find_last_of("/\\");
			if (lastSlash != std::string::npos) {
				return file_path.substr(0, lastSlash);
			}
			return "";
		}

		std::string getFileName(bool keepExtension = true) const {
			size_t lastSlash = file_path.find_last_of("/\\");
			size_t start = (lastSlash == std::string::npos) ? 0 : lastSlash + 1;
			if (keepExtension) {
				return file_path.substr(start);
			}
			else {
				size_t lastDot = file_path.find_last_of('.');
				if (lastDot != std::string::npos && lastDot > start) {
					return file_path.substr(start, lastDot - start);
				}
				else {
					return file_path.substr(start);
				}
			}
		}
	};

		struct TransformData {
			float camera_angle_x;
			int w, h;
			std::vector<CameraTransform> frames;

			// Parser robusto con nlohmann/json
			bool loadFromFile(const std::string& filename) {
				try {
					std::ifstream file(filename);
					if (!file.is_open()) {
						std::cerr << "Impossibile aprire il file: " << filename << std::endl;
						return false;
					}

					json j;
					file >> j;

					// Parse camera_angle_x
					if (j.contains("camera_angle_x")) {
						camera_angle_x = j["camera_angle_x"].get<float>();
					}
					else {
						std::cerr << "WARN: 'camera_angle_x' non trovato, uso default 0.0" << std::endl;
						camera_angle_x = 0.0f;
					}

					// Parse w e h
					w = j.value("w", 800);
					h = j.value("h", 800);

					// Parse frames array
					if (j.contains("frames") && j["frames"].is_array()) {
						for (const auto& frame : j["frames"]) {
							CameraTransform cam;

							// Inizializza matrice identità
							for (int i = 0; i < 4; i++) {
								for (int k = 0; k < 4; k++) {
									cam.transform_matrix[i][k] = (i == k) ? 1.0f : 0.0f;
								}
							}

							// Parse file_path (opzionale)
							if (frame.contains("file_path")) {
								cam.file_path = frame["file_path"].get<std::string>();
							}

							// Parse transform_matrix
							if (frame.contains("transform_matrix") &&
								frame["transform_matrix"].is_array()) {

								const auto& matrix = frame["transform_matrix"];

								// Verifica che sia una matrice 4x4
								if (matrix.size() >= 4) {
									for (int i = 0; i < 4; i++) {
										if (matrix[i].is_array() && matrix[i].size() >= 4) {
											for (int k = 0; k < 4; k++) {
												cam.transform_matrix[i][k] = matrix[i][k].get<float>();
											}
										}
										else {
											std::cerr << "WARN: Riga " << i << " della matrice non è un array valido" << std::endl;
										}
									}
								}
								else {
									std::cerr << "WARN: transform_matrix non ha 4 righe" << std::endl;
								}
							}
							else {
								std::cerr << "WARN: transform_matrix non trovata o non è un array" << std::endl;
							}

							frames.push_back(cam);
						}
					}
					else {
						std::cerr << "ERROR: 'frames' non trovato o non è un array" << std::endl;
						return false;
					}

					std::cout << "==================================================" << std::endl;
					std::cout << "Transform JSON caricato con successo!" << std::endl;
					std::cout << "  Frames: " << frames.size() << std::endl;
					std::cout << "  Risoluzione: " << w << "x" << h << std::endl;
					std::cout << "  Camera angle X: " << camera_angle_x << " radianti" << std::endl;
					std::cout << "==================================================" << std::endl;

					return frames.size() > 0;

				}
				catch (const json::parse_error& e) {
					std::cerr << "Errore di parsing JSON: " << e.what() << std::endl;
					std::cerr << "  at byte " << e.byte << std::endl;
					return false;
				}
				catch (const json::type_error& e) {
					std::cerr << "Errore di tipo JSON: " << e.what() << std::endl;
					return false;
				}
				catch (const std::exception& e) {
					std::cerr << "Errore durante il caricamento: " << e.what() << std::endl;
					return false;
				}
			}

			// Metodo helper per validare i dati caricati
			bool validate() const {
				if (frames.empty()) {
					std::cerr << "Nessun frame caricato" << std::endl;
					return false;
				}

				if (w <= 0 || h <= 0) {
					std::cerr << "Dimensioni immagine non valide: " << w << "x" << h << std::endl;
					return false;
				}

				if (camera_angle_x <= 0.0f || camera_angle_x > 3.15f) {
					std::cerr << "WARN: camera_angle_x sembra non valido: " << camera_angle_x << std::endl;
				}

				return true;
			}

			// Metodo helper per stampare informazioni di debug
			void printDebugInfo(size_t maxFrames = 3) const {
				std::cout << "\n=== Transform Data Debug Info ===" << std::endl;
				std::cout << "Numero di frames: " << frames.size() << std::endl;
				std::cout << "Risoluzione: " << w << "x" << h << std::endl;
				std::cout << "Camera FOV X: " << camera_angle_x << " rad ("
					<< (camera_angle_x * 180.0f / 3.14159f) << " gradi)" << std::endl;

				size_t numToPrint = std::min(maxFrames, frames.size());
				for (size_t i = 0; i < numToPrint; i++) {
					std::cout << "\nFrame " << i << ":" << std::endl;
					std::cout << "  File: " << frames[i].file_path << std::endl;
					std::cout << "  Position: " << frames[i].getPosition() << std::endl;
					std::cout << "  Forward: " << frames[i].getForward() << std::endl;
					std::cout << "  Up: " << frames[i].getUp() << std::endl;
				}

				if (frames.size() > maxFrames) {
					std::cout << "\n... e altri " << (frames.size() - maxFrames) << " frames" << std::endl;
				}
				std::cout << "==================================\n" << std::endl;
			}
		};

	} // ::osc