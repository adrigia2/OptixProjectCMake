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
	enum ImageType {
		IMAGE_TYPE_RGB,
		IMAGE_TYPE_DEPTH
	};

	using namespace gdt;
	using json = nlohmann::json;

	// Parser JSON robusto per transforms.json (formato NeRF)
	struct TransformFrame {
		float transform_matrix[4][4];
		std::string file_path;
		std::string depth_path; // Nuovo campo depth_path

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

		std::string getDepthPath() const {
			return depth_path;
		}

		void setDepthPath(const std::string& path) {
			depth_path = path;
			std::cout << "Depth path impostato a: " << depth_path << std::endl;
		}


		std::string getFolderPath(ImageType imageTarget) const {

			std::string path = (imageTarget == IMAGE_TYPE_RGB) ? file_path : depth_path;

			size_t lastSlash = path.find_last_of("/\\");
			if (lastSlash != std::string::npos) {
				return path.substr(0, lastSlash);
			}
			return "";
		}

		std::string getFileName(ImageType imageTarget, bool keepExtension = true) const {
			std::string file_path = (imageTarget == IMAGE_TYPE_RGB) ? this->file_path : this->depth_path;

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

		// Aggiunto campo sharpness
		float sharpness = 1.0f;
	};

		struct TransformData {
			float camera_angle_x;
			float camera_angle_y = 0.0f;  // opzionale
			float fl_x = 0.0f;            // opzionale
			float fl_y = 0.0f;            // opzionale
			float cx = 0.0f;              // opzionale
			float cy = 0.0f;              // opzionale
			float scale = 1.0f;           // opzionale
			int aabb_scale = 1;           // opzionale
			int w, h;
			std::vector<TransformFrame> frames;

			// Nuovo metodo per salvare con depth_path
			bool saveToFileWithDepth(const std::string& filename, 
								  const std::string& depthDir = "depth") const {
				try {
					json j;
					
					// Copia tutti i campi base
					j["camera_angle_x"] = camera_angle_x;
					j["w"] = w;
					j["h"] = h;
					
					// Campi opzionali (aggiungi solo se diversi dal valore di default)
					if (camera_angle_y != 0.0f) {
						j["camera_angle_y"] = camera_angle_y;
					}
					if (fl_x != 0.0f) {
						j["fl_x"] = fl_x;
					}
					if (fl_y != 0.0f) {
						j["fl_y"] = fl_y;
					}
					if (cx != 0.0f) {
						j["cx"] = cx;
					}
					if (cy != 0.0f) {
						j["cy"] = cy;
					}
					if (scale != 1.0f) {
						j["scale"] = scale;
					}
					if (aabb_scale != 1) {
						j["aabb_scale"] = aabb_scale;
					}
					
					// Crea l'array dei frames con depth_path
					j["frames"] = json::array();
					
					for (const auto& frame : frames) {
						json frameJson;
						
						// File path originale
						frameJson["file_path"] = frame.file_path;
						
						// Depth path - usa il percorso assoluto nella directory depthDir
						frameJson["depth_path"] = frame.depth_path;
						
						// Sharpness (opzionale)
						if (frame.sharpness != 1.0f) {
							frameJson["sharpness"] = frame.sharpness;
						}
						
						// Transform matrix
						json matrixJson = json::array();
						for (int i = 0; i < 4; i++) {
							json row = json::array();
							for (int k = 0; k < 4; k++) {
								row.push_back(frame.transform_matrix[i][k]);
							}
							matrixJson.push_back(row);
						}
						frameJson["transform_matrix"] = matrixJson;
						
						j["frames"].push_back(frameJson);
					}
					
					// Scrivi su file con formattazione
					std::ofstream outFile(filename);
					if (!outFile.is_open()) {
						std::cerr << "Impossibile aprire il file per la scrittura: " << filename << std::endl;
						return false;
					}
					
					outFile << j.dump(4);  // Indent con 4 spazi per leggibilità
					outFile.close();
					
					std::cout << "\n==================================================" << std::endl;
					std::cout << "File transformDepth.json salvato con successo!" << std::endl;
					std::cout << "  File: " << filename << std::endl;
					std::cout << "  Frames: " << frames.size() << std::endl;
					std::cout << "  Directory depth maps: " << depthDir << std::endl;
					std::cout << "==================================================" << std::endl;
					
					return true;
					
				} catch (const std::exception& e) {
					std::cerr << "Errore durante il salvataggio del file JSON: " << e.what() << std::endl;
					return false;
				}
			}

			// Metodo modificato loadFromFile per supportare i nuovi campi
			bool loadFromFile(const std::string& filename) {
				try {
					std::ifstream file(filename);
					if (!file.is_open()) {
						std::cerr << "Impossibile aprire il file: " << filename << std::endl;
						return false;
					}

					json j;
					file >> j;

					// Parse camera_angle_x (obbligatorio)
					if (j.contains("camera_angle_x")) {
						camera_angle_x = j["camera_angle_x"].get<float>();
					} else {
						std::cerr << "WARN: 'camera_angle_x' non trovato, uso default 0.0" << std::endl;
						camera_angle_x = 0.0f;
					}

					// Parse campi opzionali
					camera_angle_y = j.value("camera_angle_y", 0.0f);
					fl_x = j.value("fl_x", 0.0f);
					fl_y = j.value("fl_y", 0.0f);
					cx = j.value("cx", 0.0f);
					cy = j.value("cy", 0.0f);
					scale = j.value("scale", 1.0f);
					aabb_scale = j.value("aabb_scale", 1);
					
					// Parse w e h
					w = j.value("w", 800);
					h = j.value("h", 800);

					// Parse frames array
					if (j.contains("frames") && j["frames"].is_array()) {
						for (const auto& frame : j["frames"]) {
							TransformFrame cam;

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
							
							// Parse sharpness (opzionale)
							cam.sharpness = frame.value("sharpness", 1.0f);

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
					if (fl_x != 0.0f) {
						std::cout << "  Focal length X: " << fl_x << std::endl;
					}
					if (fl_y != 0.0f) {
						std::cout << "  Focal length Y: " << fl_y << std::endl;
					}
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