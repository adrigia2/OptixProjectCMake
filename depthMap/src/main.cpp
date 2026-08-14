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

// Standalone smoke test for the OptiX passes.
//
// Loads an OBJ, runs the IUM pass (texture space) and the Depth pass (camera
// space) on it, and prints a summary of what came back. It is the C++
// counterpart of the Python smoke tests in NeRF_Pytorch/scripts/, for checking
// that the ray tracing side works without going through Python at all.
//
// It deliberately writes no images: IOManager only does raw binary I/O, and
// every EXR/PNG in this project is written by the Python pipeline.
//
//   depthMap <model.obj> [texture_size] [--quiet]
//
// texture_size is the side of the square IUM texture (default 1024). The
// camera is derived from the model's bounding box, so any OBJ works.

#include "TriangleMesh.h"
#include "OptixManager.h"
#include "IUM_Generator.h"
#include "Depth_Generator.h"
#include "Camera.h"
#include "LogManager.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

namespace osc {

namespace {

struct Stats
{
	size_t valid = 0;
	float  minValue = std::numeric_limits<float>::infinity();
	float  maxValue = -std::numeric_limits<float>::infinity();
	double sum = 0.0;

	void add(float v)
	{
		++valid;
		minValue = std::min(minValue, v);
		maxValue = std::max(maxValue, v);
		sum += v;
	}
	double mean() const { return valid ? sum / double(valid) : 0.0; }
};

void printUsage()
{
	std::cerr <<
		"usage: depthMap <model.obj> [texture_size] [--quiet]\n"
		"\n"
		"  model.obj     mesh to trace; must carry UVs for the IUM pass\n"
		"  texture_size  side of the square IUM texture (default 1024)\n"
		"  --quiet       only report errors\n";
}

} // anonymous namespace

extern "C" int main(int ac, char** av)
{
	std::string modelPath;
	int textureSize = 1024;
	bool quiet = false;

	for (int i = 1; i < ac; ++i)
	{
		const std::string arg = av[i];
		if (arg == "--quiet") { quiet = true; }
		else if (arg == "-h" || arg == "--help") { printUsage(); return 0; }
		else if (modelPath.empty()) { modelPath = arg; }
		else { textureSize = std::atoi(arg.c_str()); }
	}

	if (modelPath.empty())
	{
		printUsage();
		return 2;
	}
	if (textureSize <= 0)
	{
		std::cerr << "error: texture_size must be positive, got " << textureSize << "\n";
		return 2;
	}

	if (quiet)
		LogManager::SetMinLevel(LogManager::Level::Error);

	try
	{
		// ── Mesh ──────────────────────────────────────────────────────────────
		TriangleMesh model;
		model.addFromObjFile(modelPath);

		if (model.vertex.empty() || model.index.empty())
		{
			LogManager::LogError("No geometry loaded from '%s'.", modelPath.c_str());
			return 1;
		}

		LogManager::Log("Model '%s': %zu vertices, %zu triangles, %zu UVs",
			modelPath.c_str(), model.vertex.size(), model.index.size(),
			model.texcoord.size());

		if (model.texcoord.empty())
			LogManager::LogWarning("The mesh carries no UVs: the IUM pass will produce an empty texture.");

		// Bounding box, used to place the depth camera so that any model fits.
		vec3f bbMin(std::numeric_limits<float>::infinity());
		vec3f bbMax(-std::numeric_limits<float>::infinity());
		for (const vec3f& v : model.vertex)
		{
			bbMin = min(bbMin, v);
			bbMax = max(bbMax, v);
		}
		const vec3f centre = (bbMin + bbMax) * 0.5f;
		const float radius = std::max(length(bbMax - bbMin) * 0.5f, 1e-4f);

		LogManager::Log("Bounding box: (%.3f %.3f %.3f) .. (%.3f %.3f %.3f), radius %.3f",
			bbMin.x, bbMin.y, bbMin.z, bbMax.x, bbMax.y, bbMax.z, radius);

		// ── IUM pass (texture space) ──────────────────────────────────────────
		IUM_Generator iumGenerator;
		iumGenerator.setTraversable(model);
		iumGenerator.setTextureSize(vec2i(textureSize, textureSize));
		iumGenerator.render();

		const IUM_Generator::Result ium = iumGenerator.getResult();
		size_t iumHits = 0;
		for (uint8_t m : ium.masks)
			if (m) ++iumHits;

		const size_t iumTexels = size_t(textureSize) * size_t(textureSize);
		LogManager::Log("IUM  %dx%d: %zu/%zu texels covered (%.1f%%)",
			textureSize, textureSize, iumHits, iumTexels,
			iumTexels ? 100.0 * double(iumHits) / double(iumTexels) : 0.0);

		if (iumHits == 0)
			LogManager::LogWarning("The IUM mask is empty: check that the mesh has a UV unwrap.");

		// ── Depth pass (camera space) ─────────────────────────────────────────
		// Three-quarter view from outside the bounding sphere. World is Z-up,
		// so the up vector is +Z.
		const vec3f dir = normalize(vec3f(1.0f, -1.0f, 0.6f));
		const vec3f eye = centre + dir * (radius * 2.5f);
		const Camera camera(eye, normalize(centre - eye), vec3f(0.0f, 0.0f, 1.0f),
			45.0f, vec2i(textureSize, textureSize));

		Depth_Generator depthGenerator;
		depthGenerator.setTraversable(model);
		depthGenerator.setCamera(camera);
		depthGenerator.needRenderDepth(true);
		depthGenerator.needRenderPosition(true);
		depthGenerator.needRenderNormal(true);
		depthGenerator.render();

		const Depth_Generator::Result depth = depthGenerator.getResult();

		Stats depthStats;
		for (size_t i = 0; i < depth.depthData.size(); ++i)
			if (i < depth.maskData.size() && depth.maskData[i])
				depthStats.add(depth.depthData[i]);

		const size_t pixels = size_t(textureSize) * size_t(textureSize);
		LogManager::Log("Depth %dx%d from (%.3f %.3f %.3f): %zu/%zu pixels hit (%.1f%%)",
			textureSize, textureSize, eye.x, eye.y, eye.z,
			depthStats.valid, pixels,
			pixels ? 100.0 * double(depthStats.valid) / double(pixels) : 0.0);

		if (depthStats.valid == 0)
		{
			LogManager::LogError("The depth pass hit nothing; the camera or the geometry is wrong.");
			return 1;
		}

		LogManager::Log("Depth range: min %.4f, max %.4f, mean %.4f",
			depthStats.minValue, depthStats.maxValue, depthStats.mean());
		LogManager::Log("OK");
		return 0;
	}
	catch (const std::exception& e)
	{
		LogManager::LogError("%s", e.what());
		return 1;
	}
}

} // ::osc
