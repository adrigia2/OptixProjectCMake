#pragma once

#include "gdt/math/AffineSpace.h"
#include "objLoader/tiny_obj_loader.h"
#include "CUDABuffer.h"
#include <LogManager.h>

using namespace osc;

class OptixManager {
protected:
    CUcontext          cudaContext{};
    CUstream           stream{};
    cudaDeviceProp     deviceProps{};

    OptixDeviceContext optixContext{};

    void initOptix();
    void createContext();

    // costruttore privato/protetto: impedisce creazione esterna
    OptixManager() {

        initOptix();
        createContext();
    }

public:
    // Global access to the instance
    static OptixManager& instance() {
        static OptixManager inst;   // constructed once, thread-safe (C++11 and later)
        return inst;
    }

    // No copying or assignment
    OptixManager(const OptixManager&) = delete;
    OptixManager& operator=(const OptixManager&) = delete;

    // Nor moving
    OptixManager(OptixManager&&) = delete;
    OptixManager& operator=(OptixManager&&) = delete;

    void render(int launchWidth, int launchHeight, int launchDepth, OptixPipeline& pipeline, CUDABuffer& launchParamsBuffer, OptixShaderBindingTable& sbt);

    void printStatus();
    void cleanup();
	void setLogLevel(LogManager::Level level);

    OptixDeviceContext& getContext() { return optixContext; }

    ~OptixManager() {
        cleanup();
    }
};
