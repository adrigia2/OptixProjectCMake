#pragma once

#include "LaunchParams.h"
#include "gdt/math/AffineSpace.h"
#include "objLoader/tiny_obj_loader.h"
#include "SampleRenderer.h"
#include "CUDABuffer.h"

using namespace osc;

class OptixManager {
protected:
    CUcontext          cudaContext{};
    CUstream           stream{};
    cudaDeviceProp     deviceProps{};

    OptixDeviceContext optixContext{};

    LaunchParams launchParams{};
    CUDABuffer launchParamsBuffer{};

    void initOptix();
    void createContext();

    // costruttore privato/protetto: impedisce creazione esterna
    OptixManager() {
        initOptix();
        createContext();
    }

public:
    // accesso globale all'istanza
    static OptixManager& instance() {
        static OptixManager inst;   // creato una sola volta, thread-safe (C++11+)
        return inst;
    }

    // vieta copia e assegnazione
    OptixManager(const OptixManager&) = delete;
    OptixManager& operator=(const OptixManager&) = delete;

    // opzionale: vieta anche i move
    OptixManager(OptixManager&&) = delete;
    OptixManager& operator=(OptixManager&&) = delete;

    void render(int launchWidth, int launchHeight, int launchDepth, CUDABuffer& launchParams, OptixShaderBindingTable& sbt);

    void printStatus();
    void cleanup();

    OptixDeviceContext& getContext() { return optixContext; }
    LaunchParams& getLaunchParams() { return launchParams; }

    ~OptixManager() {
        cleanup();
    }
};
