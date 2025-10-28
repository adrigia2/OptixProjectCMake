// OptixProjectCMake.cpp: definisce il punto di ingresso dell'applicazione.
//

#include "OptixProjectCMake.h"
using namespace std;

void initOptix()
{
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK(optixInit());
}


/*! main entry point to this example - initially optix, print hello
  world, then exit */
extern "C" int main(int ac, char** av)
{
    try {
        std::cout << "#osc: initializing optix..." << std::endl;

        initOptix();

        std::cout 
            << "#osc: successfully initialized optix... yay!"
            << std::endl;

        // for this simple hello-world example, don't do anything else
        // ...
        std::cout << "#osc: done. clean exit." << std::endl;

    }
    catch (std::runtime_error& e) {
        std::cout << "FATAL ERROR: " << e.what()
            << std::endl;
        exit(1);
    }
    return 0;
}