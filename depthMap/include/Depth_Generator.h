#include "OptixManager.h"
#include "Camera.h"
#include "optix7.h"
#include "LogManager.h"
#include "LaunchParams_DPN.h"

using namespace osc;

//struct FrameResult {
//	std::vector<float> depthData; // Dati della depth map
//	std::string depthFileName; // Nome del file in cui è salvata la depth map
//};
	
class Depth_Generator : public OptixActor
{
public:
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();
	void render();

	void needRenderDepth(bool isNeeded);
	void meedRenderPosition(bool isNeeded);
	void needRenderNormal(bool isNeeded);
	Depth_Generator()
	{
		createRaygenPrograms();
		createMissPrograms();
		createHitgroupPrograms();
	}

	void setTraversable(OptixTraversableHandle& gas);
	void setCamera(const Camera& camera, float fovY);


protected:

	// result buffers per la raygen program
    CUDABuffer depthBuffer;  // Aggiunto buffer per depth map
	CUDABuffer positionBuffer; // Già esistente - per la raygen program
	CUDABuffer normalBuffer;   // Già esistente - per la raygen program

	LaunchParams_DPN launchParams; // Parametri di lancio specifici per depth map
	CUDABuffer launchParamsBuffer; // Buffer per i parametri di lancio
private:
};