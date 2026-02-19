#include "OptixManager.h"
#include "Camera.h"
#include "optix7.h"
#include "LogManager.h"
#include "LaunchParams_DPN.h"
#include "OptixActor.h"

using namespace osc;

//struct FrameResult {
//	std::vector<float> depthData; // Dati della depth map
//	std::string depthFileName; // Nome del file in cui è salvata la depth map
//};

	
class Depth_Generator : public OptixActor
{
	struct Result
	{
		std::vector<float> depthData; // Dati della depth map
		std::vector<vec3f> positionalData; // Dati delle posizioni 3D
		std::vector<vec3f> normalData; // Dati delle normali 3D
		std::vector<uint8_t> maskData; // Maschere di validità per i pixel

		bool hasDepthData() const { return !depthData.empty(); }
		bool hasPositionalData() const { return !positionalData.empty(); }
		bool hasNormalData() const { return !normalData.empty(); }
	};

public:
	void createRaygenPrograms() override;
	void createMissPrograms() override;
	void createHitgroupPrograms() override;
	char* getPtxCode() override;
	void render();

	void needRenderDepth(bool isNeeded);
	void meedRenderPosition(bool isNeeded);
	void needRenderNormal(bool isNeeded);
	void setTraversable(const TriangleMesh& model) override;
	void cleanup() override;


	Depth_Generator()
	{
		createModule();
		createRaygenPrograms();
		createMissPrograms();
		createHitgroupPrograms();
		createSBT();
		createPipeline();
	}
	void setCamera(const Camera& camera, float fovY, vec2i frameSize);

protected:

	// result buffers per la raygen program
    CUDABuffer depthBuffer;  // Aggiunto buffer per depth map
	CUDABuffer positionBuffer; // Già esistente - per la raygen program
	CUDABuffer normalBuffer;   // Già esistente - per la raygen program
	CUDABuffer maskBuffer;     // Aggiunto buffer per maschere di validità

	LaunchParams_DPN launchParams; // Parametri di lancio specifici per depth map
	Result result; // Per memorizzare i risultati della generazione depth map
private:
};