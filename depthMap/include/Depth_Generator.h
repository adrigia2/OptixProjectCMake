#include "OptixManager.h"
#include "Camera.h"
#include "optix7.h"
#include "LogManager.h"
#include "LaunchParams_DPN.h"
#include "OptixActor.h"

using namespace osc;

//struct FrameResult {
//	std::vector<float> depthData; // depth map data
//	std::string depthFileName; // name of the file the depth map is saved to
//};

	
class Depth_Generator : public OptixActor
{
	

public:

	struct Result
	{
		std::vector<float> depthData; // depth map data
		std::vector<vec3f> positionalData; // 3D positions
		std::vector<vec3f> normalData; // 3D normals
		std::vector<uint8_t> maskData; // per-pixel validity mask

		bool hasDepthData() const { return !depthData.empty(); }
		bool hasPositionalData() const { return !positionalData.empty(); }
		bool hasNormalData() const { return !normalData.empty(); }
	};

	void createRaygenPrograms() override;
	void createMissPrograms() override;
	void createHitgroupPrograms() override;
	char* getPtxCode() override;
	void render();

	void needRenderDepth(bool isNeeded);
	void needRenderPosition(bool isNeeded);
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
	void setCamera(const Camera& camera);

	Result getResult() const { return result; }

protected:

	// result buffers for the raygen program
    CUDABuffer depthBuffer;
	CUDABuffer positionBuffer;
	CUDABuffer normalBuffer;
	CUDABuffer maskBuffer;

	LaunchParams_DPN launchParams; // launch parameters of the depth pass
	Result result;                 // results of the depth pass
private:
};