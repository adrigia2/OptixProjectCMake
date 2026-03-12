// ======================================================================== //
// Camera Structure - Shared camera definition                             //
// ======================================================================== //

#pragma once

#include "gdt/math/vec.h"

namespace osc {

	using namespace gdt;

	/*! Camera structure for defining viewpoint and orientation */
	struct Camera {
		Camera(const vec3f pos, const vec3f forward, const vec3f up,
			float fovY = 45.0f, const vec2i frameSize = vec2i(1024, 1024))
			: pos(pos), forward(forward), up(up), fovY(fovY), frameSize(frameSize) {
		}

	private:
		/*! camera position - *from* where we are looking */
		vec3f pos;
		/*! which direction we are looking (forward vector) */
		vec3f forward;
		/*! general up-vector */
		vec3f up;
		/*! vertical field of view (in radians or degrees, depending on usage) */
		float fovY;
		/*! frame size in pixels */
		vec2i frameSize;

	public:
		vec3f getPos() const { return pos; }
		vec3f getForward() const { return forward; }
		vec3f getUp() const { return up; }
		float getFovY() const { return fovY; }
		vec2i getFrameSize() const { return frameSize; }

		void setPos(const vec3f newPos) { pos = newPos; }
		void setForward(const vec3f newForward) { forward = newForward; }
		void setUp(const vec3f newUp) { up = newUp; }
		void setFovY(float newFovY) { fovY = newFovY; }
		void setFrameSize(const vec2i newFrameSize) { frameSize = newFrameSize; }
	};

} // namespace osc
