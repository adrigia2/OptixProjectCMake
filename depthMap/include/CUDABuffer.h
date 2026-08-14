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

#include "optix7.h"
// common std stuff
#include <vector>
#include <assert.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  /*! simple wrapper for creating, and managing a device-side CUDA
      buffer */
  struct CUDABuffer {
    // RAII: default ctor (explicit, because other special members are declared)
    CUDABuffer() = default;

    // Destructor: frees without throwing (a plain cudaFree, not CUDA_CHECK -- a dtor
    // that throws calls std::terminate; at process shutdown the CUDA context may
    // already be gone and cudaFree then returns an error that can be ignored).
    ~CUDABuffer() {
      if (d_ptr) { cudaFree(d_ptr); d_ptr = nullptr; sizeInBytes = 0; }
    }

    // Copying is forbidden: single owner -> no double free from a shallow copy.
    CUDABuffer(const CUDABuffer&)            = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    // Move: transfers ownership and clears the source.
    // noexcept is required, or std::vector copies instead of moving.
    CUDABuffer(CUDABuffer&& o) noexcept
        : sizeInBytes(o.sizeInBytes), d_ptr(o.d_ptr)
    { o.d_ptr = nullptr; o.sizeInBytes = 0; }

    CUDABuffer& operator=(CUDABuffer&& o) noexcept {
      if (this != &o) {
        if (d_ptr) cudaFree(d_ptr);
        d_ptr = o.d_ptr; sizeInBytes = o.sizeInBytes;
        o.d_ptr = nullptr; o.sizeInBytes = 0;
      }
      return *this;
    }

    inline CUdeviceptr d_pointer() const
    { return (CUdeviceptr)d_ptr; }

    //! re-size buffer to given number of bytes
    void resize(size_t size)
    {
      if (d_ptr) free();
      alloc(size);
    }
    
    //! allocate to given number of bytes
    void alloc(size_t size)
    {
      assert(d_ptr == nullptr);
      this->sizeInBytes = size;
      CUDA_CHECK(Malloc( (void**)&d_ptr, sizeInBytes));
    }

    //! free allocated memory
    void free()
    {
      CUDA_CHECK(Free(d_ptr));
      d_ptr = nullptr;
      sizeInBytes = 0;
    }

    template<typename T>
    void alloc_and_upload(const std::vector<T> &vt)
    {
      alloc(vt.size()*sizeof(T));
      upload((const T*)vt.data(),vt.size());
    }
    
    template<typename T>
    void upload(const T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      CUDA_CHECK(Memcpy(d_ptr, (void *)t,
                        count*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    template<typename T>
    void download(T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      CUDA_CHECK(Memcpy((void *)t, d_ptr,
                        count*sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    size_t sizeInBytes { 0 };
    void  *d_ptr { nullptr };
  };

} // ::osc
