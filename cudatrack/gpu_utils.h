#pragma once
// Simple device side vector implementation. 
#include <cuda_runtime.h>
#include <vector>
#include <cstdarg>

#define CUDA_SUPPORTED_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

#define CUBOTH __device__ __host__

inline void CheckCUDAError(cudaError_t err)
{
	if (err != cudaSuccess) {
		const char* errstr = cudaGetErrorString(err);
		throw std::runtime_error(SPrintf("CUDA error: %s\n" ,errstr).c_str());
	}
}

inline void CheckCUDAError()
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const char* errstr = cudaGetErrorString(err);
		dbgprintf("CUDA error: %s\n" ,errstr);
	}
}

#ifdef _DEBUG
inline void dbgCUDAErrorCheck(cudaError_t e) { CheckCUDAError(e); }
#else
inline void dbgCUDAErrorCheck(cudaError_t e) {}
#endif

template<typename T>
class device_vec {
public:
	device_vec() {
		data = 0;
		size = 0;
	}

	device_vec(size_t N) { 
		data = 0;
		size = 0;
		init(N);
	}
	device_vec(const device_vec<T>& src) {
		data = 0; size = 0;
		init(src.size);
		dbgCUDAErrorCheck(cudaMemcpy(data, src.data, sizeof(T)*size, cudaMemcpyDeviceToDevice));
	}
	device_vec(const std::vector<T>& src) {
		data=0; size=0; 
		init(src.size());
		dbgCUDAErrorCheck(cudaMemcpy(data, &src[0], sizeof(T)*size, cudaMemcpyHostToDevice));
	}
	~device_vec(){
		free();
	}
	void init(size_t s) {
		if(size != s) {
			free();
		}
		if (s!=0) {
			if (cudaMalloc(&data, sizeof(T)*s) != cudaSuccess) {
				throw std::bad_alloc(SPrintf("device_vec<%s> init %d elements failed", typeid(T).name(), s).c_str());
			}
			size = s;
		}
	}
	void free() {
		if (data) {
			dbgCUDAErrorCheck(cudaFree(data));
			data=0;
		}
	}
	operator std::vector<T>() const {
		std::vector<T> dst(size);
		dbgCUDAErrorCheck(cudaMemcpy(&dst[0], data, sizeof(T)*size, cudaMemcpyDeviceToHost));
		return dst;
	}
	device_vec<T>& operator=(const std::vector<T>& src) {
		init(src.size());
		dbgCUDAErrorCheck(cudaMemcpy(data, &src[0], sizeof(T)*size, cudaMemcpyHostToDevice));
		return *this;
	}
	device_vec<T>& operator=(const device_vec<T>& src) {
		clear();
		init(src.size);
		dbgCUDAErrorCheck(cudaMemcpy(data, src.data, sizeof(T)*size, cudaMemcpyDeviceToDevice));
		return *this;
	}
	void copyToHost(T* dst, bool async, cudaStream_t s=0) {
		if (async)
			dbgCUDAErrorCheck(cudaMemcpyAsync(dst, data, sizeof(T) * size, cudaMemcpyDeviceToHost, s));
		else
			dbgCUDAErrorCheck(cudaMemcpy(dst, data, sizeof(T) * size, cudaMemcpyDeviceToHost));
	}
	void copyToHost(std::vector<T>& dst ,bool async, cudaStream_t s=0) {
		if (dst.size() != size)
			dst.resize(size);
		copyToHost(&dst[0], async, s);
	}
	void copyToDevice(const std::vector<T>& src, bool async=false, cudaStream_t s=0) {
		copyToDevice(&src[0], src.size(), async, s);
	}
	void copyToDevice(const T* first, size_t size, bool async=false, cudaStream_t s=0) {
		if (this->size < size)
			init(size);
		if (async)
			dbgCUDAErrorCheck(cudaMemcpyAsync(data, first, sizeof(T) * size, cudaMemcpyHostToDevice, s));
		else
			dbgCUDAErrorCheck(cudaMemcpy(data, first, sizeof(T) * size, cudaMemcpyHostToDevice));
	}
	// debugging util. Be sure to synchronize before
	std::vector<T> toVector() {
		std::vector<T> v (size);
		dbgCUDAErrorCheck(cudaMemcpy(&v[0], data, sizeof(T)*size, cudaMemcpyDeviceToHost));
		return v;
	}
	size_t memsize() { return size*sizeof(T); }
	size_t size;
	T* data;
};



#if 1 //defined(_DEBUG)
struct MeasureTime {
	uint64_t freq, time;
	const char* name;
	MeasureTime(const char *name) {
		QueryPerformanceCounter((LARGE_INTEGER*)&time);
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		this->name=name;
	}
	~MeasureTime() {
		uint64_t time1;
		QueryPerformanceCounter((LARGE_INTEGER*)&time1);
		double dt = (double)(time1-time) / (double)freq;
		dbgprintf("%s: Time taken: %f ms\n", name, dt*1000);
	}
};
#else
struct MeasureTime {
	MeasureTime(const char* name) {} 
};
#endif


template<typename T, int flags=0>
class pinned_array
{
public:
	pinned_array() {
		d=0; n=0;
	}
	~pinned_array() {
		free();
	}
	pinned_array(size_t n) {
		d=0; init(n);
	}
	template<typename TOther, int f>
	pinned_array(const pinned_array<TOther,f>& src) {
		d=0;init(src.n);
		for(int k=0;k<src.n;k++)
			d[k]=src[k];
	}
	template<typename TOther, int F>
	pinned_array& operator=(const pinned_array<TOther, F>& src) {
		if (src.n != n) init(src.n);
		for(int k=0;k<src.n;k++)
			d[k]=src[k];
		return *this;
	}
	template<typename Iterator>
	pinned_array(Iterator first, Iterator end) {
		d=0; init(end-first);
		for (int k = 0; first != end; ++first) {
			d[k++] = *first;
		}
	}
	template<typename T>
	pinned_array(const device_vec<T>& src) {
		d=0; init(src.size()); src.copyToHost(d,false);
	}

	int size() const { return n; }
	T* begin() { return d; }
	T* end() { return d+n; }
	const T* begin() const { return d; }
	const T* end() const { return d+n; }
	T* data() { return d; }
	void free() {
		cudaFreeHost(d);
		d=0;n=0;
	}
	void init(int n) {
		if (d) free();
		this->n = n;
		if (cudaMallocHost(&d, sizeof(T)*n, flags) != cudaSuccess) {
			throw std::bad_alloc(SPrintf("%s init %d elements failed", typeid(*this).name(), n).c_str());
		}
	}
	T& operator[](int i) {  return d[i]; }
	const T&operator[](int i) const { return d[i];}
	size_t memsize() { return n*sizeof(T); }

protected:
	T* d;
	size_t n;
};
