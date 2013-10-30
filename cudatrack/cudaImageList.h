#pragma once

#include "gpu_utils.h"

//cudaImageList stores a large number of small images into a single large memory space, allocated using cudaMallocPitch. 
// It has no constructor/destructor, so it can be passed to CUDA kernels. 
// It allows binding to a texture
// NOTE: Maybe this should be converted into a 3D cudaArray?
template<typename T>
struct cudaImageList {
	// No constructor used to allow passing as CUDA kernel argument
	T* data;
	size_t pitch;
	int w,h;
	int count;

	CUBOTH int fullwidth() { return w; }
	CUBOTH int fullheight() { return h*count; }

	enum { MaxImageWidth = 8192 };

	CUBOTH int capacity() { return count; }
	CUBOTH int numpixels() { return w*h*count; }

	static cudaImageList<T> emptyList() {
		cudaImageList imgl;
		imgl.data = 0;
		imgl.pitch = 0;
		imgl.w = imgl.h = 0;
		imgl.count = 0;
		return imgl;
	}

	CUBOTH bool isEmpty() { return data==0; }

	static cudaImageList<T> alloc(int w,int h, int amount) {
		cudaImageList imgl;
		imgl.w = w; imgl.h = h;
		imgl.count = amount;

		if (cudaMallocPitch(&imgl.data, &imgl.pitch, sizeof(T)*imgl.fullwidth(), imgl.fullheight()) != cudaSuccess) {
			throw std::bad_alloc(SPrintf("cudaImageListf<%s> alloc %dx%dx%d failed", typeid(T).name(), w, h, amount).c_str());
		}
		return imgl;
	}

	template<int Flags>
	void allocateHostImageBuffer(pinned_array<T, Flags>& hostImgBuf) {
		hostImgBuf.init( numpixels() );
	}

	CUBOTH T* get(int i) {
		return (T*)(((char*)data) + pitch*h*i);
	}

	CUBOTH T pixel_oobcheck(int x,int y, int imgIndex, T border=0.0f) {
		if (x < 0 || x >= w || y < 0 || y >= h)
			return border;

		computeImagePos(x,y,imgIndex);
		T* row = (T*) ( (char*)data + y*pitch );
		return row[x];
	}

	CUBOTH T& pixel(int x,int y, int imgIndex) {
		computeImagePos(x,y,imgIndex);
		T* row = (T*) ( (char*)data + y*pitch );
		return row[x];
	}

	CUBOTH T* pixelAddress(int x,int y, int imgIndex) {
		computeImagePos(x,y,imgIndex);
		T* row = (T*) ( (char*)data + y*pitch );
		return row + x;
	}

	
	// Returns true if bounds are crossed
	CUBOTH bool boundaryHit(float2 center, float radius)
	{
		return center.x + radius >= w ||
			center.x - radius < 0 ||
			center.y + radius >= h ||
			center.y - radius < 0;
	}


	void free()
	{
		if(data) {
			cudaFree(data);
			data=0;
		}
	}

	// Copy a single subimage to the host
	void copyImageToHost(int img, T* dst, bool async=false, cudaStream_t s=0) {
		T* src = pixelAddress (0,0, img); 

		if (async)
			cudaMemcpy2DAsync(dst, sizeof(T)*w, src, pitch, w*sizeof(T), h, cudaMemcpyDeviceToHost, s);
		else
			cudaMemcpy2D(dst, sizeof(T)*w, src, pitch, w*sizeof(T), h, cudaMemcpyDeviceToHost);
	}
	// Copy a single subimage to the device
	void copyImageToDevice(int img, T* src, bool async=false, cudaStream_t s=0) {
		T* dst = pixelAddress (0,0, img); 

		if (async)
			cudaMemcpy2DAsync(dst, pitch, src, w*sizeof(T), w*sizeof(T), h, cudaMemcpyHostToDevice, s);
		else
			cudaMemcpy2D(dst, pitch, src, w*sizeof(T), w*sizeof(T), h, cudaMemcpyHostToDevice);
	}

	void copyToHost(T* dst, bool async=false, cudaStream_t s=0) {
		if (async)
			cudaMemcpy2DAsync(dst, sizeof(T)*w, data, pitch, w*sizeof(T), count*h, cudaMemcpyDeviceToHost);
		else
			cudaMemcpy2D(dst, sizeof(T)*w, data, pitch, w*sizeof(T), count*h, cudaMemcpyDeviceToHost);
	}
	
	void copyToDevice(T* src, bool async=false, cudaStream_t s=0) {
		if (async)
			cudaMemcpy2DAsync(data, pitch, src, w*sizeof(T), w*sizeof(T), count*h, cudaMemcpyHostToDevice);
		else
			cudaMemcpy2D(data, pitch, src, w*sizeof(T), w*sizeof(T), count*h, cudaMemcpyHostToDevice);
	}

	void copyToDevice(T* src, int numImages, bool async=false, cudaStream_t s=0) {
		if (async)
			cudaMemcpy2DAsync(data, pitch, src, w*sizeof(T), w*sizeof(T), numImages*h, cudaMemcpyHostToDevice);
		else
			cudaMemcpy2D(data, pitch, src, w*sizeof(T), w*sizeof(T), numImages*h, cudaMemcpyHostToDevice);
	}

	void clear() {
		if(data) cudaMemset2D(data, pitch, 0, w*sizeof(T), count*h);
	}

	CUBOTH int totalNumPixels() { return pitch*h*count; }
	CUBOTH int totalNumBytes() { return pitch*h*count*sizeof(T); }
	
	CUBOTH static inline T interp(T a, T b, float x) { return a + (b-a)*x; }

	CUBOTH T interpolate(float x,float y, int idx, bool &outside)
	{
		int rx=x, ry=y;

		if (rx < 0 || ry < 0 || rx >= w-1 || ry >= h-1) {
			outside=true;
			return 0.0f;
		}

		T v00 = pixel(rx, ry, idx);
		T v10 = pixel(rx+1, ry, idx);
		T v01 = pixel(rx, ry+1, idx);
		T v11 = pixel(rx+1, ry+1, idx);

		T v0 = interp (v00, v10, x-rx);
		T v1 = interp (v01, v11, x-rx);

		outside=false;
		return interp (v0, v1, y-ry);
	}

	void bind(texture<T, cudaTextureType2D, cudaReadModeElementType>& texref) {
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaBindTexture2D(NULL, &texref, data, &desc, w, h*count, pitch);
	}
	void unbind(texture<T, cudaTextureType2D, cudaReadModeElementType>& texref) {
		cudaUnbindTexture(&texref);
	}

	CUBOTH void computeImagePos(int& x, int& y, int idx)
	{
		y += idx * h;
	}

	// Using the texture cache can result in significant speedups
	__device__ T interpolateFromTexture(texture<T, cudaTextureType2D, cudaReadModeElementType> texref, float x,float y, int idx, bool& outside)
	{
		int rx=x, ry=y;

		if (rx < 0 || ry < 0 || rx >= w-1 || ry >= h-1) {
			outside=true;
			return 0.0f;
		}

		computeImagePos(rx, ry, idx);

		float fx=x-floor(x), fy = y-floor(y);
		float u = rx + 0.5f;
		float v = ry + 0.5f;

		T v00 = tex2D(texref, u, v);
		T v10 = tex2D(texref, u+1, v);
		T v01 = tex2D(texref, u, v+1);
		T v11 = tex2D(texref, u+1, v+1);

		T v0 = interp (v00, v10, fx);
		T v1 = interp (v01, v11, fx);

		outside = false;
		return interp (v0, v1, fy);
	}
};



// 4D image, implemented by having layers with each a grid of 2D images.
template<typename T>
struct cudaImage4D
{
	cudaArray_t array;
	int imgw, imgh;
	int layerw, layerh; // layer width/height in images. (In pixels it would be layerw * imgw)
	int nlayers;
	int numImg; // images per layer

	cudaExtent getExtent() {
		return make_cudaExtent(imgw * layerw, imgh * layerh, nlayers);
	}

	// Properties to be passed to kernels
	struct KernelInst {
		int imgw, imgh;
		int layerw;

		CUBOTH int2 getImagePos(int image) { return make_int2(imgw * (image % layerw), imgh * (image / layerw)); }
		
		CUBOTH T readSurfacePixel(surface<void, cudaSurfaceType2DLayered> surf, int x, int y,int z)
		{
			T r;
			surf2DLayeredread (&r, image_lut_surface, sizeof(T)*x, y, z, cudaBoundaryModeTrap);
			return r;
		}

		CUBOTH void writeSurfacePixel(surface<void, cudaSurfaceType2DLayered> surf, int x,int y,int z, T value)
		{
			surf2DLayeredwrite(value, image_lut_surface, sizeof(T)*x, y, z, cudaBoundaryModeTrap);
		}
	};

	KernelInst kernelInst() {
		KernelInst inst;
		inst.imgw = imgw; inst.imgh = imgh;
		inst.layerw = layerw;
		return inst;
	}

	void bind(texture<T, cudaTextureType2DLayered, cudaReadModeElementType>& texref) {
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaBindTextureToArray(texref, array, &desc);
	}
	void unbind(texture<T, cudaTextureType2DLayered, cudaReadModeElementType>& texref) {
		cudaUnbindTexture(texref);
	}

	void bind(surface<void, cudaSurfaceType2DLayered>& surf) {
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaBindSurfaceToArray(surf, array);
	}
	// there is no unbind surface
	// void unbind(surface<void, cudaSurfaceType2DLayered>& surf) {	}

	cudaImage4D(int sx, int sy, int numImg , int sL) {
		array = 0;
		int d;
		cudaGetDevice(&d);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, d);
		
		imgw = sx;
		imgh = sy;
		this->numImg = numImg;

//		layerh = (int)(prop.maxSurface2DLayered[1] / imgh);
		layerh = 2048 / imgh;
		layerw = (numImg + layerh - 1) / layerh;
		nlayers = sL;

		dbgprintf("creating image4D: %d layers of %d x %d images of %d x %d (%dx%dx%d)", 
			sL, layerw, layerh, imgw, imgh, getExtent().width,getExtent().height,getExtent().depth);

		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaError_t err = cudaMalloc3DArray(&array, &desc, getExtent(), cudaArrayLayered | cudaArraySurfaceLoadStore);
		if (err != cudaSuccess) {
			throw std::bad_alloc(SPrintf("CUDA error during cudaSurf2DList(): %s", cudaGetErrorString(err)).c_str());
		}
	}

	int2 getImagePos(int image) {
		int2 pos = { image % layerw , image / layerw };
		return pos;
	}

	~cudaImage4D() {
		free();
	}

	void copyToDevice(T* src, bool async=false, cudaStream_t s=0) 
	{
		for (int L=0;L<nlayers;L++) {
			for (int i=0;i<numImg;i++)
				copyImageToDevice(i, L, &src[ imgw * imgh * ( numImg * L + i ) ], async, s);
		}
	}

	void copyToHost(T* dst, bool async=false, cudaStream_t s=0)
	{
		for (int L=0;L<nlayers;L++) {
			for (int i=0;i<numImg;i++)
				copyImageToHost(i, L, &dst[ imgw * imgh * ( numImg * L + i ) ], async, s);
		}
	}

	void clear()
	{
		// create a new black image in device memory and use to it clear all the layers
		T* d;
		cudaMalloc(&d, sizeof(T)*imgw*imgh);
		cudaMemset(d, 0, sizeof(T)*imgw*imgh);

		cudaMemcpy3DParms p = {0};
		p.dstArray = array;
		p.extent = make_cudaExtent(imgw,imgh,1);
		p.kind = cudaMemcpyDeviceToDevice;
		p.srcPtr = make_cudaPitchedPtr(d, sizeof(T)*imgw, imgw, imgh);
		for (int l=0;l<nlayers;l++)
			for (int img=0;img<numImg;img++) {
				int2 imgpos = getImagePos(img);
				p.dstPos.z = l;
				p.dstPos.x = imgpos.x;
				p.dstPos.y = imgpos.y;
				cudaMemcpy3D(&p);
			}
		cudaFree(d);
	}

	// Copy a single subimage to the host
	void copyImageToHost(int img, int layer, T* dst, bool async=false, cudaStream_t s=0) 
	{
		cudaMemcpy3DParms p = {0};
		p.srcArray = array;
		p.extent = make_cudaExtent(imgw,imgh,1);
		p.kind = cudaMemcpyDeviceToHost;
		p.srcPos.z = layer;
		int2 imgpos = getImagePos(img);
		p.srcPos.x = imgpos.x;
		p.srcPos.y = imgpos.y;
		p.dstPtr = make_cudaPitchedPtr(dst, sizeof(T)*imgw, imgw, imgh);
		if (async)
			cudaMemcpy3DAsync(&p, s);
		else
			cudaMemcpy3D(&p);
	}

	void copyImageToDevice(int img, int layer, T* src, bool async=false, cudaStream_t s=0)
	{
		cudaMemcpy3DParms p = {0};
		p.dstArray = array;
		int2 imgpos = getImagePos(img);
		p.dstPos.z = layer;
		p.dstPos.x = imgpos.x;
		p.dstPos.y = imgpos.y;
		p.extent = make_cudaExtent(imgw,imgh,1);
		p.kind = cudaMemcpyHostToDevice;
		p.srcPtr = make_cudaPitchedPtr(src, sizeof(T)*imgw, imgw, imgh);
		if (async)
			cudaMemcpy3DAsync(&p, s);
		else
			cudaMemcpy3D(&p);
	}

	void free() {
		if (array) {
			cudaFreeArray(array);
			array = 0;
		}
	}
};

