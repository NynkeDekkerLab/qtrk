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

	static cudaImageList<T> alloc(int w, int h, int amount) {
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
			cudaMemcpy2DAsync(dst, sizeof(T)*w, data, pitch, w*sizeof(T), count*h, cudaMemcpyDeviceToHost, s);
		else
			cudaMemcpy2D(dst, sizeof(T)*w, data, pitch, w*sizeof(T), count*h, cudaMemcpyDeviceToHost);
	}
	
	void copyToDevice(T* src, bool async=false, cudaStream_t s=0) {
		if (async)
			cudaMemcpy2DAsync(data, pitch, src, w*sizeof(T), w*sizeof(T), count*h, cudaMemcpyHostToDevice, s);
		else
			cudaMemcpy2D(data, pitch, src, w*sizeof(T), w*sizeof(T), count*h, cudaMemcpyHostToDevice);
	}

	void copyToDevice(T* src, int numImages, bool async=false, cudaStream_t s=0) {
		if (async)
			cudaMemcpy2DAsync(data, pitch, src, w*sizeof(T), w*sizeof(T), numImages*h, cudaMemcpyHostToDevice, s);
		else
			cudaMemcpy2D(data, pitch, src, w*sizeof(T), w*sizeof(T), numImages*h, cudaMemcpyHostToDevice);
	}

	void clear() {
		if(data) cudaMemset2D(data, pitch, 0, w*sizeof(T), count*h);
	}

	CUBOTH int totalNumPixels() { return w*h*count; }
	CUBOTH int totalNumBytes() { return w*h*count*sizeof(T); }
	
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
struct Image4DCudaArray
{
	cudaArray_t array;
	int imgw, imgh;
	int layerw, layerh; // layer width/height in images. (In pixels it would be layerw * imgw)
	int nlayers;
	int numImg; // images per layer

	// CUDA 3D arrays use width in elements, linear memory should use width in bytes.
	// http://stackoverflow.com/questions/10611451/how-to-use-make-cudaextent-to-define-a-cudaextent-correctly
	cudaExtent getExtent() {
		return make_cudaExtent(imgw * layerw, imgh * layerh, nlayers);
	}

	// Properties to be passed to kernels
	struct KernelInst {
		int imgw, imgh;
		int layerw;

		CUBOTH int2 getImagePos(int image) 
		{ 
			return make_int2(imgw * (image % layerw), imgh * (image / layerw));
		}
		
		__device__ T readSurfacePixel(surface<void, cudaSurfaceType2DLayered> surf, int x, int y,int z)
		{
			T r;
			surf2DLayeredread (&r, image_lut_surface, sizeof(T)*x, y, z, cudaBoundaryModeTrap);
			return r;
		}

		__device__ void writeSurfacePixel(surface<void, cudaSurfaceType2DLayered> surf, int x,int y,int z, T value)
		{
			surf2DLayeredwrite(value, surf, sizeof(T)*x, y, z, cudaBoundaryModeTrap);
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
		CheckCUDAError( cudaBindTextureToArray(texref, array, &desc) );
	}
	void unbind(texture<T, cudaTextureType2DLayered, cudaReadModeElementType>& texref) {
		cudaUnbindTexture(texref);
	}

	void bind(surface<void, cudaSurfaceType2DLayered>& surf) {
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		CheckCUDAError( cudaBindSurfaceToArray(surf, array) );
	}
	// there is no unbind surface
	// void unbind(surface<void, cudaSurfaceType2DLayered>& surf) {	}

	Image4DCudaArray(int sx, int sy, int numImg , int sL) {
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

		dbgprintf("creating image4D: %d layers of %d x %d images of %d x %d (%dx%dx%d)\n", 
			sL, layerw, layerh, imgw, imgh, getExtent().width,getExtent().height,getExtent().depth);

		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaError_t err = cudaMalloc3DArray(&array, &desc, getExtent(), cudaArrayLayered | cudaArraySurfaceLoadStore);
		//cudaError_t err = cudaMalloc3DArray(&array, &desc, getExtent(), cudaArraySurfaceLoadStore);
		if (err != cudaSuccess) {
			throw std::bad_alloc(SPrintf("CUDA error during cudaSurf2DList(): %s", cudaGetErrorString(err)).c_str());
		}
	}

	int2 getImagePos(int image) {
		int2 pos = { imgw * ( image % layerw ), imgh * ( image / layerw ) };
		return pos;
	}

	~Image4DCudaArray() {
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
		size_t srcpitch;
		CheckCUDAError( cudaMallocPitch(&d, &srcpitch, sizeof(T)*imgw, imgh) );
		CheckCUDAError( cudaMemset2D(d, srcpitch, 0, sizeof(T)*imgw, imgh) );

		cudaMemcpy3DParms p = {0};
		p.dstArray = array;
		p.extent = make_cudaExtent(imgw,imgh,1);
		p.kind = cudaMemcpyDeviceToDevice;
		p.srcPtr = make_cudaPitchedPtr(d, srcpitch, sizeof(T)*imgw, imgh);
		for (int l=0;l<nlayers;l++)
			for (int img=0;img<numImg;img++) {
				int2 imgpos = getImagePos(img);
				p.dstPos.z = l;
				p.dstPos.x = imgpos.x;
				p.dstPos.y = imgpos.y;
				CheckCUDAError( cudaMemcpy3D(&p) );
			}
		CheckCUDAError( cudaFree(d) );
	}

	// Copy a single subimage to the host
	void copyImageToHost(int img, int layer, T* dst, bool async=false, cudaStream_t s=0) 
	{
		// According to CUDA docs:
		//		The extent field defines the dimensions of the transferred area in elements. 
		//		If a CUDA array is participating in the copy, the extent is defined in terms of that array's elements. 
		//		If no CUDA array is participating in the copy then the extents are defined in elements of unsigned char.

		cudaMemcpy3DParms p = {0};
		p.srcArray = array;
		p.extent = make_cudaExtent(imgw,imgh,1);
		p.kind = cudaMemcpyDeviceToHost;
		p.srcPos.z = layer;
		int2 imgpos = getImagePos(img);
		p.srcPos.x = imgpos.x;
		p.srcPos.y = imgpos.y;
		p.dstPtr = make_cudaPitchedPtr(dst, sizeof(T)*imgw, sizeof(T)*imgw, imgh);
		if (async)
			CheckCUDAError( cudaMemcpy3DAsync(&p, s) );
		else
			CheckCUDAError( cudaMemcpy3D(&p) );
	}

	void copyImageToDevice(int img, int layer, T* src, bool async=false, cudaStream_t s=0)
	{
		// Memcpy3D needs the right pitch for the source, so we first need to copy it to 2D pitched memory before moving the data to the cuda array
//		cudaMallocPitch(

		cudaMemcpy3DParms p = {0};
		p.dstArray = array;
		int2 imgpos = getImagePos(img);

		//The srcPos and dstPos fields are optional offsets into the source and destination objects and are defined in units of each object's elements. 
		// The element for a host or device pointer is assumed to be unsigned char. For CUDA arrays, positions must be in the range [0, 2048) for any dimension. 
		p.dstPos.z = layer;
		p.dstPos.x = imgpos.x;
		p.dstPos.y = imgpos.y;
		p.extent = make_cudaExtent(imgw,imgh,1);
		p.kind = cudaMemcpyHostToDevice;
		p.srcPtr = make_cudaPitchedPtr(src, sizeof(T)*imgw, sizeof(T)*imgw, imgh);
		if (async)
			CheckCUDAError( cudaMemcpy3DAsync(&p, s) );
		else
			CheckCUDAError( cudaMemcpy3D(&p) );
	}

	void free() {
		if (array) {
			CheckCUDAError( cudaFreeArray(array) );
			array = 0;
		}
	}
};



template<typename T>
class Image4DMemory
{
public:
	struct KernelParams {
		T* d_data;
		size_t pitch;
		int cols;
		int depth;
		int imgw, imgh;

		CUBOTH int2 GetImagePos(int z, int l) {
			int img = z+depth*l;
			return make_int2( (img % cols) * imgw, (img / cols) * imgh);
		}
	};

	KernelParams kp;
	int layers, totalImg;
	int rows;

	Image4DMemory(int w, int h, int d, int L) {
		kp.imgh = h;
		kp.imgw = w;
		kp.depth = d;
		layers = L;
		totalImg = d*L;

		rows = 2048 / kp.imgh;
		kp.cols = (totalImg + rows - 1) / rows;

		CheckCUDAError( cudaMallocPitch (&kp.d_data, &kp.pitch, sizeof(T) * kp.cols * kp.imgw, rows * kp.imgh) );
	}

	~Image4DMemory() {
		free();
	}
	void free(){
		if(kp.d_data) cudaFree(kp.d_data);
		kp.d_data=0;
	}

	void copyToDevice(T* src, bool async=false, cudaStream_t s=0) 
	{
		for (int L=0;L<layers;L++) {
			for (int i=0;i<kp.depth;i++)
				copyImageToDevice(i, L, &src[ kp.imgw * kp.imgh * ( kp.depth * L + i ) ], async, s);
		}
	}

	void copyToHost(T* dst, bool async=false, cudaStream_t s=0)
	{
		for (int L=0;L<layers;L++) {
			for (int i=0;i<kp.depth;i++)
				copyImageToHost(i, L, &dst[ kp.imgw * kp.imgh * ( kp.depth * L + i ) ], async, s);
		}
	}


	void clear()
	{
		cudaMemset2D(kp.d_data, kp.pitch, 0, sizeof(T)*(kp.cols*kp.imgw), rows*kp.imgh);
	}

	float* getImgAddr(int2 imgpos)
	{
		char* d = (char*)kp.d_data;
		d += imgpos.y * kp.pitch;
		return &((float*)d)[imgpos.x];
	}

	// Copy a single subimage to the host
	void copyImageToHost(int z, int l, T* dst, bool async=false, cudaStream_t s=0) 
	{
		int2 imgpos = kp.GetImagePos(z, l);
		if (async)
			cudaMemcpy2DAsync(dst, sizeof(T)*kp.imgw, getImgAddr(imgpos), kp.pitch, kp.imgw * sizeof(T), kp.imgh, cudaMemcpyDeviceToHost, s);
		else
			cudaMemcpy2D(dst, sizeof(T)*kp.imgw, getImgAddr(imgpos), kp.pitch, kp.imgw * sizeof(T), kp.imgh, cudaMemcpyDeviceToHost);
	}

	void copyImageToDevice(int z, int l, T* src, bool async=false, cudaStream_t s=0)
	{
		int2 imgpos = kp.GetImagePos(z, l);
		if (async)
			cudaMemcpy2DAsync(getImgAddr(imgpos), kp.pitch, src, sizeof(T)*kp.imgw, sizeof(T)*kp.imgw, kp.imgh, cudaMemcpyHostToDevice, s);
		else
			cudaMemcpy2D(getImgAddr(imgpos), kp.pitch, src, sizeof(T)*kp.imgw, sizeof(T)*kp.imgw, kp.imgh, cudaMemcpyHostToDevice);
	}

	// no binding required
	KernelParams bind() { return kp; }
	void unbind() {}

	static __device__ T read(const KernelParams& kp, int x, int y, int2 imgpos) {
		return ((T*)( (char*)kp.d_data + (y + imgpos.y) * kp.pitch))[ x + imgpos.x ];
	}
	static __device__ void write(T value, const KernelParams& kp, int x, int y, int2 imgpos) {
		((T*)( (char*)kp.d_data + (y + imgpos.y) * kp.pitch)) [ x + imgpos.x ] = value;
	}
};


