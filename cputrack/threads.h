#pragma once
#include <list>
// Thread OS related code is abstracted into a simple "Threads" struct
#ifdef USE_PTHREADS

#include "pthread.h"

struct Threads
{
	struct Handle;

	pthread_attr_t joinable_attr;

	struct Mutex {
		pthread_mutex_t h;
		Mutex() { pthread_mutex_init(&h, 0);  }
		~Mutex() { pthread_mutex_destroy(&h);  }
		void lock() { 
			pthread_mutex_lock(&h); }
		void unlock() { pthread_mutex_unlock(&h); }
	};

	static Handle* Create(DWORD (WINAPI *method)(void* param), void* param) {
		pthread_t h;
		pthread_attr_t joinable_attr;
		pthread_attr_init(&joinable_attr);
		pthread_attr_setdetachstate(&joinable_attr, PTHREAD_CREATE_JOINABLE);
		pthread_create(&h, &joinable_attr, method, param);
		if (!h) {
			throw std::runtime_error("Failed to create processing thread.");
		}

		pthread_attr_destroy(&joinable_attr);
		return (Handle*)h;
	}

	static void WaitAndClose(Handle* h) {
		pthread_join((pthread_t)h, 0);
	}
};


#else

#include <Windows.h>
#undef AddJob
#undef Sleep
#undef max
#undef min

struct Threads
{
	typedef void (*ThreadEntryPoint)(void* param);
	struct Handle {
		DWORD threadID;
		ThreadEntryPoint callback;
		HANDLE winhdl;
		void* param;
	};

	struct Mutex {
		HANDLE h;
		std::string name;
		bool trace;
		int lockCount;

		Mutex(const char*name=0) : name(name?name:"") { 
			msg("create"); 
			h=CreateMutex(0,FALSE,0); 
			trace=false;
			lockCount=0;
		}
		~Mutex() { msg("end");  CloseHandle(h); }
		void lock() { 
			msg("lock"); 
			WaitForSingleObject(h, INFINITE);
			lockCount++;
		}
		void unlock() { 
			msg("unlock"); 
			lockCount--;
			ReleaseMutex(h); 
		}
		void msg(const char* m) {
			if(name.length()>0 && trace) {
				char buf[32];
				SNPRINTF(buf, sizeof(buf), "mutex %s: %s\n", name.c_str(), m);
				OutputDebugString(buf);
			}
		}
	};

	static DWORD WINAPI ThreadCaller (void *param) {
		Handle* hdl = (Handle*)param;
		hdl->callback (hdl->param);
		return 0;
	}

	static Handle* Create(ThreadEntryPoint method, void* param) {
		Handle* hdl = new Handle;
		hdl->param = param;
		hdl->callback = method;
		hdl->winhdl = CreateThread(0, 0, ThreadCaller, hdl, 0, &hdl->threadID);
		
		if (!hdl->winhdl) {
			throw std::runtime_error("Failed to create processing thread.");
		}
		return hdl;
	}

	static bool RunningVistaOrBetter ()
	{
		OSVERSIONINFO v;
		GetVersionEx(&v);
		return v.dwMajorVersion >= 6;
	}

	static void SetBackgroundPriority(Handle* thread, bool bg)
	{
		HANDLE h = (HANDLE)thread;
		// >= Windows Vista
		if (RunningVistaOrBetter())
			SetThreadPriority(h, bg ? THREAD_MODE_BACKGROUND_BEGIN : THREAD_MODE_BACKGROUND_END);
		else
			SetThreadPriority(h, bg ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_NORMAL);
	}

	static void WaitAndClose(Handle* h) {
		WaitForSingleObject(h->winhdl, INFINITE);
		CloseHandle(h->winhdl);
		delete h;
	}

	static void Sleep(int ms) {
		::Sleep(ms);
	}

	static int GetCPUCount() {
		// preferably 
		#ifdef WIN32
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		return sysInfo.dwNumberOfProcessors;
		#else
		return 4;
		#endif
	}
};

typedef Threads::Handle ThreadHandle;


#endif

template<typename T> 
class Atomic {
	mutable Threads::Mutex m;
	T data;
public:
	Atomic(const T& o=T()) : data(o) {} // no need for locking: the object is not allowed to be used before the constructor is done anyway
	operator T() const { return get(); };
	Atomic& operator=(const T& x) { set(x); return *this; }
	void set(const T& x) {
		m.lock();
		data=x;
		m.unlock();
	}
	T get() const {
		m.lock();
		T x=data;
		m.unlock();
		return x;
	}
};

template<typename TWorkItem, typename TFunctor>
class ThreadPool {
public:
	ThreadPool(TFunctor f, int Nthreads=-1) : worker(f) {
		if (Nthreads<0) 
			Nthreads = Threads::GetCPUCount();
		threads.resize(Nthreads);
		quit=false;
		inProgress=0;
		for (int i=0;i<Nthreads;i++)
			threads[i]=Threads::Create(&ThreadEntryPoint,this);
	}
	~ThreadPool() {
		Quit();
	}
	void ProcessArray(TWorkItem* items, int n) {
		for(int i=0;i<n;i++)
			AddWork(items[i]);
	}
	void AddWork(TWorkItem w) {
		workMutex.lock();
		work.push_back(w);
		workMutex.unlock();
	}
	void WaitUntilDone() {
		while(!IsDone()) Threads::Sleep(1);
	}
	bool IsDone() {
		workMutex.lock();
		bool r=work.empty() && inProgress==0;
		workMutex.unlock();
		return r;
	}
	void Quit() {
		quit=true;
		for(uint i=0;i<threads.size();i++)
			Threads::WaitAndClose(threads[i]);
		threads.clear();
	}
protected:
	static void ThreadEntryPoint(void *param) {
		ThreadPool* pool = ( ThreadPool *)param;
		TWorkItem item;
		while (!pool->quit) {
			if ( pool->GetNewItem(item) ) {
				pool->worker(item);
				pool->ItemDone();
			} else Threads::Sleep(1);
		}
	}
	void ItemDone() {
		workMutex.lock();
		inProgress--;
		workMutex.unlock();
	}
	bool GetNewItem(TWorkItem& item) {
		workMutex.lock();
		bool r = !work.empty();
		if (r) {
			item = work.front();
			work.pop_front();
			inProgress++;
		}
		workMutex.unlock();
		return r;
	}
	std::vector<Threads::Handle*> threads;
	Threads::Mutex workMutex;
	std::list<TWorkItem> work;
	int inProgress;
	Atomic<bool> quit;
	TFunctor worker;
};

template<typename TF>
void parallel_for(int count, TF f) {

	if (count == 1)
		f(0);
	else {
		ThreadPool<int, TF> threadPool(f, std::min (count, Threads::GetCPUCount()) );
		for (int i=0;i<count;i++) threadPool.AddWork(i);
		threadPool.WaitUntilDone();
	}
}
