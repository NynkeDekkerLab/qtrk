// Simple memory tracker for debugging memory leaks
// Completed version of the one posted http://stackoverflow.com/questions/438515/how-to-track-memory-allocations-in-c-especially-new-delete
#ifdef USE_MEMDBG
#include <map>
#include <cstdint>

#include <Windows.h>

#include "dllmacros.h"

void dbgprintf(const char *fmt,...);

struct Allocation {
	const char *srcfile;
	int line;
	size_t size;
};

template<typename T>
struct track_alloc : std::allocator<T> {
    typedef typename std::allocator<T>::pointer pointer;
    typedef typename std::allocator<T>::size_type size_type;

    template<typename U>
    struct rebind {
        typedef track_alloc<U> other;
    };

    track_alloc() {}

    template<typename U>
    track_alloc(track_alloc<U> const& u)
        :std::allocator<T>(u) {}

    pointer allocate(size_type size, 
                     std::allocator<void>::const_pointer = 0) {
        void * p = std::malloc(size * sizeof(T));
        if(p == 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) {
        std::free(p);
    }
};


struct MemDbgMutex {
	HANDLE h;
	int lockCount;

	MemDbgMutex (){ 
		h=CreateMutex(0,FALSE,0); 
		lockCount=0;
	}
	~MemDbgMutex () { CloseHandle(h); }
	void lock() { 
		WaitForSingleObject(h, INFINITE);
		lockCount++;
	}
	void unlock() { 
		lockCount--;
		ReleaseMutex(h); 
	}
};


struct AllocMap {
	typedef 	std::map< void*, Allocation, std::less<void*>, 
                  track_alloc< std::pair<void* const, std::size_t> > > MapType;
	MapType map;


	MemDbgMutex mutex;
	void Store(void *mem, const Allocation& alloc) {
		mutex.lock();
		map[mem] = alloc;
		mutex.unlock();
	}
	void Remove(void *mem) {
		mutex.lock();
		map.erase(mem);
		mutex.unlock();
	}
	void Print()
	{
		dbgprintf("Allocations: %d\n", map.size());
		size_t total = 0;
		for (auto i = map.begin(); i != map.end(); ++i)
		{
			Allocation& a = i->second;
			dbgprintf("Allocation: %d bytes: @ line %d in '%s'\n" , a.size, a.line, a.srcfile);
			total += a.size;
		}
		dbgprintf("Total: %d bytes\n", total);
	}
};

struct track_printer {
    AllocMap* track;
    track_printer(AllocMap * track):track(track) {}
    ~track_printer()
	{
		track->Print();
	}
};

AllocMap * get_map() {
    // don't use normal new to avoid infinite recursion.
    static AllocMap * track = new (std::malloc(sizeof *track)) AllocMap;
    static track_printer printer(track);
    return track;
}

void * operator new(size_t s, const char* file, int line) {
    // we are required to return non-null
    void * mem = std::malloc(s == 0 ? 1 : s);
    if(mem == 0) {
        throw std::bad_alloc();
    }
	Allocation alloc = { file, line ,s };
    get_map()->Store (mem, alloc);
    return mem;
}
void * operator new[](size_t s, const char* file, int line) {
    // we are required to return non-null
    void * mem = std::malloc(s == 0 ? 1 : s);
    if(mem == 0) {
        throw std::bad_alloc();
    }
	Allocation alloc = { file, line ,s };
    get_map()->Store(mem,alloc);
    return mem;
}

void operator delete(void * mem) {
	get_map()->Remove(mem);
    std::free(mem);
}
void operator delete[](void * mem) {
	get_map()->Remove(mem);
	std::free(mem);
}


#endif
