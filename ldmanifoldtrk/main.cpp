
#include "std_incl.h"
#include "utils.h"
#include "tinydir.h"
#include <list>
#include "random_distr.h"

typedef void (*process_image_cb)(ImageData *img, int bead, int frame);

std::string file_ext(const char *f){
	int l=strlen(f)-1;
	while (l > 0) {
		if (f[l] == '.')
			return &f[l+1];
		l--;
	}
	return "";
}

struct BeadPos{ int x,y; };

std::vector<BeadPos> read_beadlist(std::string fn)
{
	FILE *f = fopen(fn.c_str(), "r");
	std::vector<BeadPos> beads;
	while (!feof(f)) {
		BeadPos bp;
		fscanf(f, "%d\t%d\n", &bp.x,&bp.y);
		beads.push_back(bp);
	}

	fclose(f);
	return beads;
}

void extract_regions(std::vector<BeadPos> beads, int size, int frame, ImageData* img, process_image_cb cb)
{
	ImageData roi = ImageData::alloc(size,size);

	for (int i=0;i<beads.size();i++)
	{
		int xs = beads[i].x - size/2;
		int ys = beads[i].y - size/2;

		for (int y=0;y<size;y++) {
			for(int x=0;x<size;x++)
				roi.at(x,y) = img->at(xs+x, ys+y);
		}

		if(cb) cb(&roi, i, frame);
	}
	roi.free();
}

void process_beads(const char *path, int size, std::vector<BeadPos> beadlist, process_image_cb cb, int framelimit)
{
	tinydir_dir d;
	if (tinydir_open(&d, path) == -1)
		throw std::runtime_error("can't open given path");

	int frame=0;
	while(d.has_next) {
		tinydir_file f;
		tinydir_readfile(&d, &f);

		if (!f.is_dir && file_ext(f.name) == "jpg") {
			dbgprintf("File: %s\n", f.name);

			ImageData img = ReadJPEGFile(f.path);
			extract_regions(beadlist, size, frame++, &img, cb);
			img.free();
		}

		if (frame==framelimit)
			break;

		tinydir_next(&d);
	}
}

struct RPTree {
	int D;
	std::vector<float*> points;

	struct Node {
		Node(int D) :D(D) { a=b=0; }
		~Node() { delete a; delete b; }
		Node *a, *b;

		int D;
		std::vector<float> mean;
		std::vector<float> normal;
		float planeDist; // seperation plane is defined as normal * planeDist
		std::vector<float*> points; // also stored in RPTree::points
		
		// non-normalized random vector
		void computeRandomVector(float* r) {
			for (int i=0;i<D;i++) {
				r[i] = rand_uniform<float>();
			}
		}

		void finalize()
		{
			assert( !points.empty() );

			if (points.size() < 10) {
				mean.resize(D);
				std::fill(mean.begin(),mean.end(), 0.0f);

				for (int j=0;j<points.size();j++) {
					float* p = points[j];
					for (int i=0;i<D;i++)
						mean[i] += p[i];
				}
				for (int i=0;i<D;i++)
					mean[i] /= points.size();

			} else
				split();
		}


		// i think this should answer the question: to what extend is the set of points gaussian distributed?
		float computeVQ()
		{
			float err = 0.0f;
			for (int i=0;i<points.size();i++) {

			}
			return err;
		}

		void split()
		{
			normal.resize(D);
			computeRandomVector(&normal[0]);
		
			struct ValueIndexPair {
				int index;
				float value;
				bool operator<(ValueIndexPair o) { return value < o.value; }
			};

			std::vector<ValueIndexPair> dist;
			// compute dot product between random vector and point
			dist.resize(points.size());

			for (int i=0;i<points.size();i++) 
			{
				float sum = 0.0f;
				float* pt = points[i];

				for (int j=0;j<D;j++)
					sum += normal[j] * pt[j];
				dist[i].value = sum;
				dist[i].index = i;
			}

			// find median point
			ValueIndexPair median = qselect(&dist[0], 0, points.size(), points.size()/2);
	
			planeDist = median.value;
			a = new Node(D);
			b = new Node(D);
			a->points.reserve(points.size()/2);
			b->points.reserve(points.size()-points.size()/2);

			int i = 0;
			for (;i<points.size()/2;i++)
				a->points.push_back( points[dist[i].index] );
			for (;i<points.size();i++)
				b->points.push_back( points[dist[i].index] );

			a->finalize();
			b->finalize();
		}
	};

	Node* root;

	RPTree(int D) : D(D) { root = 0; }
	~RPTree() { DeleteAllElems(points); delete root; }

	int memuse() { return points.size()*D*sizeof(float); }

	void add(float* pt) {
		float* cp = new float[D];
		for (int i=0;i<D;i++) cp[i]=pt[i];
		points.push_back(cp);
	}


	void compute()
	{
		root = new Node(D);
		root->points = points;
		root->split();
	}
};

std::vector<RPTree*> trees;

void build_tree(ImageData* img, int bead, int frame)
{
	trees[bead]->add(img->data);
}

void print_memuse()
{
	int memuse=0;
	for (int i=0;i<trees.size();i++)
		memuse+=trees[i]->memuse();
	dbgprintf("Memuse: %d\n", memuse);
}


void test_median()
{
	const int L = 5;
	int x[L] = { 5,3,6,7,2 };
	int y[L];

	for (int i=0;i<L;i++) {
		for (int j=0;j<L;j++) y[j]=x[j];
		int v = qselect(y, 0, L, i);
		dbgprintf("sorted[%d]=%d\n", i,v);
	}
}


int main(int argc, char* argv[])
{
	//test_median();

	const char *path = "../../datasets/1/tmp_001";
	auto beadlist = read_beadlist(std::string(path) + "/beadlist.txt");
	dbgprintf("%d beads.\n", beadlist.size());

	int W=80;
	for (int i=0;i<beadlist.size();i++)
		trees.push_back(new RPTree(W*W));
	process_beads(path, W, beadlist, build_tree, 40);

	print_memuse();

	for (int j=0;j<trees.size();j++) {
		trees[j]->compute();
	}

	DeleteAllElems(trees);
	return 0;
}

