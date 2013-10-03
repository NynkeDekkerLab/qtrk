
#include "std_incl.h"
#include "utils.h"
#include "../utils/tinydir.h"
#include <list>
#include "random_distr.h"


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

