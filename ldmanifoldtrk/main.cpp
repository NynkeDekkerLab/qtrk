
#include "std_incl.h"
#include "utils.h"
#include "tinydir.h"
#include <list>

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
	std::list<float*> points;

	int memuse() { return points.size()*D*sizeof(float); }
	RPTree(int D) : D(D) {}
	~RPTree() {	DeleteAllElems(points); }

	void add(float* pt) {
		float* cp = new float[D];
		for (int i=0;i<D;i++) cp[i]=pt[i];
		points.push_back(cp);
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

int main(int argc, char* argv[])
{
	const char *path = "../../datasets/1/tmp_001";
	auto beadlist = read_beadlist(std::string(path) + "/beadlist.txt");
	dbgprintf("%d beads.\n", beadlist.size());

	int W=80;
	for (int i=0;i<beadlist.size();i++)
		trees.push_back(new RPTree(W*W));
	process_beads(path, W, beadlist, build_tree, 40);

	print_memuse();

	DeleteAllElems(trees);
	return 0;
}

