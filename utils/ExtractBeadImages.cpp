#include "std_incl.h"
#include "utils.h"
#include "ExtractBeadImages.h"
#include "tinydir.h"

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

