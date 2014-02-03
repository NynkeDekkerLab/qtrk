#pragma once

#include <functional>

typedef std::function< void(ImageData *img, int bead, int frame) > process_image_cb;

struct BeadPos{ int x,y; };

std::vector<BeadPos> read_beadlist(std::string fn);
void extract_regions(std::vector<BeadPos> beads, int size, int frame, ImageData* img, process_image_cb cb);
void process_beads(const char *path, int size, std::vector<BeadPos> beadlist, process_image_cb cb, int framelimit);

void process_bead_dir(const char *path, int roisize, process_image_cb cb, int framelimit);

