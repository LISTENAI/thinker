#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <assert.h>

#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <sys/resource.h>
#endif
#include "thinker/thinker.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "ghc/filesystem.hpp"

void load_bin_file(const char *file, int8_t **ptr, uint64_t *size) {
  FILE *fp = fopen(file, "rb");

  fseek(fp, 0 ,SEEK_END);
  *size = ftell(fp);
  fseek(fp, 0 ,SEEK_SET);
  *ptr = (int8_t*)malloc(*size);
  fread(*ptr, *size, 1, fp);
  fclose(fp);
}

void save_bin_file(const char *file, int8_t *ptr, int32_t size)
{
	FILE *fp = fopen(file, "ab+");

	fwrite(ptr, size, 1, fp);

	fclose(fp);
}