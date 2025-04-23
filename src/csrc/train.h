#ifndef __TRAIN__H__
#define __TRAIN__H__

#include "base.h"
#include "main.h"

extern "C" {
  char* normalize_text(const char* input);  // normalize input text to NFKC form and replace spaces with "▁"
}

#endif  //!__TRAIN__H__