#ifndef __THREADS__H__
#define __THREADS__H__

extern "C" {
  void initialize_threads();  // initializes threads for training
  int get_max_threads();  // returns the max no of threads
}

#endif