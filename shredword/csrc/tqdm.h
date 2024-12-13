/*
  tqdm.h
  - header file for tqdm library's implementation in c
  - functions & logic implementation are similar to that of ``tqdm`` by TinyGrad
  url: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py
  - for compilation:
    -- ".so": g++ -shared -fPIC -o libtqdm.so tqdm.cpp / for linux
    -- ".dll": g++ -shared -o libtqdm.dll tqdm.cpp / for windows
*/

#ifndef __TQDM__H__
#define __TQDM__H__

#include <stdbool.h>
#include <stddef.h>

typedef struct {
  const char* desc;       // description (e.g. "Loading...")
  bool disable;           // progress bar is disabled or not?
  const char* unit;       // unit of measurement (iters/sec or sec/iters)
  bool unit_scale;        // scale the units or not (K/M/G)?
  int total;              // total iters
  int current;            // current step/iters
  int skip;               // skipping steps
  double start_time;      // start time in secs
  int rate;               // rate of updation in hertz
} tqdm;   // struct that represents the progress bar

typedef struct {
  int id;            // Unique ID for the cached object
  int count;         // Reference count
  bool visited;      // Flag to indicate whether the object is being visited
} PrettyCacheEntry;

extern "C" {
  void init_tqdm(tqdm* bar, const char* desc, bool disable, const char* unit, bool unit_scale, int total, int rate);
  void update_tqdm(tqdm* bar, int increments, bool close);
  void print_tqdm(tqdm* bar, bool close);
  void HMS(double seconds, char* output);
  void SI(double value, char* output);
  void close_tqdm(tqdm* bar);

  void init_trange(tqdm *bar, int n, const char *desc, bool disable, const char *unit, bool unit_scale, int rate);
  char *pretty_print(          // pretty_print: recursively format and print hierarchical data
    void *x,                   // current node
    char *(*rep)(void *),      // function to generate string representation of x
    void **(*srcfn)(void *),   // function to retrieve the sources of x
    PrettyCacheEntry *cache,   // cache for already visited nodes
    size_t cache_size,         // size of the cache
    int depth                  // current depth in the hierarchy
  );
  // depth-first search for pretty_print
  void dfs(void *x, PrettyCacheEntry *cache, size_t cache_size, void **(*srcfn)(void *));
}

#endif