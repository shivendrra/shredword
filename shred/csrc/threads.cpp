#include <stdio.h>
#include <pthread.h>
#include "threads.h"

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#define DEFAULT_MAX_THREADS 8

int get_max_threads() {
  int num_threads = 1;
#ifdef _WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  num_threads = sysinfo.dwNumberOfProcessors;
#else
  num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  int max_threads = (num_threads > 2) ? (num_threads - 2) : 1;
  return max_threads;
}
int MAX_THREADS = DEFAULT_MAX_THREADS;
void initialize_threads() {
  MAX_THREADS = get_max_threads();
  printf("Detected CPU threads: %d, using max threads: %d\n", MAX_THREADS + 2, MAX_THREADS);
}