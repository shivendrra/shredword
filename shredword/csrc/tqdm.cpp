#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "tqdm.h"

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void HMS(double seconds, char* output) {
  int hrs = (int)seconds / 3600;
  int mins = ((int)seconds % 3600) / 60;
  int secs = (int)seconds % 60;
  snprintf(output, 16, "%02d:%02d:%02d", hrs, mins, secs);
}

void SI(double value, char* output) {
  const char* units = " kMGTPEZY";
  int idx = 0;
  while (value >= 1000.0 && idx < 8) {
    value /= 1000.0;
    idx++;
  }
  snprintf(output, 16, "%.2f%c", value, units[idx]);
}

void init_tqdm(tqdm* bar, const char* desc, bool disable, const char* unit, bool unit_scale, int total, int rate) {
  bar->desc = desc;
  bar->disable = disable;
  bar->unit = unit;
  bar->unit_scale = unit_scale;
  bar->total = total;
  bar->current = 0;
  bar->skip = 1;
  bar->start_time = get_time();
  bar->rate = rate;
}

void update_tqdm(tqdm* bar, int increments, bool close) {
  if (bar->disable) return;

  if (bar->current + increments > bar->total) {
    bar->current = bar->total;
  } else {
    bar->current += increments;
  }
  double elapsed = get_time() - bar->start_time;
  double progress = bar->total > 0 ? (double)bar->current / bar->total : 0.0;

  if (elapsed > 0.0 && bar->current / elapsed > bar->rate && bar->current > 0) {
    bar->skip = (int)(bar->current / elapsed) / bar->rate;
    if (bar->skip < 1) bar->skip = 1;
  }

  if (bar->current % bar->skip != 0 && !close) return;

  print_tqdm(bar, close);
}

void print_tqdm(tqdm* bar, bool close) {
  if (bar->disable) return;

  double elapsed = get_time() - bar->start_time;
  double progress = bar->total > 0 ? (double)bar->current / bar->total : 0.0;
  char elapsed_text[16], remaining_text[16], rate_text[16];
  HMS(elapsed, elapsed_text);

  if (bar->total > 0 && bar->current > 0) {
    double remaining = elapsed / progress - elapsed;
    HMS(remaining, remaining_text);
  } else {
    strcpy(remaining_text, "?");
  }

  if (bar->current > 0) {
    double rate = (double)bar->current / elapsed;
    if (bar->unit_scale) {
      SI(rate, rate_text);
    } else {
      snprintf(rate_text, 16, "%.2f", rate);
    }
  } else {
    strcpy(rate_text, "?");
  }

  int bar_width = 20;
  int filled = (int)(bar_width * progress);
  char progress_bar[bar_width + 1];
  memset(progress_bar, '=', filled);
  memset(progress_bar + filled, '-', bar_width - filled);
  progress_bar[bar_width] = '\0';

  printf("\r%s [%s] %.1f%% %d/%d [%s<%s, %s%s/s]", bar->desc, progress_bar, progress * 100, bar->current, bar->total, elapsed_text, remaining_text, rate_text, bar->unit);
  if (close) {
    printf("\n");
  }
}

void close_tqdm(tqdm* bar) {
  bar->disable = true;
}

void dfs(void* x, PrettyCacheEntry* cache, size_t cache_size, void** (*srcfn)(void*)) {
  for (void** srcs = srcfn(x); srcs && *srcs; ++srcs) {
    PrettyCacheEntry* entry = &cache[(size_t)*srcs % cache_size];
    entry->count++;
    if (entry->count == 1) {
      dfs(*srcs, cache, cache_size, srcfn);
    }
  }
}

void init_trange(tqdm* bar, int n, const char* desc, bool disable, const char* unit, bool unit_scale, int rate) {
  init_tqdm(bar, desc, disable, unit, unit_scale, n, rate);
}

char* pretty_print(void* x, char* (*rep)(void*), void** (*srcfn)(void*), PrettyCacheEntry* cache, size_t cache_size, int depth) {
  size_t index = (size_t)x % cache_size;
  PrettyCacheEntry* entry = &cache[index];
  if (!entry->visited) {
    entry->visited = true;
    entry->id = index;
  }
  if (entry->count++ > 0) {
    char* visited_buffer = (char*)malloc(128);
    snprintf(visited_buffer, 128, "%*s<visited x%d>", depth * 2, "", entry->id);
    return visited_buffer;
  }

  char* rep_str = rep(x);
  char* result_buffer = (char*)malloc(20480);
  snprintf(result_buffer, 1024, "%*s x%d: %s", depth * 2, "", entry->id, rep_str);
  for (void** srcs = srcfn(x); srcs && *srcs; ++srcs) {
    char* child_str = pretty_print(*srcs, rep, srcfn, cache, cache_size, depth + 1);
    strncat(result_buffer, "\n", 1024 - strlen(result_buffer) - 1);
    strncat(result_buffer, child_str, 1024 - strlen(result_buffer) - 1);
    free(child_str);
  }
  return result_buffer;
}
