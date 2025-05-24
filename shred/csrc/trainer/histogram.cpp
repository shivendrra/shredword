#include <stdio.h>
#include <stdlib.h>
#include "trainer/histogram.h"
#include "trainer/bpe.h"
#include "inc/hash.h"

void build_symbol_cb(const char* w, uint64_t count, void* u) {
  BuildCtx* ctx = (BuildCtx*)u;
  Trainer* trainer = ctx->trainer;
  size_t pos = *(ctx->idx);

  Symbol *head = NULL, *prev = NULL;
  for (const unsigned char* p = (const unsigned char*)w; *p; ++p) {
    Symbol* s = (Symbol*)malloc(sizeof(Symbol));
    int32_t id = ctx->keep_char[*p] ? (int32_t)*p : trainer->config.unk_id;

    s->id = id;
    s->prev = prev;
    s->next = NULL;
    if (prev) prev->next = s;
    else head = s;
    prev = s;
  }
  trainer->corpus.words[pos] = head;
  trainer->corpus.word_counts[pos] = count;
  (*(ctx->idx))++;
}

// callback to build char histogram from each word
void char_hist(const char* word, uint64_t wcount, void* u) {
  StrMap *cmap = (StrMap*)u;
  for (const unsigned char *p = (const unsigned char*)word; *p; ++p) {
    char tmp[2] = { (char)*p, 0 };
    strmap_increment(cmap, tmp);
  }
}

// callback to collect CharCount entries into the context
void collect_char(const char* kc, uint64_t vc, void* u) {
  CharCountCtx *ctx = (CharCountCtx*)u;
  ctx->arr[ctx->idx].c = kc[0];
  ctx->arr[ctx->idx].count = vc;
  ctx->idx++;
}

// qsort comparator for CharCount descending
int charcount_cmp(const void *a, const void *b) {
  const CharCount *ca = (const CharCount*)a;
  const CharCount *cb = (const CharCount*)b;
  if (cb->count > ca->count) return 1;
  if (cb->count < ca->count) return -1;
  return 0;
}

// --- helper called for each (key, count) --- 
void load_entry(const char* key, uint64_t val, void* user) {
  struct load_ctx* ctx = (struct load_ctx*)user;
  Symbol *head = NULL, *prev = NULL;
  
  // build linked list of raw bytes
  for (const unsigned char* p = (const unsigned char*)key; *p; ++p) {
    Symbol* s = (Symbol*)malloc(sizeof(Symbol));
    s->id = (int32_t)*p;
    s->prev = prev, s->next = NULL;
    if (prev) prev->next = s;
    else head = s;
    prev = s;
  }
  // store into corpus
  ctx->trainer->corpus.words[ctx->idx] = head;
  ctx->trainer->corpus.word_counts[ctx->idx] = val;
  ctx->idx++;
}