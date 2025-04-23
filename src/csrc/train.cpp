#include <unicode/unorm2.h>
#include <unicode/utypes.h>
#include <unicode/ustring.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "train.h"

/**
  @brief Normalize input UTF-8 text to NFKC form and replace spaces with "▁" (U+2581).

  * This function performs normalization using ICU's NFKC normalizer. Due to limitations
  * in ICU versions prior to 60, we cannot directly normalize UTF-8 strings using
  * `unorm2_normalizeUTF8`, which is only available in ICU v60+.

  * Instead, the function performs the following steps:
  *   1. Converts the input UTF-8 string to UTF-16 using `u_strFromUTF8`.
  *   2. Normalizes the UTF-16 string using `unorm2_normalize`, which supports NFKC.
  *   3. Converts the normalized UTF-16 result back to UTF-8 using `u_strToUTF8`.
  *   4. Replaces all ASCII space characters (' ') with the Unicode character "▁" (U+2581),
  *      which is commonly used in subword tokenization schemes such as SentencePiece.

  * While this roundtrip conversion incurs minor overhead, ICU's internal optimizations
  * make it efficient enough for most practical applications. This approach ensures
  * compatibility with older ICU installations while preserving normalization accuracy.

  @param input_text A null-terminated UTF-8 encoded string.
  @return A newly allocated UTF-8 normalized string with spaces replaced.
          - The caller is responsible for freeing the returned string.
*/

char* normalize_text(const char* input_text) {
  UErrorCode status = U_ZERO_ERROR;

  const UNormalizer2* normalizer = unorm2_getNFKCInstance(&status);
  if (U_FAILURE(status)) {
    fprintf(stderr, "Failed to get NKFC-instance: %s\n", u_errorName(status));
    return NULL;
  }
 
  // Converting UTF-8 to UTF-16
  UChar utf16[4096];
  int32_t utf16_len = 0;
  u_strFromUTF8(utf16, 4096, &utf16_len, input_text, -1, &status);
  if (U_FAILURE(status)) {
    fprintf(stderr, "Failed to convert UTF-8 to UTF-16: %s\n", u_errorName(status));
    return NULL;
  }

  // Normalizing in UTF-16
  UChar norm16[4096];
  int32_t norm16_len = unorm2_normalize(normalizer, utf16, utf16_len, norm16, 4096, &status);
  if (U_FAILURE(status)) {
    fprintf(stderr, "Normalization failed: %s\n", u_errorName(status));
    return NULL;
  }

  // Converting UTF-16 back to UTF-8
  char* norm8 = (char*)malloc(8192);
  int32_t norm8_len = 0;
  u_strToUTF8(norm8, 8192, &norm8_len, norm16, norm16_len, &status);
  if (U_FAILURE(status)) {
    fprintf(stderr, "Failed to convert UTF-16 back to UTF-8: %s\n", u_errorName(status));
    free(norm8);
    return NULL;
  }

  // replacing all the ASCII spaces with U+2581 (UTF-8: E2 96 81)
  const char* rep = "\xE2\x96\x81";
  size_t rep_len = 3;

  size_t final_len = 0;
  for (int i = 0; i < norm8_len; i++)
    if (norm8[i] == ' ') final_len += rep_len;
    else final_len++;

  char* final = (char*)malloc(final_len + 1);
  size_t j = 0;
  for (int i = 0; i < norm8_len; i++) {
    if (norm8[i] == ' ') {
      memcpy(&final[j], rep, rep_len);
      j += rep_len;
    } else {
      final[j++] = norm8[i];
    }
  }
  final[j] = '\0';
  free(norm8);

  return final;
}