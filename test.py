import unicodedata

def normalize_and_segment(text):
  text = unicodedata.normalize("NFKC", text)
  text = text.replace(" ", "▁")
  return text

with open("captions.txt", "r", encoding="utf-8") as fin, open("train.txt", "w", encoding="utf-8") as fout:
  for line in fin:
    norm = normalize_and_segment(line.strip())
    fout.write(norm + "\n")