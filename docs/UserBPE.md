# ShredWord-BPE Trainer Documentation

## Overview

The BPE Trainer is a Python wrapper for training Byte Pair Encoding (BPE) tokenizers. BPE is a subword tokenization algorithm commonly used in natural language processing for creating vocabulary from text corpora.

## Installation

Ensure you have the required dependencies installed and the underlying C library (`cbase`) is properly configured.

From [PyPI.org](https://pypi.org/project/shredword-trainer/):

```bash
  pip install shredword-trainer
```

Importing:

```python
  from shredword.trainer import BPETrainer
```

## Quick Start

```python
# Initialize trainer with default settings
trainer = BPETrainer()

# Load your text corpus
trainer.load_corpus("path/to/your/corpus.txt")

# Train the BPE model
trainer.train()

# Save the trained model and vocabulary
trainer.save("base.model", "base.vocab")

# Clean up resources
trainer.destroy()
```

## Class Reference

### BPETrainer

The main class for training BPE tokenizers.

#### Constructor

```python
BPETrainer(target_vocab_size=8192, unk_id=0, character_coverage=0.995, min_pair_freq=2000)
```

**Parameters:**

- `target_vocab_size` (int, default=8192): Target vocabulary size for the trained model
- `unk_id` (int, default=0): ID assigned to unknown tokens
- `character_coverage` (float, default=0.995): Percentage of characters to be covered by the model (0.0-1.0)
- `min_pair_freq` (int, default=2000): Minimum frequency required for a character pair to be considered for merging

**Raises:**
- `RuntimeError`: If the trainer fails to initialize

#### Methods

##### `load_corpus(path: str)`

Loads a text corpus from the specified file path.

**Parameters:**
- `path` (str): Path to the text file containing the training corpus

**Raises:**
- `IOError`: If the corpus file cannot be loaded

**Example:**
```python
trainer.load_corpus("/path/to/training_data.txt")
```

##### `train()`

Trains the BPE model using the loaded corpus.

**Returns:**
- Prints the number of merges performed during training

**Raises:**
- `RuntimeError`: If training fails

**Example:**
```python
trainer.train()
# Output: Training completed: 7500 merges performed.
```

##### `save(model_path: str, vocab_path: str)`

Saves the trained BPE model and vocabulary to specified files.

**Parameters:**
- `model_path` (str): Path where the BPE model will be saved
- `vocab_path` (str): Path where the vocabulary will be saved

**Example:**
```python
trainer.save("my_model.model", "my_vocab.vocab")
# Output: Model saved to: my_model.model
# Output: Vocabulary saved to: my_vocab.vocab
```

##### `destroy()`

Manually releases resources used by the trainer. This is automatically called when the object is deleted.

**Example:**
```python
trainer.destroy()
```

## Configuration Parameters

### Target Vocabulary Size
- **Default:** 8192
- **Description:** The desired size of the final vocabulary. Larger vocabularies can capture more nuanced subword patterns but require more memory.
- **Typical Range:** 1000-50000

### Unknown Token ID
- **Default:** 0
- **Description:** The ID assigned to out-of-vocabulary tokens during tokenization.

### Character Coverage
- **Default:** 0.995 (99.5%)
- **Description:** The percentage of characters from the training corpus that should be covered by the model. Higher values ensure better coverage but may include very rare characters.
- **Range:** 0.0-1.0

### Minimum Pair Frequency
- **Default:** 2000
- **Description:** The minimum number of times a character pair must appear in the corpus to be considered for merging. Higher values result in more conservative merging.

## Usage Examples

### Basic Training

```python
from shredword.trainer import BPETrainer

# Create trainer with custom parameters
trainer = BPETrainer(
    target_vocab_size=16000,
    character_coverage=0.9999,
    min_pair_freq=1000
)

# Load corpus and train
trainer.load_corpus("large_corpus.txt")
trainer.train()
trainer.save("production_model.model", "production_vocab.vocab")
trainer.destroy()
```

### Context Manager Pattern

```python
class BPETrainerContext:
    def __init__(self, **kwargs):
        self.trainer = BPETrainer(**kwargs)
    
    def __enter__(self):
        return self.trainer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.trainer.destroy()

# Usage
with BPETrainerContext(target_vocab_size=32000) as trainer:
    trainer.load_corpus("corpus.txt")
    trainer.train()
    trainer.save("model.model", "vocab.vocab")
```

### Multiple Corpus Training

```python
trainer = BPETrainer(target_vocab_size=25000)

# Load multiple corpus files (call load_corpus multiple times)
corpus_files = ["corpus1.txt", "corpus2.txt", "corpus3.txt"]
for corpus_file in corpus_files:
    trainer.load_corpus(corpus_file)

trainer.train()
trainer.save("multi_corpus_model.model", "multi_corpus_vocab.vocab")
trainer.destroy()
```

## Error Handling

The trainer can raise several exceptions:

```python
try:
    trainer = BPETrainer(target_vocab_size=10000)
    trainer.load_corpus("corpus.txt")
    trainer.train()
    trainer.save("model_10k.model", "vocab_10k.vocab")
except RuntimeError as e:
    print(f"Training error: {e}")
except IOError as e:
    print(f"File error: {e}")
finally:
    trainer.destroy()
```

## Best Practices

1. **Resource Management:** Always call `destroy()` or use a context manager to properly clean up resources
2. **Corpus Size:** Ensure your corpus is large enough (typically millions of tokens) for meaningful BPE training
3. **Parameter Tuning:** Experiment with different `min_pair_freq` values based on your corpus size
4. **File Paths:** Use absolute paths to avoid issues with relative path resolution
5. **Memory Usage:** Monitor memory usage with large corpora and adjust parameters accordingly

## Troubleshooting

### Common Issues

**"Failed to create BPE trainer"**
- Check that the underlying C library is properly installed
- Verify that configuration parameters are within valid ranges

**"Failed to load corpus"**
- Ensure the corpus file exists and is readable
- Check file encoding (UTF-8 is typically expected)
- Verify sufficient disk space and memory

**"Training failed"**
- Corpus may be too small or empty
- Try reducing `min_pair_freq` for small corpora
- Check available memory for large vocabularies

### Performance Tips

- Use SSD storage for faster corpus loading
- Consider the trade-off between vocabulary size and training time
- Monitor memory usage during training with large corpora
- For very large corpora, consider preprocessing to remove extremely rare characters

## File Format Requirements

### Corpus Format
- Plain text files
- UTF-8 encoding recommended
- One sentence per line (typical)
- No special preprocessing required

### Output Files
- **Model file (.model):** Contains the trained BPE merge operations
- **Vocabulary file (.txt):** Contains the vocabulary mapping

## Version Compatibility

This documentation assumes the trainer interface is stable. Check your specific version for any API changes or additional features.
