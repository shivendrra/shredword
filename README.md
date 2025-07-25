# ShredWord

ShredWord is a byte-pair encoding (BPE) based tokenizer-trainer designed for fast, efficient, and flexible text processing & vocab training. It offers training, and text normalization functionalities and is backed by a C/C++ core with a Python interface for easy integration into machine learning workflows.

## Features

1. **Efficient Tokenization**: Utilizes BPE for compressing text data and reducing the vocabulary size, making it well-suited for NLP tasks.
2. **Customizable Vocabulary**: Allows users to define the target vocabulary size during training.
3. **Save and Load Models**: Supports saving and loading trained tokenizers for reuse.
4. **Python Integration**: Provides a Python interface for seamless integration and usability.

## How It Works

### Byte-Pair Encoding (BPE)

BPE is a subword tokenization algorithm that compresses a dataset by merging the most frequent pairs of characters or subwords into new tokens. This process continues until a predefined vocabulary size is reached.

Key steps:

1. Initialize the vocabulary with all unique characters in the dataset.
2. Count the frequency of character pairs.
3. Merge the most frequent pair into a new token.
4. Repeat until the target vocabulary size is achieved.

ShredWord implements this process efficiently in C/C++, exposing training, encoding, and decoding methods through Python.

## Installation

### Prerequisites

- Python 3.11+
- GCC or a compatible compiler (for compiling the C/C++ code)

### Steps

1. Install the Python package from [PyPI.org](https://pypi.org/project/shredword-trainer/):

   ```bash
   pip install shredword-trainer
   ```

## Usage

Below is a simple example demonstrating how to use ShredWord for training, encoding, and decoding text.

### Example

```python
from shredword.trainer import BPETrainer

trainer = BPETrainer(target_vocab_size=500, min_pair_freq=1000)
trainer.load_corpus("test data/final.txt")
trainer.train()
trainer.save("model/merges_1k.model", "model/vocab_1k.vocab")
```

## API Overview

### Core Methods

- `train(text, vocab_size)`: Train a tokenizer on the input text to a specified vocabulary size.
- `save(file_path)`: Save the trained tokenizer to a file.

### Properties

- `merges`: View or set the merge rules for tokenization.
- `vocab`: Access the vocabulary as a dictionary of token IDs to strings.
- `pattern`: View or set the regular expression pattern used for token splitting.
- `special_tokens`: View or set special tokens used by the tokenizer.

## Advanced Features

### Saving and Loading

Trained tokenizers can be saved to a file and reloaded for use in future tasks. The saved model includes merge rules and any special tokens or patterns defined during training.

```python
# Save the trained model
tokenizer.save("vocab/trained_vocab.model")

# Load the model
tokenizer.load("vocab/trained_vocab.model")
```

### Customization

Users can define special tokens or modify the merge rules and pattern directly using the provided properties.

```python
# Set special tokens
special_tokens = [("<PAD>", 0), ("<UNK>", 1)]
tokenizer.special_tokens = special_tokens

# Update merge rules
merges = [(101, 32, 256), (32, 116, 257)]
tokenizer.merges = merges
```

## Contributing

We welcome contributions! Feel free to open an issue or submit a pull request if you have ideas for improvement.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

ShredWord was inspired by the need for efficient and flexible tokenization in modern NLP pipelines. Special thanks to contributors and the open-source community for their support.
