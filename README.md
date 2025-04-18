# Transformer-Decoder

A lightweight, character-level Transformer decoder implementation designed for next-token prediction tasks. This package provides a complete, structured, and modular approach to using transformer decoders, inspired by the seminal work by Vaswani et al., "Attention is All You Need" (2017).

## 🎯 What Is This?
This project implements a Transformer decoder architecture tailored specifically for character-level language modeling. The Transformer decoder leverages multi-head self-attention, feed-forward neural networks, and positional encoding to effectively model sequential data. This implementation is designed to be easy-to-understand, extendable, and production-ready.

## 🚀 Key Features
- **Multi-head self-attention:** Captures dependencies regardless of their distance in the input sequence.
- **Layer normalization:** Stabilizes training by normalizing activations.
- **Sinusoidal positional encoding:** Injects information about token positions.
- **Modular architecture:** Clearly separated components like attention, feed-forward, and embedding layers.
- **Configurable parameters:** Easily adjust depth, width, context size, and dropout rates.
- **Testing suite:** Robust unit tests provided for reliability.

## 🛠 Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/transformer-decoder.git
cd transformer-decoder
pip install -r requirements.txt
pip install -e .
```

## 📚 Usage

Here's how you can quickly get started with training and generating text:

### Training and Evaluating the Model

Run the provided demo script to train the model using your own data:

```bash
python examples/run_demo.py
```

Make sure you have a text file named `train.txt` at the root of your directory containing your training data, or update the script accordingly.

### Using the Transformer Decoder in Your Own Project

You can directly import and utilize the Transformer decoder in any Python script after installing:

```python
from transformer.utils import prep, build_vocab, get_dataloaders
from transformer.model import TransformerDecoder
import torch

# Prepare data
with open("train.txt", "r") as f:
    lines = f.read().splitlines()
cleaned = prep(lines)
chars, stoi, itos = build_vocab(cleaned)

# Setup DataLoaders
context_size = 128
batch_size = 32
train_loader, val_loader = get_dataloaders(cleaned, stoi, context_size, batch_size)

# Initialize model
model = TransformerDecoder(
    vocab_size=len(chars),
    d_model=128,
    num_heads=8,
    head_size=16,
    num_layers=8,
    dropout=0.2,
    context_size=context_size
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Text generation
prompt = "Hello, how are you?"
prompt_idx = torch.tensor([encode_string(prompt + '#', stoi)], device=device)
generated = model.generate(prompt_idx, max_new_tokens=100)
output = decode_indices(generated[0].tolist(), itos)
print(output)
```

## 🧬 Project Structure

Here's a quick overview of the project structure:

```
transformer-decoder/
├── transformer/       # Core modules (model, layers, utils)
├── examples/          # Demonstration scripts
├── tests/             # Unit tests
├── README.md          # Project overview and usage instructions
├── requirements.txt   # List of dependencies
├── setup.py           # Package metadata for installation
├── LICENSE            # MIT License
└── CONTRIBUTING.md    # Guidelines for contributors
```

## 📝 Tests

To ensure correctness and reliability, unit tests are provided. Run tests with:

```bash
pytest
```

## 📖 References
- Vaswani, Ashish, et al. "Attention is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

## 🤝 Contributing

We welcome contributions! Check out the [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

© 2025 Your Name
