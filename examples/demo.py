import torch
from transformer.utils import prep, build_vocab, encode_string, decode_indices, get_dataloaders
from transformer.model import TransformerDecoder


def main():
    # Load and preprocess data
    with open("train.txt", "r") as f:
        lines = f.read().splitlines()
    cleaned = prep(lines)
    chars, stoi, itos = build_vocab(cleaned)

    # Create DataLoaders
    context_size = 128
    batch_size = 32
    train_loader, val_loader = get_dataloaders(
        cleaned, stoi, context_size, batch_size
    )

    # Instantiate model
    model = TransformerDecoder(
        vocab_size=len(chars),
        d_model=128,
        num_heads=8,
        head_size=16,
        num_layers=8,
        dropout=0.2,
        context_size=context_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training loop
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

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    # Generation demo
    prompt = "berk: naber nası gidiyo"
    prompt_idx = torch.tensor([encode_string(prompt + '#', stoi)], device=device)
    generated = model.generate(prompt_idx, max_new_tokens=68)
    output = decode_indices(generated[0].tolist(), itos)
    print(output)


if __name__ == "__main__":
    main()