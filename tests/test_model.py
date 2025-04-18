import torch
from transformer.model import TransformerDecoder


def test_model_forward_pass():
    vocab_size = 50
    context_size = 10
    model = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        head_size=8,
        num_layers=2,
        dropout=0.1,
        context_size=context_size,
    )
    x = torch.randint(0, vocab_size, (8, context_size))
    logits = model(x)
    assert logits.shape == (8, context_size, vocab_size)
    print("Success")
    # test loss computation
    y = torch.randint(0, vocab_size, (8, context_size))
    logits, loss = model(x, y)
    assert isinstance(loss.item(), float)
    print("Success")


if __name__ == "__main__":
    test_model_forward_pass()
    