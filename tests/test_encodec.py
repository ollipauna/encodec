import torch
from encodec import EncodecModel


def test_encodec():
    # Instantiate a model
    model = EncodecModel.encodec_model_24khz()

    # Create a random audio signal
    x = torch.randn(2, 1, 16000, requires_grad=True)

    model.eval()
    x_org_quantized = model._original_forward(x)

    model.train()
    for _ in range(10):
    
        x_quantized = model(x)

        # Check that the audio reconstructions match
        assert torch.allclose(x_quantized, x_org_quantized, atol=1e-3)
        assert x_quantized.size() == x.size()

    loss = torch.pow(x_quantized, 2).sum()
    loss.backward()
    model.zero_grad()

    # Check that gradients are being tracked
    assert x.grad is not None
