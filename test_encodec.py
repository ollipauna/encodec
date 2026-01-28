import torch
from encodec import EncodecModel


def test_encodec():
    # Instantiate a model
    model = EncodecModel.encodec_model_24khz()

    # Create a random audio signal
    x = torch.randn(1, 1, 16000, requires_grad=True)

    model.eval()
    x_quantized = model(x)
    x_org_quantized = model._original_forward(x)

    # Check that the audio reconstructions match
    assert torch.allclose(x_quantized, x_org_quantized, atol=1e-3)
    assert x_quantized.size() == x.size()

    loss = torch.pow(x_quantized, 2).sum()
    loss.backward()
    model.zero_grad()

    # Check that gradients are being tracked
    assert x.grad is not None

    # Check that gradient values look sensible
    print(x.grad)

    # Check that adding noise changes the gradients
    x_other = x.detach() + (0.1**0.5) * torch.randn(1, 1, 16000)
    x_other.requires_grad_()
    x_quantized_other = model(x_other)
    loss_other = torch.pow(x_quantized_other, 2).sum()
    loss_other.backward()
    print(torch.mean(torch.abs(x.grad - x_other.grad)))


if __name__ == "__main__":
    test_encodec()
