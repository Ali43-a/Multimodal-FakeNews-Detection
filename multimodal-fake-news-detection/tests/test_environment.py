import torch


def test_pytorch_available():
    assert torch.__version__ is not None


def test_tensor_creation():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert tensor.shape == (3,)
