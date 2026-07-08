import torch

from DL_Template.models.model import ExampleModel


def test_model_forward_shape():
    model = ExampleModel(input_dim=10, hidden_dim=16, output_dim=3)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 3)
