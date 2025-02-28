import torch


class OneDAffineTransformation(torch.nn.Module):
    """1-d affine steering."""

    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dim = kwargs.get("embed_dim", 768)
        v = torch.empty(self.embed_dim)
        v = torch.nn.init.normal_(v)
        self.v = torch.nn.Parameter(v, requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, base, subspaces=None):
        self.proj = torch.outer(self.v, self.v) / torch.dot(self.v, self.v)
        output = (
            base
            - torch.matmul(base.to(self.proj.dtype), self.proj)
            + self.beta * self.v
        )
        return output.to(base.dtype)

    def __str__(self):
        return f"OneDAffineTransformation"


class AddVector(torch.nn.Module):
    """Add a vector to activations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dim = kwargs.get("embed_dim", 768)
        v = torch.empty(self.embed_dim)
        v = torch.nn.init.normal_(v)
        self.v = torch.nn.Parameter(v, requires_grad=True)

    def forward(self, base, subspaces=None):
        output = torch.add(base, self.v)
        return output.to(base.dtype)

    def __str__(self):
        return f"AddVector"


if __name__ == "__main__":
    ad = OneDAffineTransformation(embed_dim=768)
    ad(torch.randn(10, 768))
