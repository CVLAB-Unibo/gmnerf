import random
import sys
import torch

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from typing import Callable, List, Optional, Union

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x


class MlpRadianceField(torch.nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = False,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 3,
        encoding: str = "Frequency",
        mlp: str = "FullyFusedMLP",
        activation: str = "ReLU",
        n_hidden_layers: int = 3,
        n_neurons: int = 64,
        encoding_size: int = 24,
        seed: Optional[int] = None
    ) -> None:
        super().__init__()
        if seed is None:
            seed = random.randint(0, sys.maxsize)

        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32,)
        self.register_buffer("aabb", aabb, persistent=False)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded

        self.geo_feat_dim = geo_feat_dim if use_viewdirs else 0

        if self.use_viewdirs:
            single_mlp_encoding_config = {
                "otype": "Composite",
                "nested": [
                    # POSITION ENCODING
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Frequency",
                        "n_frequencies": 6,

                    },
                    # DIRECTION ENCODING
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 1,  # determines the output dimension, which is degree^2
                    }
                ]
            }
        else:
            if encoding == "Frequency":
                single_mlp_encoding_config = {
                    "otype": "Frequency",
                    "n_frequencies": encoding_size
                }
            else:
                single_mlp_encoding_config = {
                    "otype": "Identity"
                }

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            seed=seed,
            n_input_dims=self.num_dim+self.geo_feat_dim,
            n_output_dims=4,
            encoding_config=single_mlp_encoding_config,
            network_config={
                "otype": mlp,  # FullyFusedMLP, CutlassMLP
                "activation": activation,
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers
            }
        )

    def query_density_and_rgb(self, x, dir=None):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)

        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        if self.use_viewdirs:
            if dir is not None:
                dir = (dir + 1.0) / 2.0
                x = torch.cat([x, dir], dim=-1)
            else:
                random = torch.rand(x.shape[0], self.geo_feat_dim, device=x.device)
                x = torch.cat([x, random], dim=-1)

        # Sometimes the ray marching algorithm calls the model with an input with 0 length.
        # The CutlassMLP crashes in these cases, therefore this fix has been applied.
        if len(x) == 0:
            rgb = torch.zeros([0, 3], device=x.device)
            density = torch.zeros([0, 1], device=x.device)
            return rgb, density

        out = self.mlp_base(x.view(-1, self.num_dim+self.geo_feat_dim)).to(x)
        
        rgb, density_before_activation = out[..., :3], out[..., 3]
        density_before_activation = density_before_activation[:, None]
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        rgb = torch.nn.Sigmoid()(rgb)

        return rgb, density

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        rgb, density = self.query_density_and_rgb(positions, directions)

        return rgb, density
