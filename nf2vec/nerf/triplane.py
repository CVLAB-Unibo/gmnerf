import random
import sys
import torch

from typing import Callable, List, Optional, Union

from nf2vec.nerf.mlp import contract_to_unisphere, trunc_exp

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class TriplaneRadianceField(torch.nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = False,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 3,
        encoding: str = "Frequency",
        mlp: str = "CutlassMLP",
        activation: str = "Sine",
        n_hidden_layers: int = 3,
        n_neurons: int = 64,
        encoding_size: int = 48,
        triplane_res: int = 32,
        triplane_feat_dim: int = 16,
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

        fdim = triplane_feat_dim
        res = triplane_res
        self.triplane = torch.nn.Parameter(
            torch.normal(
                torch.zeros(3, res, res, fdim, device="cuda"), 
                torch.ones(3, res, res, fdim, device="cuda") * 0.001
            )
        )
        self.encoding = tcnn.Encoding(
            seed=seed, 
            n_input_dims=3, 
            encoding_config=single_mlp_encoding_config
        )
        self.mlp_base = tcnn.Network(
            seed=seed,
            n_input_dims=self.encoding.n_output_dims + fdim, 
            n_output_dims=4, 
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
            scaled_coords = x.clone()
            scaled_coords = 2*(scaled_coords - aabb_min) / (aabb_max - aabb_min) - 1
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
        if len(x) <= 10:
            rgb = torch.zeros([len(x), 3], device=x.device)
            density = torch.zeros([len(x), 1], device=x.device)
            return rgb, density
        else:
            coords_xy = torch.cat((scaled_coords[:, 0].unsqueeze(1), scaled_coords[:, 1].unsqueeze(1)), dim=1)
            coords_xz = torch.cat((scaled_coords[:, 0].unsqueeze(1), scaled_coords[:, 2].unsqueeze(1)), dim=1)
            coords_zy = torch.cat((scaled_coords[:, 1].unsqueeze(1), scaled_coords[:, 2].unsqueeze(1)), dim=1)

            num_points = x.shape[0]
            grid = torch.stack([coords_xy, coords_xz, coords_zy], dim=0).unsqueeze(0)
            grid = grid.reshape(-1, num_points, 2).unsqueeze(1)

            features_sample = torch.nn.functional.grid_sample(
                self.triplane.permute(0, 3, 1, 2), 
                grid, 
                align_corners=True
            ).squeeze(2).squeeze(2).permute(0, 2, 1)
            features_sample = features_sample.sum(0)
            x = torch.cat([self.encoding(x),features_sample], dim=1)

            out = self.mlp_base(x).to(x)

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
