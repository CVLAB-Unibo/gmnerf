import numpy as np
import random
import sys
import torch

from typing import Callable, List, Optional, Union

from gmn.graph_construct.hash_grid import MultiResHashGrid
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


class HashGridRadianceField(torch.nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        mlp: str = "FullyFusedMLP",
        activation: str = "ReLU",
        n_hidden_layers: int = 3,
        n_neurons: int = 64,
        encoding: str = "cuda",
        n_levels: int = 4,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 12,
        base_resolution: int = 16,
        max_resolution: int = 128,
        seed: Optional[int] = None
    ) -> None:
        super().__init__()
        if seed is None:
            seed = random.randint(0, sys.maxsize)

        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32,)
        self.register_buffer("aabb", aabb, persistent=False)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale,
        }
        network_config={
            "otype": mlp,  # FullyFusedMLP, CutlassMLP
            "activation": activation,
            "output_activation": "None",
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers
        }
        if encoding == "cuda":
            self.encoding = tcnn.Encoding(
                seed=seed,
                n_input_dims=num_dim, 
                encoding_config=encoding_config
            )
        else:
            self.encoding = MultiResHashGrid(
                num_dim,
                n_levels,
                n_features_per_level,
                log2_hashmap_size,
                base_resolution,
                max_resolution,
            )
        self.mlp_base = tcnn.Network(
            seed=seed,
            n_input_dims=self.encoding.n_output_dims if encoding == "cuda" else self.encoding.output_dim, 
            n_output_dims=4, 
            network_config=network_config
        )

    def query_density_and_rgb(self, x, dir=None):
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)

        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        # Sometimes the ray marching algorithm calls the model with an input with 0 length.
        # The CutlassMLP crashes in these cases, therefore this fix has been applied.
        if len(x) == 0:
            rgb = torch.zeros([0, 3], device=x.device)
            density = torch.zeros([0, 1], device=x.device)
            return rgb, density

        out = self.mlp_base(self.encoding(x.view(-1, self.num_dim))).to(x)
        
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
