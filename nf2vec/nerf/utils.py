import collections
import numpy as np
import torch
import torch.nn.functional as F


Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def get_rays(
    device,
    camera_angle_x=0.8575560450553894,  # taken from traned NeRFs
    width=224,
    height=224
):
    # Get camera pose
    theta = torch.tensor(90.0, device=device)  # horizontal camera position
    phi = torch.tensor(-30.0, device=device)   # vertical camera position
    t = torch.tensor(1.5, device=device)       # camera distance from object
    c2w = pose_spherical(theta, phi, t)
    c2w = c2w.to(device)

    # Compute focal length 
    focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)

    rays = generate_rays(device, width, height, focal_length, c2w)
    return rays


def generate_rays(device, width, height, focal, c2w, OPENGL_CAMERA=True):
    x, y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    K = torch.tensor(
        [
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )  # (3, 3)

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5)
                / K[1, 1]
                * (-1.0 if OPENGL_CAMERA else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if OPENGL_CAMERA else 1.0),
    )  # [num_rays, 3]
    camera_dirs.to(device)

    directions = (camera_dirs[:, None, :] * c2w[:3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

    origins = torch.reshape(origins, (height, width, 3))
    viewdirs = torch.reshape(viewdirs, (height, width, 3))
    
    rays = Rays(origins=origins, viewdirs=viewdirs)
    return rays


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.from_numpy(np.array(
        [
            [-1, 0, 0, 0], 
            [0, 0, 1, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 1]
        ], dtype=np.float32)
    ) @ c2w  
    return c2w


def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [torch.cos(theta), 0, -torch.sin(theta), 0],
        [0, 1, 0, 0],
        [torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(matrix, dtype=torch.float32)
