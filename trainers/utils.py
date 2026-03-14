from tqdm import tqdm
from typing import Iterable, Optional


def progress_bar(
    iterable: Iterable, 
    desc: str = "", 
    total: Optional[float] = None, 
    num_cols: int = 60, 
    disable: bool = False
) -> Iterable:
    
    bar_format = "{percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    if len(desc) > 0:
        bar_format = "{desc}: " + bar_format
    return tqdm(iterable, desc=desc, total=total, ncols=num_cols, disable=disable, bar_format=bar_format, leave=False)


def get_latest_ckpt_path(ckpts_path):
    ckpt_paths = [p for p in ckpts_path.glob("*.pt") if "best" not in p.name]
    error_msg = "Expected only one ckpt apart from best, found none or too many."
    assert len(ckpt_paths) == 1, error_msg

    ckpt_path = ckpt_paths[0]
    return ckpt_path
