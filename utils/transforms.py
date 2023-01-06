import torch
import numpy as np
from PIL import Image


def tensor2PIL(tsr: torch.Tensor):
    tsr = torch.clamp(tsr.detach().cpu(), min=-1, max=1)
    assert len(tsr.shape) == 4
    batch_size = tsr.shape[0]
    ret = []
    for c in range(batch_size):
        arr = tsr[c].numpy().copy()
        arr = np.around((arr + 1.0) / 2.0 * 255).astype(int)
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(1, 2)
        ret.append(Image.fromarray(np.uint8(arr)))
    return ret