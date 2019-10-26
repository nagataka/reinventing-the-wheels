import torch
import numpy as np

def img2tensor(x: np.ndarray, device):
    """Convert image (numpy array with WxHx3) to PyTorch tensor
    """
    x = torch.from_numpy(x).to(device, dtype=torch.float)
    x = x.permute(2,0,1).unsqueeze_(0)

    return x
