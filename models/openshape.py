import torch
import torch.nn as nn
from .ppat import Projected, PointPatchTransformer

def PointBertG14():
    model = Projected(
        PointPatchTransformer(512, 12, 8, 512 * 3, 256, 384, 0.2, 64, 6),
        nn.Linear(512, 1280),
    )
    return model