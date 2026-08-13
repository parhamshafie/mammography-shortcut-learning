# -*- coding: utf-8 -*-
"""Loss for the binary task: plain binary cross-entropy (Sec. 2.4)."""


def compute_loss(logits, targets):
    import torch.nn.functional as F
    return F.binary_cross_entropy_with_logits(logits, targets)
