# -*- coding: utf-8 -*-
"""EfficientNet-B5 encoder (Mammo-CLIP pretrained) + a linear head. n_outputs=1 gives the binary
sigmoid classifier; n_outputs=3 gives the dataset-origin softmax classifier -- same architecture,
different head width."""
import logging
from pathlib import Path

log = logging.getLogger("mammo_pipeline")


def build_model(cfg, n_outputs=1):
    import torch.nn as nn, torch
    from efficientnet_pytorch import EfficientNet
    bk = cfg["backbone"]
    ckpt = bk.get("mammo_clip_checkpoint")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.amp_enabled = False
            if ckpt is None and bk.get("imagenet_fallback", True):
                self.encoder = EfficientNet.from_pretrained("efficientnet-b5")
            else:
                self.encoder = EfficientNet.from_name("efficientnet-b5")
            feat_dim = self.encoder._fc.in_features
            self.encoder._fc = nn.Identity()
            self.encoder._dropout = nn.Identity()
            hidden = bk.get("head_hidden")
            if hidden:
                self.head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden),
                                          nn.GELU(), nn.Dropout(bk["head_dropout"]),
                                          nn.Linear(hidden, n_outputs))
            else:
                self.head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(bk["head_dropout"]),
                                          nn.Linear(feat_dim, n_outputs))

        def set_trainable(self, train_encoder, last_blocks):
            for p in self.encoder.parameters():
                p.requires_grad = False
            if train_encoder:
                blocks = self.encoder.parameters() if last_blocks is None else \
                    (p for blk in list(self.encoder._blocks)[-last_blocks:] for p in blk.parameters())
                for p in blocks:
                    p.requires_grad = True
            for p in self.head.parameters():
                p.requires_grad = True

        def forward(self, x):
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                feat = self.encoder.extract_features(x)
                feat = self.encoder._avg_pooling(feat).flatten(1)
                out = self.head(feat)
                return out.squeeze(-1) if n_outputs == 1 else out

    net = Net()
    if ckpt:
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Mammo-CLIP checkpoint not found: {ckpt}")
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = c["model"] if (isinstance(c, dict) and "model" in c) else c
        enc = {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")}
        miss, unexp = net.encoder.load_state_dict(enc, strict=False)
        tot = len(net.encoder.state_dict())
        log.info("Mammo-CLIP encoder: matched ~%d/%d (missing=%d unexpected=%d)",
                 tot - len(miss), tot, len(miss), len(unexp))
        if (tot - len(miss)) < 0.9 * tot:
            log.warning("[WARN] <90%% of encoder weights matched -- check the checkpoint's key names.")
    net.set_trainable(bk["train_encoder"], bk["last_blocks"])
    return net
