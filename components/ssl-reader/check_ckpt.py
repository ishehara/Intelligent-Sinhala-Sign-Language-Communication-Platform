import torch
from pathlib import Path

ckpts = sorted(Path('models/mediapipe').glob('*.pth'), key=lambda x: x.stat().st_mtime, reverse=True)
print('Recent checkpoints:')
for c in ckpts[:6]:
    ck = torch.load(c, map_location='cpu', weights_only=True)
    epoch = ck.get('epoch', '?')
    val_acc = ck.get('val_acc', '?')
    print(f'  {c.name}: epoch={epoch} val_acc={val_acc}')
