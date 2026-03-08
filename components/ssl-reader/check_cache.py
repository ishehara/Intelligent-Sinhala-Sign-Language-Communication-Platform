import pickle, numpy as np, random
from pathlib import Path

cache_dir = Path('data/processed/mediapipe_normalized')
files = list(cache_dir.glob('*.pkl'))
random.seed(42)
sample = random.sample(files, 20)

print(f"Total cached files: {len(files)}")
print()

face_all_zero = 0
hand_all_zero = 0
for f in sample:
    d = pickle.load(open(f, 'rb'))
    hm = d[:, :126].mean()
    fm = d[:, 126:].mean()
    zeros = (d == 0).all(axis=1).sum()
    if fm == 0: face_all_zero += 1
    if hm == 0: hand_all_zero += 1
    print(f"  {f.name[:40]}: hand_mean={hm:.4f} face_mean={fm:.4f} zero_rows={zeros}/{d.shape[0]}")

print()
print(f"Files with face all zeros: {face_all_zero}/20")
print(f"Files with hand all zeros: {hand_all_zero}/20")
