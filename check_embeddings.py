"""check_embeddings.py — Audit audio/text embedding quality in the local cache."""

import numpy as np
from pathlib import Path

cache_dir = Path('training/cache/embeddings/')
files = list(cache_dir.glob('*.npz'))

total_videos = len(files)
total_windows = 0
audio_zero_videos = 0
audio_zero_windows = 0
partial_audio_zero_videos = 0
partial_audio_zero_windows = 0
text_zero_videos = 0

for f in files:
    data = np.load(f)
    audio = data['audio_embs']
    text  = data['text_embs']
    n = len(data['labels'])
    total_windows += n

    audio_row_zero = ~np.any(audio, axis=1)
    n_audio_zero = int(audio_row_zero.sum())

    if n_audio_zero == n:
        audio_zero_videos += 1
        audio_zero_windows += n
    elif n_audio_zero > 0:
        partial_audio_zero_videos += 1
        partial_audio_zero_windows += n_audio_zero

    if not np.any(text):
        text_zero_videos += 1

print(f"Videos total:               {total_videos:,}")
print(f"Windows total:              {total_windows:,}")
print()
print(f"=== Audio Embedding Quality ===")
print(f"Fully zero-audio videos:    {audio_zero_videos:,}  ({audio_zero_videos/total_videos*100:.1f}%)")
print(f"Partially zero-audio:       {partial_audio_zero_videos:,}  ({partial_audio_zero_videos/total_videos*100:.1f}%)")
print(f"Zero-audio windows total:   {audio_zero_windows + partial_audio_zero_windows:,}  ({(audio_zero_windows+partial_audio_zero_windows)/total_windows*100:.1f}%)")
print()
print(f"=== Text Embedding Quality ===")
print(f"Fully zero-text videos:     {text_zero_videos:,}  ({text_zero_videos/total_videos*100:.1f}%)")
