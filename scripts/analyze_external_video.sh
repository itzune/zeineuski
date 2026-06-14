#!/bin/bash
# GPU server — analyze external video: extract audio, segment, classify
set -euo pipefail

VIDEO_IN=/root/zeineuski/data/raw/speech/external/ahots-hariak_hd.mp4
OUT_DIR=/root/zeineuski/data/processed/speech/external_analysis
export PATH=$HOME/.local/bin:$PATH
cd /root/zeineuski

MANIFEST="$OUT_DIR/manifest.csv"
SEGMENT_LENGTH=10
OVERLAP=2

mkdir -p "$OUT_DIR/segments"

echo "=== Step 1: Extract audio ==="
AUDIO_FULL="$OUT_DIR/full_audio.wav"
ffmpeg -y -i "$VIDEO_IN" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_FULL" 2>&1 | tail -2

DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO_FULL")
echo "Audio: ${DURATION}s, 16kHz mono"

echo ""
echo "=== Step 2: Segment into ${SEGMENT_LENGTH}s chunks with ${OVERLAP}s overlap ==="
uv run python -c "
import soundfile as sf, numpy as np
from pathlib import Path

audio, sr = sf.read('$AUDIO_FULL')
assert sr == 16000
total_samples = len(audio)

seg_samples = $SEGMENT_LENGTH * sr
hop_samples = ($SEGMENT_LENGTH - $OVERLAP) * sr
out_dir = Path('$OUT_DIR/segments')
lines = ['path,start_sec,end_sec,dialect']
seg_idx = 0

for start in range(0, total_samples, hop_samples):
    end = min(start + seg_samples, total_samples)
    chunk = audio[start:end]
    if len(chunk) < 2 * sr:
        continue
    seg_path = out_dir / f'segment_{seg_idx:04d}.wav'
    sf.write(str(seg_path), chunk, sr)
    lines.append(f'{seg_path.absolute()},{start/sr:.1f},{end/sr:.1f},unknown')
    seg_idx += 1

Path('$MANIFEST').write_text('\n'.join(lines) + '\n')
print(f'{seg_idx} segments')
"

echo ""
echo "=== Step 3: Extract Whisper embeddings (GPU) ==="
uv run python -m src.models.speech.whisper_did extract \
    --manifest "$MANIFEST" \
    --output "$OUT_DIR/embeddings.pkl" \
    --device cuda

echo ""
echo "=== Step 4: Classify each segment ==="
uv run python -c "
import pickle, json, torch
import numpy as np
from collections import Counter
from src.models.speech.whisper_did import load_speech_model

encoder, mlp, label_encoder, scaler, config = load_speech_model(
    'models/speech/whisper_dialect_merged', 'cuda'
)

with open('$OUT_DIR/embeddings.pkl', 'rb') as f:
    samples = pickle.load(f)

DIALECT_NAMES = {
    'western': 'Mendebaldekoa / Bizkaiera',
    'central': 'Erdialdekoa / Gipuzkera',
    'navarrese': 'Nafarrera',
    'nav-lab': 'Napar-Lapurtera',
    'souletin': 'Zuberera',
}

results = []
for sample in samples:
    emb = scaler.transform(sample['embedding'].reshape(1, -1))
    X = torch.tensor(emb, dtype=torch.float32).to('cuda')
    with torch.no_grad():
        logits = mlp(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze(0)
    pred_idx = probs.argmax()
    dialect = label_encoder.classes_[pred_idx]
    results.append({
        'start_sec': float(sample.get('start_sec', 0)),
        'end_sec': float(sample.get('end_sec', 0)),
        'dialect': dialect,
        'dialect_name': DIALECT_NAMES.get(dialect, dialect),
        'confidence': float(probs[pred_idx]),
        'probs': {label_encoder.classes_[i]: float(p) for i, p in enumerate(probs)},
    })

with open('$OUT_DIR/results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Summary
counts = Counter(r['dialect'] for r in results)
total = len(results)
print(f'\n=== Results for {total} segments ===')
for dialect, count in counts.most_common():
    pct = 100 * count / total
    name = DIALECT_NAMES.get(dialect, dialect)
    avg_conf = np.mean([r['confidence'] for r in results if r['dialect'] == dialect])
    print(f'  {dialect:12s} ({name:30s}) {count:4d} ({pct:5.1f}%)  conf={avg_conf:.3f}')

# Full timeline
print(f'\n=== Full timeline ===')
for r in results:
    ts = f'[{r[\"start_sec\"]:6.0f}s - {r[\"end_sec\"]:6.0f}s]'
    print(f'  {ts} {r[\"dialect\"]:12s} ({r[\"confidence\"]:.2f})')

print(f'\nDone → $OUT_DIR/results.json')
"

echo "=== Done ==="
