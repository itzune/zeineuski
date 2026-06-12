# Azpieuskalki F1 Optimization — Deferred Ideas

Session: 2026-06-12 · 16 runs (3 keep, 13 discard) · Best weighted_f1 = 0.8255

## What worked
- **Targeted oversampling factor=-2** — only oversample bottom-half classes (below median count) to max/2 level. Boosts weak classes without hurting strong ones. +0.0026 weighted_f1, +1.8pp bottom5_mean.
- **Mendebal-sartaldea data expansion** — scraped 19 new towns from Ahotsak (+167 passages). Doubled that class's recall (0.377→0.766). F1 from 0.506→0.792. No weighted F1 gain because class is small, but huge per-class improvement.

## What didn't work
- `wordNgrams=3` — identical to bigrams
- `minn=3, maxn=8` — worse, minn=2 needed for Basque short morphemes
- `epoch=150, lr=0.1` — identical, 2× cost
- `lr=0.3` — slightly worse
- `minCount=2` — slightly worse, rare dialect words matter
- `loss=ova` — within noise of ns
- `loss=hs` — catastrophic -5.2%
- `oversample_factor=4` (flat) — catastrophic -3.5%
- `dim=300` — identical, 2× cost
- `targeted oversample factor=-1` — too aggressive, top classes degrade

## Best config
```
loss=ns, dim=200, epoch=100, lr=0.2, wordNgrams=2, minn=2, maxn=6,
minCount=1, targeted_oversample=-2
```

## Remaining problem classes (0.64-0.70 F1)
- nafar-erdigunea (0.638) — no more Ahotsak data available
- ekialde-nafarra (0.64 range) — only 125 test samples
- nafar-hego-sartaldea (0.670) — limited data
- naflap-sartaldea (0.695) — limited Iparralde data

## Promising untested ideas
- Two-stage classifier: broad dialect group → sub-dialect within group (would eliminate cross-group confusions)
- Data augmentation via backtranslation for Nafarroa minority classes
- External corpora: Klasikoak.eus, Mintzoak.eus for Nafarroa/Iparralde texts
- `bucket=50000` on the targeted-oversampled model (could be better with more balanced data)
- Quantize the best model for deployment (34MB vs 251MB, ~1% F1 cost)
