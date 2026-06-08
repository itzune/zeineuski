# Azpieuskalki Improvement Ideas

## Completed
- [x] ✓ Ahotsak-aligned classification: 68 town mismatches fixed, 10 classes, 68.32%
- [x] ✓ Slow training (lr=0.2, epoch=50): 72.43% — massive improvement over autotune
- [x] ✓ NS vs HS comparison: NS wins (72.43% vs 68.40%) — HS trades 10pp large-class for small-class gain
- [x] ✓ NS+HS ensemble: 70.91% — doesn't beat solo NS
- [x] ✓ OVA loss: 71.58% — slightly worse than NS
- [x] ✓ dim=300: 67.68-71.93% — always worse than dim=200
- [x] ✓ wordNgrams=3: 64.09% — overfits, collapses small classes
- [x] ✓ wordNgrams=1: 72.14% — slightly worse than bigrams
- [x] ✓ minCount=2: 66.79% — filters out too much signal
- [x] ✓ char n-grams (minn=2,maxn=5): 67.50% — no help
- [x] ✓ lrUpdateRate variation: no effect
- [x] ✗ Oversampling: collapsed to 25.73%

## Optimal Config (72.43%)
- Model: fastText supervised
- dim=200, lr=0.2, epoch=50, wordNgrams=2, loss=ns, minCount=1
- NO autotune (autotune lr decay is too aggressive for imbalanced data)
- 11 classes, 35K train sentences

## Current Per-Class Performance (best model)
| Class | Train | Test | Accuracy |
|---|---|---|---|
| mendebal-sortaldea | 13,059 | 2,304 | 86-87% |
| erdialde-sartaldea | 9,804 | 1,729 | 77% |
| erdialde-sortaldea | 4,966 | 876 | 64% |
| nafar-sortaldea | 1,516 | 267 | 56-59% |
| naflap-sortaldea | 1,395 | 246 | 61-63% |
| nafar-hego-sartaldea | 1,101 | 194 | 32% |
| naflap-sartaldea | 726 | 127 | 41-49% |
| ekialde-nafarra | 710 | 125 | 33-38% |
| nafar-erdigunea | 497 | 87 | 18-22% |
| mendebal-sartaldea | 439 | 77 | 34-36% |
| zuberera | 375 | 66 | 45-49% |

## Data Expansion Opportunities
- [ ] **Scrape nafar-ipar-sartaldea towns** (arantza 43, bera 31, etxalar 30, igantzi 27, lesaka 34 transcriptions). These are Bortziriak/Malerreka towns — currently 0 passages for this class. ~165 passages → ~2,600 sentences.
- [ ] Scrape Zuberoa towns: barkoxe, eskiula, maule, atharratze, etc. Currently only 27 passages. Could boost zuberera from 375→1,000+ train.
- [ ] Scrape remaining Nafarroa towns: 53 unscraped. Many are in unscraped regions.
- [ ] **BLOCKED**: Scraper has bug — `save_passages` function undefined. Need to fix before scraping.

## Architecture Ideas
- Per-dialect submodels (previously tried, complex deployment)
- Two-tier: dialect classifier → per-dialect azpieuskalki classifier (reduces class count per model)
- Weighted loss: upweight small classes via `autotuneModelSize` with validation file
- Pre-trained Basque embeddings (cc.eu.300.bin) as initialization — might inject general Basque knowledge

## Analysis
- Dialect continuum: nafar-erdigunea most confused with nafar-sortaldea (22 times) — geographically adjacent
- Text-only fastText ceiling: ~72-73% for 11-way azpieuskalki classification
- Core Bizkaian/Gipuzkoan classes are very strong (86-87%)
- Nafarroan/Labourdin/Zuberera classes bottlenecked by data quantity, not model capacity
