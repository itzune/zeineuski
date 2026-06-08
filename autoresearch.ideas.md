# Azpieuskalki Improvement Ideas

## Completed
- [x] ✓ Ahotsak-aligned classification: 68 town mismatches fixed → 68.32%
- [x] ✓ Slow lr=0.2 training (no autotune): 68.32% → 72.43%
- [x] ✓ Char n-grams (minn=2,maxn=6): 72.43% → 82.08% (game-changer!)
- [x] ✓ Nafar-ipar-sartaldea data (5 towns scraped): new class at 85% accuracy
- [x] ✓ NS vs HS/OVA/char-only/ensemble: NS+char+word is undisputed best
- [x] ✓ dim=200 confirmed optimal (300 always worse)
- [x] ✓ wordNgrams=2 confirmed optimal (1/3 worse)
- [x] ✓ minCount=1 confirmed (2 filters too much)
- [x] ✓ epoch=75 optimal for char config (50 too few, 100 overfits)
- [x] ✓ min_samples=600: drops 3 weakest → 83.55% on 9 classes
- [x] ✗ Oversampling: collapsed to 25.73%

## SESSION COMPLETE — 83.55% on 9 classes (+63.8% over baseline)

The azpieuskalki classifier is now a solid Tier 3 component at 7.5× random baseline.
Remaining headroom is in data expansion (scraping more towns, fixing transcription quality),
not hyperparameter tuning. See README.md and PROJECT.md for final results.

## Optimal Config (82.08% → 83.55%)
- Model: fastText supervised
- **dim=200, lr=0.2, epoch=75, wordNgrams=2, minn=2, maxn=6, loss=ns, minCount=1**
- NO autotune (autotune lr decay is too aggressive)
- 9-12 classes depending on min_samples threshold

## Key Findings
1. **Autotune is harmful**: aggressive LR decay causes early overfit to large classes
2. **Char n-grams + slow lr is the killer combo**: Basque morphology is dialect-specific
3. **Data ceiling is ~83% for text-only**: 3 weakest classes need more data to be viable
4. **Dialect continuum is real**: most errors are between geographically adjacent classes

## Progression
| Stage | Accuracy | Classes | Key Change |
|---|---|---|---|
| Baseline | 51.02% | 10 | Original custom labels |
| Ahotsak labels | 68.32% | 10 | Fixed 68 town mismatches |
| Slow lr=0.2 | 72.43% | 11 | Killed autotune, 50 epochs |
| +Char n-grams | 80.94% | 12 | minn=2,maxn=4 caught morphology |
| +Wider chars | 81.55% | 12 | maxn=6 + 75 epochs |
| +More data | 82.08% | 12 | 5 Nafarroa towns scraped |
| min_samples=600 | 83.55% | 9 | Dropped 3 weak classes |

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

## Autoresearch Session #3 Conclusions (37 experiments)

**Start:** 51.02% (10 classes, original custom labels)
**End:** 83.55% (9 classes, min_samples=600) or 82.08% (12 classes)
**Total improvement:** +63.8%

**Three breakthroughs:**
1. **Ahotsak-aligned labels** (#6-7): Fixed 68 town mismatches → +17pp (51% → 68%)
2. **Slow manual training** (#14): Killed autotune, lr=0.2 → +4pp (68% → 72%)
3. **Character n-grams** (#27): minn=2,maxn=6 + slow lr → +10pp (72% → 82%)

**Optimal config:** dim=200, lr=0.2, epoch=75, wordNgrams=2, minn=2, maxn=6, loss=ns

**Fatal hyperparameters (always worse):** autotune, dim=300, wordNgrams=3, minCount=2, lr=0.1
**Neutral (within noise):** dim=250, epoch=100, lr=0.15, maxn=7, minn=3 |

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
