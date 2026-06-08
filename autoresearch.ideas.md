# Azpieuskalki Improvement Ideas

## Verified
- [x] ✓ Fix Nafarroa town CSV gap (53 towns added, 0 unmapped)
- [x] ✓ Fix hyphen normalization in town matching
- [x] ✓ Scrape hego-goi-nafarrera towns (180 passages, 28 towns)
- [x] ✓ Label verification audit: chain is clean, 2-way split simplifies Wikipedia's 4-5 nafar classes
- [x] ✗ Class balancing via oversampling (made things worse, 25.73%)

## Label Quality Improvements
- Split ipar-goi-nafarrera into 2: mendebalekoa (Bortziriak/Baztan/Malerreka/Leitzaldea) + iparraldekoa (Ultzamaldea). Currently mixed with Gipuzkoan border towns.
- Split hego-goi-nafarrera: true hegoaldekoa (Sakana) separate from ekialdekoa (Zaraitzu/Erronkari/Aezkoa)
- But: data scarcity makes finer splits risky. Sakana only has 71 passages, Zaraitzu 22, Erronkari 8.
- Consider removing Gipuzkoan border towns (errenteria, hondarribia, oiartzun) from ipar-goi-nafarrera — they're transitional and confuse the model

## Hyperparameter Tuning
- Larger dim (300-400) with more epochs (50-100) — current 200/25 may underfit
- Higher wordNgrams (3-4) to capture longer lexical patterns
- Try fastText autotune with longer duration (300s)
- Lower lr (0.1-0.2) with more epochs for better convergence

## Model Architecture
- Per-dialect submodels (was tried, got 95% macro avg but complex deployment)
- Ensemble of flat + hierarchical predictions
- Character n-gram focus: minn=2, maxn=5 for morphological patterns

## Data Augmentation
- Scrape remaining Nafarroa towns (especially ipar-goi-nafarrera from Nafarroa proper, not Gipuzkoa)
- The big ipar-goi-nafarrera boost (5,842 sentences) is from just 3 Gipuzkoan towns — these are transitional, not pure nafarrera

## Analysis
- Dialect continuum explains most errors: sartaldekoa↔debagoiena (160), sartaldeko-naf-lap↔beterri (165)
- These are geographically adjacent varieties; lexical overlap is high
- Text-only fastText may have hit a ceiling around 55% for 11-way classification
- Speech features (prosody, phonetics) would distinguish better than text alone
