# SÜ AZIA Zuberotarra Corpus

**Status**: integrated into training pipeline
**Date scraped**: 2026-06-11
**Source**: [www.suazia.com](http://www.suazia.com) (archived via Wayback Machine, 2011 snapshot)
**Scraper**: `nongoeuskara/build/scrape_suazia.py`
**Output**: `data/raw/text/suazia/suazia_train_clean.txt` (6,676 sentences)

## Overview

SÜ AZIA ("SU AZIA elkartea") was an association dedicated to the promotion of the
**Xiberotarra (Zuberoan/Souletin)** dialect of Basque. Their website (circa 2004–2011)
published pastoral play scripts, blog articles, and cultural event coverage — all in
Zuberotarra.

The Wayback Machine archived a snapshot from September 2011 that preserves these texts.

## Why this corpus?

Zuberera is the most severely under-represented subdialect in the Ahotsak training
data for the tier-3 azpieuskalki classifier:

| Azpieuskalki | Ahotsak sentences | With SÜ AZIA |
|---|---|---|
| mendebal-sortaldea | 13,059 | 13,059 |
| erdialde-sartaldea | 9,804 | 9,804 |
| zuberera | **750** | **~7,430** |
| ekialde-nafarra | 1,420 | 1,420 |

The 18x class imbalance between zuberera and mendebal-sortaldea makes it difficult
for the classifier to learn Zuberotarra's distinctive features (ü, verb forms in -zü,
lexicon like `jin` for `etorri`).

## Corpus contents

| Source | Content type | Sentences |
|---|---|---|
| Antso Handia (2004) | Pastoral script | 902 |
| Bereterretx (2005) | Pastoral script | 769 |
| Santa Engrazi (2006) | Pastoral script | 932 |
| Eñaut Elizagarai (2007) | Pastoral script | 825 |
| Xiberoko Jauna (2008) | Pastoral script | 764 |
| Belagileen Trajeria (2009) | Pastoral script | 1,019 |
| Xahakoa (2010) | Pastoral script | 1,037 |
| News/blog articles (2009-2011) | Journalistic Zuberotarra | ~332 |
| Association info pages | Informational text | ~96 |
| **Total** | | **6,676** |

## Data characteristics

### Strengths
- **Authentic Zuberotarra orthography**: Extensive use of `ü`, `-zü-` verb forms,
  Xiberotar lexicon (`jin`, `düzü`, `züan`, `hanitx`, etc.)
- **Large volume**: 6,676 sentences is nearly 9x the current 750 Ahotsak zuberera sentences
- **Diverse register**: Pastoral verse (formal/poetic) + blog prose (semi-formal journalistic)

### Limitations
- **Domain mismatch**: Pastoral texts are written poetry, not conversational speech.
  The model may learn Zuberotarra well for written text but generalization to spoken
  Zuberotarra (like the "Neská jin düzü" example) may be limited.
- **Noise**: Pastoral pages include verse number prefixes, stage directions, and
  occasional Spanish passages (authentic to the pastoral tradition — these are
  largely filtered out).
- **Single time snapshot**: All content from 2004-2011 period, one association.

### Impact on training

Since fastText relies on character n-grams, the pastoral texts provide valuable
**character-level** signal:
- `ü` trigrams (unique to Zuberotarra among Basque dialects)
- `zu`/`zü` verb suffixes
- Xiberotar phonological patterns (final -a → -ia, etc.)

These character patterns transfer well even when the surrounding text domain differs.

## Integration in training pipeline

The corpus is loaded by `src/data/train_azpieuskalki.py` via `inject_suazia_zuberera()`:

```python
# In prepare_azpieuskalki_data():
injected = inject_suazia_zuberera(azpi_sentences)
```

The injection happens **after** Ahotsak sentence extraction but **before** the
train/test split, so SÜ AZIA sentences participate in both training and testing.

## Re-scraping

To re-scrape the corpus (if Wayback Machine adds more archives):

```bash
cd nongoeuskara
uv run python build/scrape_suazia.py
```

Output goes to `data/raw/text/suazia/suazia_train_clean.txt`.

**Note**: The Wayback Machine rate-limits aggressively. The scraper uses 2-second
delays between requests and curl for reliability. Running the full scrape takes
~2-3 minutes.

## Wayback Machine URLs

The scraper uses a 2011-09-20 snapshot. URLs:

- Static pages (pastoral full texts): `id=57` through `id=67`, `id=78`
- Blog listing: `?Hizkuntza:Xiberotarra` and `?limitstart=5,10,15,20`

## Future improvements

1. **Spoken Zuberotarra**: Seek oral corpora (radio interviews, Iparralde oral
   history collections) to complement the written material
2. **Data augmentation**: Apply dialect-specific augmentation (ü substitution,
   morphological transformations) to increase per-token diversity
3. **Test set coverage**: Add Zuberotarra examples to `test_azpieuskalki.txt`
   (currently has 0 zuberera labels)

## Related files

- Scraper: `nongoeuskara/build/scrape_suazia.py`
- Training integration: `src/data/train_azpieuskalki.py` → `inject_suazia_zuberera()`
- Raw output: `data/raw/text/suazia/suazia_train_clean.txt`
- Individual raw files: `data/raw/text/suazia/*.txt`
