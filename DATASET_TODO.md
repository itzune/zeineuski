# Dataset Improvement TODO

Pending tasks and ideas for improving the Basque dialect identification corpus.

---

## ✅ Done (2026-06-12)

| Task | Result |
|---|---|
| Fix hyphen-vs-space normalization | `_normalize_town()` in scraper. Backfilled 21 previously-unlabeled passages (Etxarri Aranatz, Espartza, Uharte-Arakil). |
| Scrape mendebal-sartaldea towns (phase 1) | Zeberio, Orozko, Bilbo, Igorre, Mungia, Derio, Galdakao, Getxo. +903 train (439→1,342). |
| Scrape mendebal-sartaldea towns (phase 2) | 19 additional sartaldekoa-m towns (Arrankudiaga, Arrigorriaga, Erandio, Sopela, Morga, etc.). +2,294 train (1,342→3,636). F1: 0.506→0.792 (+28.6pp!). |
| Scrape nafar-erdigunea towns | Odieta, Lantz, Txulapain, Anue, Ezkabarte, Olaibar. +790 train (497→1,287). |
| Scrape naflap-sartaldea towns | Arbona, Sara, Ainhoa, Donibane Lohizune, Baigorri. +145 train (726→871). |
| Restore nafar-ipar-sartaldea passages | 165 passages (Arantza 43, Bera 31, Etxalar 30, Igantzi 27, Lesaka 34) were accidentally dropped during the backfill merge. Restored from `213742.jsonl`. Class back in the 12-class model. |
| Retrain with optimal config | 12-class model, `dim=200/lr=0.2/epoch=75/minn=2/maxn=6/loss=ns`. Overall accuracy **82.38%**. |

**Current 12-class state (best model: targeted oversample=-2, epoch=100):**

```
Class                      Train    Test     F1    Notes
──────────────────────────────────────────────────────────────────
zuberera                    6050    1067   95.1%   +31.7pp ★ (SU AZIA corpus)
mendebal-sortaldea         13059    2304   86.8%   largest class
nafar-ipar-sortaldea        1965     346   82.3%   stable
──────────────────────────────────────────────────────────────────
erdialde-sartaldea          9804    1729   81.5%   good
mendebal-sartaldea          3636     641   79.2%   +28.6pp ★ rescued (phase 2)
erdialde-sortaldea          4966     876   79.1%   good
naflap-sortaldea            1395     246   74.0%   improved
naflap-sartaldea             871     153   70.3%   +4.6pp (targeted oversampling)
nafar-sortaldea             1516     267   70.5%   improved
nafar-hego-sartaldea        1101     194   66.8%   limited data
nafar-erdigunea             1287     227   63.8%   🔴 no more Ahotsak data
ekialde-nafarra              710     125   62.5%   🔴 tiny class, high variance
```
nafar-sortaldea             1516     267   66.7%    -9.7pp ⚠
naflap-sartaldea             871     153   64.0%    -3.0pp
ekialde-nafarra              710     125   63.1%    -2.5pp
nafar-erdigunea             1287     227   56.1%   +21.6pp ★ rescued
nafar-hego-sartaldea        1101     194   55.8%    +0.6pp
```

---

## 🔴 Next Priority: Nafar-sortaldea regression (-9.7pp)

Went from 76.4%→66.7% F1 despite zero data changes. Now the single biggest unexplained loss.
This class was solid at tier-2 but now competes against 11 other azpieuskalkiak instead of 8.

- [ ] **Build confusion matrix** — which classes are absorbing nafar-sortaldea's errors?
- [ ] If confusion is with nafar-hego-sartaldea / nafar-erdigunea, those are genuinely confusable transitional Nafarroa zones
- [ ] If confusion is with the new mendebal-sartaldea / nafar-erdigunea, a two-tier model would largely fix it
- [ ] If confusion is with erdialde-sortaldea (eastern Gipuzkoa), that's a known transitional continuum effect

---

## 🟡 Ahotsak-Exhausted Classes

All weak classes are now fully scraped. These are at the data ceiling on Ahotsak.

| Class | Train | F1 | Ahotsak tc | Alternative sources |
|---|---|---|---|---|
| nafar-hego-sartaldea | 1,101 | 55.8% | 58 | Klasikoak (Sakana), old journalistic texts |
| nafar-erdigunea | 1,287 | 56.1% | 64 | Klasikoak (Ultzamaldea sermons), Pamplona texts |
| ekialde-nafarra | 710 | 63.1% | 30 | Klasikoak (Erronkari/Zaraitzu), Oroitzapenak at EKE, classical 18th c. catechisms |
| naflap-sartaldea | 871 | 64.0% | 59 | Mintzoak.eus (EKE oral archive), Lapurdiko klasikoak |
| mendebal-sartaldea | 1,342 | 68.3% | 127 | Bizkaiera dotrina (old Western Bizkaian catechisms) |

- [ ] **Investigate Klasikoak corpus** (klasikoak.eus) — has dialect-annotated classical texts
- [ ] **Check Mintzoak.eus** for downloadable Iparralde transcriptions

---

## 🟢 Additional Ahotsak Data Available (but not priority)

**150 unmapped towns** (not in `municipality_dialect.csv`) with 4,299 transcriptions:

| Likely dialect | Transcriptions | Top towns |
|---|---|---|
| erdialde (Gipuzkoa) | 2,619 | Aretxabaleta 516, Antzuola 307, Elgeta 233, Soraluze 233, Eskoriatza 218 |
| mendebal (Bizkaia) | 892 | Markina-Xemein 150, Mallabia 136, Gernika 122, Amorebieta-Etxano 104, Muxika 103 |
| naflap (Iparralde) | 66 | Gamarte 66 |
| Other | 722 | Nafarroa towns, Araba |

These would strengthen **already-solid** classes, not fix the weak ones. Adding them requires:
- [ ] Fill `eskualdea` (region) column for Nafarroa towns in `municipality_dialect.csv`
- [ ] Add Gipuzkoa/Bizkaia towns to CSV with correct regional dialect assignments
- Only then scraping would be useful

**0 unmapped towns map to nafar classes** (the Nafarroa towns have no `eskualdea` in the CSV). If those ~700 Nafarroa transcriptions map to nafar-erdigunea, nafar-hego-sartaldea, or nafar-sortaldea via proper `eskualdea` assignments, they'd be worth scraping.

---

## 💡 Structural improvements (no new data needed)

- [ ] **Two-tier classification**: 5 macro-dialect model → per-dialect azpieuskalki. Would eliminate cross-tier confusions (ekialde-nafarra vs naflap classes never competing).
- [ ] **Speaker-disjoint split audit**: small classes (30 tc ekialde-nafarra) may have single-speaker leakage inflating accuracy.
- [ ] **Confusion matrix analysis**: understand where each weak class's errors go.
- [ ] **Targeted oversampling**: bottom 5 classes with appropriate `loss=ova` regularization.

---

_Last updated: 2026-06-12_
