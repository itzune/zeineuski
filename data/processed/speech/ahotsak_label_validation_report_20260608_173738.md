# Ahotsak Label Cross-Validation Report

Generated: 2026-06-08 17:37
Model: `hier_dialect_final.bin`
Total passages validated: 289

## Outcome Distribution

| Outcome | Count | % |
|---------|-------|---|
| agreement | 94 | 32.5% |
| agreement_medium | 5 | 1.7% |
| agreement_low | 67 | 23.2% |
| flag_mismatch | 100 | 34.6% |
| ambiguous | 23 | 8.0% |

**Total agreement (any confidence):** 166/289 (57.4%)
**High-confidence agreement:** 94/289 (32.5%)

## Per-Dialect Agreement

| Dialect | Total | Agreement | Mismatch | Ambiguous | Agreement % |
|---------|-------|-----------|----------|-----------|-------------|
| central | 77 | 61 | 12 | 4 | 79.2% |
| nav-lab | 43 | 37 | 6 | 0 | 86.0% |
| navarrese | 56 | 0 | 48 | 8 | 0.0% |
| souletin | 12 | 0 | 11 | 1 | 0.0% |
| western | 101 | 68 | 23 | 10 | 67.3% |

## Confusion Matrix

Municipality label → Text model prediction

| Muni \ Model | western | central | navarrese | nav-lab | souletin |
|---|---|---|---|---|---|
| western | 68 | 30 | 0 | 3 | 0 |
| central | 15 | 61 | 0 | 1 | 0 |
| navarrese | 4 | 34 | 0 | 18 | 0 |
| nav-lab | 0 | 6 | 0 | 37 | 0 |
| souletin | 0 | 0 | 0 | 12 | 0 |

## Top Disagreements (100 total)

| # | Town | Municipality | Model Pred | Model Conf | Transcription (first 120 chars) |
|---|------|-------------|------------|------------|----------------------------------|
| 1 | getaria | nav-lab (high) | central | 0.99 | - Eta beste jolasikan?- Hori, horrek, horrena…- Bestela, harrapatzen o korrikan o…- A, bai, bai. Apuill-apuillaka! Honbr |
| 2 | elgoibar | western (low) | central | 0.98 | - Eta han orduan esan dezu bizi izan ziñala hamazazpi urte arte. Eta gero kriada...?- Hamazortzi.- Hamazazpi-hemezortzi  |
| 3 | mendaro | western (low) | central | 0.98 | -Eta umetan eskolara nora juten ziñan?-Mendarora Azpilikuetara. Zortzixan behiñ, hamabostian. "Ba, oin ganauak zainttu b |
| 4 | domintxaine-berroeta | souletin (low) | nav-lab | 0.98 | - Eta orain lehenago baina gutxiago? Ele... elekatzen...- A ba...- Ez?- A ba, ba, ba, ba. Leheno oono, ez dakit zendako. |
| 5 | bergara | western (high) | central | 0.98 | Eta errondan eittia zer zan? Ba fiesta moduan, batzuk kantau ta, beste batzuk dantzan ein dda... Langintza horretan zebi |
| 6 | hondarribia | navarrese (low) | central | 0.98 | -Hara jun aurretik hemen eskolan ibili ziñen?-Bai, biñon, señorita Julikin, parbulitosekin. Han ikasi nuen eskribitzen,  |
| 7 | getaria | nav-lab (high) | central | 0.98 | - Eta, umetan eta, sei, zortzi, hamar urtekin eta, zeintzuk izaten zian zuen jolasak eta? Jolasik eta iten zenduen?- Bai |
| 8 | oiartzun | navarrese (low) | central | 0.97 | - 'Ta hoik zer zin, nik aitzia badut Diputaziyunak 'ro zila eskolak? Beste auzotan ez bezela diputaziyunak zila 'ro...-  |
| 9 | etxarri | souletin (low) | nav-lab | 0.97 | - Eta hor, baduzue gaztelu handia, jauregi handi bat...- Bai.- Hor, Arüe da hori, ez?- Ez, Etxarri da.- Etxarri da?- Lim |
| 10 | hondarribia | navarrese (low) | central | 0.97 | -… gero, noiz esango dizut /esangoizut/, hiru urtekin, oroitzen naiz, geo aldatu giñen, ta bizi giñen San Pedro kalian,  |
| 11 | etxarri | souletin (low) | nav-lab | 0.97 | - Anitz... anitz euskaldun da, hemen? Etxarrin eta...?- Etxarrin...- Euskaldun anitz zarete?- Ba, ba, pixka bat oono. Ga |
| 12 | etxarri | souletin (low) | nav-lab | 0.97 | - Eta nola deitzen zara?- Ni Cécile, "Xexil"... "Xexil".- "Xexil"?- Ba.- Eta e...- Etxeberri.- Etxeberri. Eta noiz jaio  |
| 13 | lezo | navarrese (low) | central | 0.95 | -Ni beti zea izandu naz, osea, nik bizkar hezurra oso hondatua dakat ba! Osea, haurra nitzala sehaskan egondu nitzan eny |
| 14 | elgoibar | western (low) | central | 0.95 | Bueno, hor, ni gehienbat juaten nitzan gero ya Elgoibarko Izarrara, bueno, elkartera, dantza-taldeko arduradun moduan-et |
| 15 | mutriku | western (low) | central | 0.93 | -Lehenengo esaten badidazue izen-abizenak eta noiz jaio zineten.-Gregorio Arreitaonaindia Ulazia. Jaixo nintzan “el año  |
| 16 | lezo | navarrese (low) | central | 0.93 | -Ze mouzko maistra zan Agustina Lizarazu?-Ona. Zoragarriya. Oi! Ze amon ona zen!…-Ona zen…-Amona, bai. Maitte zigun! Nol |
| 17 | bergara | western (high) | central | 0.93 | - Esan kriada nun egon ziñan aurretik...- Kriara bai oixe, kriara eon nintzan ni Azkortan, Azkortan giñuan aittan arrebi |
| 18 | mutriku | western (low) | central | 0.93 | -Jaio zinen erdiko kalian?-Erdiko Kalian, bai.-Zeure akorduan zelakua zan inguru haura?-Ba inguru haura zan edarra, plaz |
| 19 | domintxaine-berroeta | souletin (low) | nav-lab | 0.93 | - Eta zuk, zure semeari eta alabari, eta erakutsi diezu?- Ba, guti batekin... guti batekin. Heen, Domintxineko eskolan t |
| 20 | lezo | navarrese (low) | central | 0.92 | -Zertan ite zun lana aitak?-Atta zen mekanikua. Ajustadoria. Ta aittu zen lanian milla ta betzirehun ta hamalabian, aitt |

## Recommendations

1. **Training-ready:** 94 passages (agreement + high confidence) → use directly for speech training
2. **Include with caution:** 5 passages (agreement + medium confidence) → include but note lower reliability
3. **Low confidence agreement:** 67 passages → consider excluding from training, use for weakly-supervised evaluation
4. **Needs review:** 100 passages where municipality label ≠ model prediction → manual review recommended before use
5. **Ambiguous:** 23 passages where model has low confidence → may reflect transitional/contact dialects
