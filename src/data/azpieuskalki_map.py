"""
Azpieuskalki mapping for Basque dialectal regions.
Based on Ahotsak.eus / Koldo Zuazo's dialect classification:
  - https://ahotsak.eus/euskalkiak

Matches the Ahotsak.eus sub-dialect scheme exactly.
Town assignments verified against official Ahotsak azpieuskalki pages (2026-06-08).

Key corrections from Ahotsak data (68 mismatches fixed):
  - Deba valley (arrasate, bergara, eibar, etc.) → sortaldekoa-m (Bizkaian), not sartaldekoa-g
  - Urola coast (zarautz, zumaia, azpeitia, etc.) → sartaldekoa-g, not erdigunekoa-g
  - Leitzaldea (araitz, leitza, larraun, etc.) → sortaldekoa-g (Gipuzkoan), not erdigunekoa-n
  - Bidasoa Behea (hondarribia, irun) → sortaldekoa-g (Gipuzkoan), not ipar-sartaldekoa
  - Baztan (urdazubi, zugarramurdi) → sartaldekoa-nl (Navarro-Labourdin), not baztangoa
  - Tolosa → erdigunekoa-g (coastal Gipuzkoan), not sortaldekoa-g

The authoritative town→azpieuskalki assignments are stored in
  data/reference/ahotsak_azpieuskalki_towns.json
and in the 'notes' field of data/reference/municipality_dialect.csv as
"ahotsak:<azpieuskalki_label>".

The train_azpieuskalki.py pipeline uses AHOTSAK_TO_OUR_LABEL to convert
Ahotsak azpieuskalki labels to our internal training labels.
"""

# Mapping: Ahotsak azpieuskalki → our training label
# Small classes (<300 train samples) merged into larger siblings.
# Only naflap-erdigunea (155) merged → naflap-sartaldea.
AHOTSAK_TO_OUR_LABEL = {
    # Mendebalekoa / Bizkaiera
    "sartaldekoa-m": "mendebal-sartaldea",
    "sortaldekoa-m": "mendebal-sortaldea",
    "tartekoa-m": "mendebal-sortaldea",       # merged: tartekoa → sortaldekoa (small, transitional)
    # Erdialdekoa / Gipuzkera  
    "erdigunekoa-g": "erdialde-sartaldea",    # merged: erdigunekoa → sartaldekoa (coastal clustered)
    "sartaldekoa-g": "erdialde-sartaldea",
    "sortaldekoa-g": "erdialde-sortaldea",
    # Nafarra
    "baztangoa": "nafar-sortaldea",           # merged: baztangoa → sortaldekoa (only Baztan town)
    "erdigunekoa-n": "nafar-erdigunea",
    "hegoaldeko-nafarra": "nafar-erdigunea",  # merged: hegoaldekoa → erdigunea
    "hego-sartaldekoa": "nafar-hego-sartaldea",
    "ipar-sartaldekoa": "nafar-ipar-sartaldea",
    "sortaldekoa-n": "nafar-sortaldea",
    # Nafar-lapurtarra
    "erdigunekoa-nl": "naflap-sartaldea",      # merged: erdigunekoa → sartaldekoa
    "sartaldekoa-nl": "naflap-sartaldea",
    "sortaldekoa-nl": "naflap-sortaldea",
    # Zuberotarra
    "basaburua": "zuberera",                  # merged: basaburua + pettarra → single zuberera
    "pettarrakoa": "zuberera",
    # Ekialdeko nafarra
    "zaraitzukoa": "ekialde-nafarra",
    "erronkarikoa": "ekialde-nafarra",
}

# Human-readable names — aligned with Ahotsak.eus classification
# Small Ahotsak classes merged into siblings for trainability (see AHOTSAK_TO_OUR_LABEL)
AZPIEUSKALKI_NAMES = {
    # Mendebalekoa / Bizkaiera
    "mendebal-sartaldea":  "Mendebal-sartaldea (Western Bizkaian)",
    "mendebal-sortaldea":  "Mendebal-sortaldea (Eastern Bizkaian + transitional)",
    # Erdialdekoa / Gipuzkera
    "erdialde-sartaldea":  "Erdialde-sartaldea (coastal+western Gipuzkoan)",
    "erdialde-sortaldea":  "Erdialde-sortaldea (eastern Gipuzkoan)",
    # Nafarra
    "nafar-ipar-sartaldea": "Nafar ipar-sartaldea (Bortziriak/Malerreka)",
    "nafar-erdigunea":     "Nafar erdigunea (central Navarre)",
    "nafar-hego-sartaldea": "Nafar hego-sartaldea (Sakana)",
    "nafar-sortaldea":     "Nafar sortaldea (eastern Navarre)",
    # Nafar-lapurtarra
    "naflap-sartaldea":    "Nafar-lapur sartaldea (coastal Labourdin)",
    "naflap-sortaldea":    "Nafar-lapur sortaldea (Basse-Navarre)",
    # Zuberotarra
    "zuberera":            "Zuberera (Souletin)",
    # Ekialdeko nafarra
    "ekialde-nafarra":     "Ekialdeko nafarra (Zaraitzu/Erronkari)",
}

# Legacy region-based mapping: eskualdea → azpieuskalki (used as fallback)
# Corrected based on Ahotsak town assignments (2026-06-08)
AZPIEUSKALKI_MAP = {
    # Mendebalekoa / Bizkaiera
    "Bilbo Handia":        "mendebal-sartaldea",
    "Getxo":               "mendebal-sartaldea",
    "Mungialdea":          "mendebal-sartaldea",
    "Txorierri":           "mendebal-sartaldea",
    "Arratia-Nerbioi":     "mendebal-sortaldea",
    "Durangaldea":         "mendebal-sortaldea",
    "Lea-Artibai":         "mendebal-sortaldea",
    "Busturialdea":        "mendebal-sortaldea",
    "Gorbeialdea":         "mendebal-tartekoa",     # Legutio/Zigoitia → tartekoa-m per Ahotsak
    # Deba valley (Ahotsak: sortaldekoa-m, not sartaldekoa-g)
    "Debagoiena":          "mendebal-sortaldea",
    "Debabarrena":         "mendebal-sortaldea",

    # Erdialdekoa / Gipuzkera
    "Donostialdea":        "erdialde-erdigunea",
    "Tolosaldea":          "erdialde-erdigunea",    # Tolosa → erdigunekoa-g per Ahotsak
    # Urola (Ahotsak: sartaldekoa-g, not erdigunekoa-g or sortaldekoa-g)
    "Urola Kosta":         "erdialde-sartaldea",
    "Urola Erdia":         "erdialde-sartaldea",
    "Urola Garaia":        "erdialde-sartaldea",
    "Goierri":             "erdialde-sartaldea",    # Ordizia → sartaldekoa-g per Ahotsak
    # Eastern Gipuzkoa (Ahotsak: sortaldekoa-g)
    "Bidasoa Behea":       "erdialde-sortaldea",    # Irun/Hondarribia → sortaldekoa-g
    "Oiartzualdea":        "erdialde-sortaldea",    # Errenteria/Oiartzun → sortaldekoa-g
    "Leitzaldea":          "erdialde-sortaldea",    # Leitza/Larraun → sortaldekoa-g (Gipuzkoan!)
    "Ultzamaldea":         "erdialde-sortaldea",    # Imotz/Basaburua → sortaldekoa-g

    # Nafarra
    "Bortziriak":          "nafar-ipar-sartaldea",
    "Malerreka":           "nafar-ipar-sartaldea",
    "Baztan":              "nafar-baztangoa",
    "Sakana":              "nafar-hego-sartaldea",
    "Artzibar":            "nafar-hegoaldea",       # Ahotsak: hegoaldeko-nafarra
    "Ezkabarte":           "nafar-hegoaldea",
    "Erroibar":            "nafar-sortaldea",
    "Esteribar":           "nafar-sortaldea",
    "Aezkoa":              "nafar-sortaldea",

    # Nafar-lapurtarra
    "Donibane Lohizune-Hendaiako Kantonamendua": "naflap-sartaldea",
    "Hiriburuko Kantonamendua":  "naflap-sartaldea",
    "Biarritz-eko Kantonamendua": "naflap-sartaldea",
    "Uztaritzeko Kantonamendua": "naflap-sartaldea",
    "Hazparneko Kantonamendua": "naflap-sartaldea",
    "Baigorriko Kantonamendua": "naflap-erdigunea",  # Ahotsak: erdigunekoa-nl
    "Donapaleuko Kantonamendua": "naflap-sortaldea",
    "Donibane Garaziko Kantonamendua": "naflap-sortaldea",

    # Zuberotarra
    "Maule Atharratzeko Kantonamendua": "zuberera",
    "Bearnoko herria (kanpokoa)": "zuberera",

    # Ekialdeko nafarra
    "Zaraitzu":            "ekialde-nafarra",
    "Erronkari":           "ekialde-nafarra",
}
