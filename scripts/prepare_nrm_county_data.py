#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas>=2.0",
#   "openpyxl>=3.0",
#   "odfpy>=1.4",
#   "requests>=2.28",
#   "xlrd>=2.0",
#   "numpy>=1.24",
# ]
# ///
"""
prepare_nrm_county_data.py
==========================
Reads NRM Table 12 (county referral counts), downloads population and
deprivation domain data from official UK sources, and outputs
data/nrm_county_data.csv for use in nrm_county_analysis.jl.

Run as:
    uv run scripts/prepare_nrm_county_data.py

Data sources
------------
- NRM referrals: local ODS file (Table_12)
- Population:
    England & Wales: ONS mid-2023 mid-year estimates by LAD (mye23tablesew.xlsx)
    Scotland:        embedded from NRS mid-2023 council area estimates
    Northern Ireland: NISRA mid-2023 LGD estimates
- Deprivation (5 domains: income, employment, education, health, crime):
    England:  IMD 2019 File 10 – lower-tier LAD domain average scores
    Scotland: SIMD 2020v2 – data zone domain ranks, aggregated to council area
    Wales:    WIMD 2019 – LSOA domain scores, aggregated to local authority
    N. Ireland: NIMDM 2017 – LGD-level domain indicators

Geographic mapping
------------------
The NRM uses historic county names. Each is mapped to one or more modern
geographic unit codes. Scotland/NI have a separate hardcoded population lookup
because their national statistics agencies use different portals from ONS.

Normalisation
-------------
Each deprivation domain is min-max normalised to [0, 1] within its nation
(higher = more deprived) before concatenation, to remove the different
calibrations of the four national indices.
"""

import os
import sys
import io
import hashlib
import requests
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
CACHE_DIR   = os.path.join(DATA_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ODS_FILE   = os.path.join(DATA_DIR,
    "national-referral-mechanism-statistics-uk-quarter-3-2025-jul-to-sep-tables.ods")
OUT_FILE   = os.path.join(DATA_DIR, "nrm_county_data.csv")


# ============================================================================
# Data source URLs (all verified 2026-03-31)
# ============================================================================

URLS = {
    "ons_pop": (
        "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
        "populationandmigration/populationestimates/datasets/"
        "estimatesofthepopulationforenglandandwales/"
        "mid20232023localauthorityboundarieseditionofthisdataset/"
        "mye23tablesew.xlsx"
    ),
    "imd2019_file10": (
        "https://assets.publishing.service.gov.uk/media/"
        "5d8b3cfbe5274a08be69aa91/"
        "File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx"
    ),
    "simd2020": (
        "https://www.gov.scot/binaries/content/documents/govscot/publications/"
        "statistics/2020/01/scottish-index-of-multiple-deprivation-2020-data-zone-"
        "look-up-file/documents/scottish-index-of-multiple-deprivation-data-zone-"
        "look-up/scottish-index-of-multiple-deprivation-data-zone-look-up/"
        "govscot%3Adocument/SIMD%2B2020v2%2B-%2Bdatazone%2Blookup%2B-%2Bupdated%2B2025.xlsx"
    ),
    "wimd2019": (
        "https://www.gov.wales/sites/default/files/statistics-and-research/"
        "2022-02/wimd-2019-index-and-domain-scores-by-small-area.ods"
    ),
    "nimdm2017_lgd": (
        "https://www.nisra.gov.uk/files/nisra/publications/"
        "NIMDM17%20Tables%20-%20LGD2014.xls"
    ),
    "nisra_pop": (
        "https://www.nisra.gov.uk/system/files/statistics/"
        "2025-05/MYE23_POP_TOTALS_NI_LGD.xlsx"
    ),
}


# ============================================================================
# NRM county → modern geography lookup
# ============================================================================
# For each NRM historic county name (as it appears in Table_12):
#   "nation"    : "england" | "wales" | "scotland" | "ni"
#   "pop_codes" : ONS/NRS/NISRA geographic unit codes for population
#   "imd_codes" : IMD 2019 LAD codes for deprivation (England)
#                 For most areas pop_codes == imd_codes.
#                 They differ only where LAD boundaries changed after 2019.
#
# Notes on Scotland codes: SIMD 2020v2 uses S12xxxxxx council area codes.
# "Inverness" and "Ross and Cromarty" both map to Highland (S12000017);
# they receive Highland's deprivation values. Population of Highland is
# assigned to both — a noted limitation given the small referral counts (4, 1).
# "Wigtown" maps to Dumfries and Galloway (S12000006).

COUNTY_LOOKUP = {

    # =========================================================
    # ENGLAND
    # =========================================================

    "Greater London": {
        "nation": "england",
        # Use London Region code (aggregates all 33 boroughs) for population.
        # For deprivation: aggregate all 33 London boroughs from File 10.
        "pop_codes": ["E12000007"],          # London Region
        "imd_codes": [
            "E09000001","E09000002","E09000003","E09000004","E09000005",
            "E09000006","E09000007","E09000008","E09000009","E09000010",
            "E09000011","E09000012","E09000013","E09000014","E09000015",
            "E09000016","E09000017","E09000018","E09000019","E09000020",
            "E09000021","E09000022","E09000023","E09000024","E09000025",
            "E09000026","E09000027","E09000028","E09000029","E09000030",
            "E09000031","E09000032","E09000033",
        ],
    },
    "West Midlands": {
        "nation": "england",
        "pop_codes": ["E11000005"],           # West Midlands Met County
        "imd_codes": [
            "E08000025","E08000026","E08000027","E08000028",
            "E08000029","E08000030","E08000031",
        ],
    },
    "Greater Manchester": {
        "nation": "england",
        "pop_codes": ["E11000001"],           # Greater Manchester Met County
        "imd_codes": [
            "E08000001","E08000002","E08000003","E08000004","E08000005",
            "E08000006","E08000007","E08000008","E08000009","E08000010",
        ],
    },
    "West Yorkshire": {
        "nation": "england",
        "pop_codes": ["E11000006"],           # West Yorkshire Met County
        "imd_codes": [
            "E08000032","E08000033","E08000034","E08000035","E08000036",
        ],
    },
    "Merseyside": {
        "nation": "england",
        "pop_codes": ["E11000002"],           # Merseyside Met County
        "imd_codes": [
            "E08000011","E08000012","E08000013","E08000014","E08000015",
        ],
    },
    "Leicestershire": {
        "nation": "england",
        # County council area + Leicester UA (Rutland listed separately in NRM)
        "pop_codes": ["E10000018", "E06000016"],
        "imd_codes": ["E10000018", "E06000016"],
    },
    "Essex": {
        "nation": "england",
        # Essex county + Southend-on-Sea UA + Thurrock UA
        "pop_codes": ["E10000012", "E06000033", "E06000034"],
        "imd_codes": ["E10000012", "E06000033", "E06000034"],
    },
    "Northamptonshire": {
        "nation": "england",
        # Post-2021 unitaries; IMD 2019 uses the 7 predecessor districts
        "pop_codes": ["E06000061", "E06000062"],
        "imd_codes": [
            "E07000150","E07000151","E07000152","E07000153",
            "E07000154","E07000155","E07000156",
        ],
    },
    "Kent": {
        "nation": "england",
        # Kent county council + Medway UA
        "pop_codes": ["E10000016", "E06000035"],
        "imd_codes": ["E10000016", "E06000035"],
    },
    "Lancashire": {
        "nation": "england",
        # Lancashire county + Blackburn with Darwen UA + Blackpool UA
        "pop_codes": ["E10000017", "E06000008", "E06000009"],
        "imd_codes": ["E10000017", "E06000008", "E06000009"],
    },
    "Nottinghamshire": {
        "nation": "england",
        # Nottinghamshire county + Nottingham UA
        "pop_codes": ["E10000024", "E06000018"],
        "imd_codes": ["E10000024", "E06000018"],
    },
    "Berkshire": {
        "nation": "england",
        # 6 unitary authorities (no county council since 1998)
        "pop_codes": [
            "E06000036","E06000037","E06000038",
            "E06000039","E06000040","E06000041",
        ],
        "imd_codes": [
            "E06000036","E06000037","E06000038",
            "E06000039","E06000040","E06000041",
        ],
    },
    "South Yorkshire": {
        "nation": "england",
        "pop_codes": ["E11000003"],           # South Yorkshire Met County
        "imd_codes": [
            "E08000016","E08000017","E08000018","E08000019",
        ],
    },
    "Bristol": {
        "nation": "england",
        "pop_codes": ["E06000023"],
        "imd_codes": ["E06000023"],
    },
    "Hampshire": {
        "nation": "england",
        # Hampshire county + Portsmouth + Southampton + Isle of Wight
        "pop_codes": ["E10000014", "E06000044", "E06000045", "E06000046"],
        "imd_codes": ["E10000014", "E06000044", "E06000045", "E06000046"],
    },
    "Hertfordshire": {
        "nation": "england",
        "pop_codes": ["E10000015"],
        "imd_codes": ["E10000015"],
    },
    "Bedfordshire": {
        "nation": "england",
        # Bedford UA + Central Bedfordshire UA + Luton UA
        "pop_codes": ["E06000055", "E06000056", "E06000032"],
        "imd_codes": ["E06000055", "E06000056", "E06000032"],
    },
    "East Sussex": {
        "nation": "england",
        # East Sussex county + Brighton and Hove UA
        "pop_codes": ["E10000011", "E06000043"],
        "imd_codes": ["E10000011", "E06000043"],
    },
    "North Yorkshire": {
        "nation": "england",
        # Post-April 2023: North Yorkshire UA + York UA.
        # IMD 2019 uses the 7 predecessor district codes.
        "pop_codes": ["E06000065", "E06000014"],
        "imd_codes": [
            "E07000163","E07000164","E07000165","E07000166",
            "E07000167","E07000168","E07000169",
            "E06000014",
        ],
    },
    "Tyne & Wear": {
        "nation": "england",
        "pop_codes": ["E11000007"],           # Tyne and Wear Met County
        "imd_codes": [
            "E08000037","E08000021","E08000022","E08000023","E08000024",
            # E08000037 = Gateshead (new code in IMD 2019; old code was E08000020)
        ],
    },
    "Cambridgeshire": {
        "nation": "england",
        # Cambridgeshire county + Peterborough UA
        "pop_codes": ["E10000003", "E06000031"],
        "imd_codes": ["E10000003", "E06000031"],
    },
    "Lincolnshire": {
        "nation": "england",
        # Lincolnshire county + North East Lincolnshire UA + North Lincolnshire UA
        "pop_codes": ["E10000019", "E06000012", "E06000013"],
        "imd_codes": ["E10000019", "E06000012", "E06000013"],
    },
    "Buckinghamshire": {
        "nation": "england",
        # Buckinghamshire UA (from April 2020) + Milton Keynes UA.
        # IMD 2019 uses the 4 predecessor Buckinghamshire districts.
        "pop_codes": ["E06000060", "E06000042"],
        "imd_codes": [
            "E07000004","E07000005","E07000006","E07000007",
            "E06000042",
        ],
    },
    "Durham": {
        "nation": "england",
        "pop_codes": ["E06000047"],
        "imd_codes": ["E06000047"],
    },
    "Shropshire": {
        "nation": "england",
        # Shropshire UA + Telford and Wrekin UA
        "pop_codes": ["E06000051", "E06000020"],
        "imd_codes": ["E06000051", "E06000020"],
    },
    "Staffordshire": {
        "nation": "england",
        # Staffordshire county + Stoke-on-Trent UA
        "pop_codes": ["E10000028", "E06000021"],
        "imd_codes": ["E10000028", "E06000021"],
    },
    "West Sussex": {
        "nation": "england",
        "pop_codes": ["E10000032"],
        "imd_codes": ["E10000032"],
    },
    "Dorset": {
        "nation": "england",
        # Dorset UA + Bournemouth, Christchurch and Poole UA (both from 2019)
        "pop_codes": ["E06000059", "E06000058"],
        "imd_codes": ["E06000059", "E06000058"],
    },
    "Cheshire": {
        "nation": "england",
        # Cheshire East + Cheshire West and Chester + Halton + Warrington
        "pop_codes": ["E06000049", "E06000050", "E06000006", "E06000007"],
        "imd_codes": ["E06000049", "E06000050", "E06000006", "E06000007"],
    },
    "Devon": {
        "nation": "england",
        # Devon county + Plymouth UA + Torbay UA
        "pop_codes": ["E10000008", "E06000026", "E06000027"],
        "imd_codes": ["E10000008", "E06000026", "E06000027"],
    },
    "Oxfordshire": {
        "nation": "england",
        "pop_codes": ["E10000025"],
        "imd_codes": ["E10000025"],
    },
    "Warwickshire": {
        "nation": "england",
        "pop_codes": ["E10000031"],
        "imd_codes": ["E10000031"],
    },
    "Norfolk": {
        "nation": "england",
        "pop_codes": ["E10000020"],
        "imd_codes": ["E10000020"],
    },
    "Cornwall": {
        "nation": "england",
        # Cornwall UA + Isles of Scilly Council
        "pop_codes": ["E06000052", "E06000053"],
        "imd_codes": ["E06000052", "E06000053"],
    },
    "Gloucestershire": {
        "nation": "england",
        "pop_codes": ["E10000013"],
        "imd_codes": ["E10000013"],
    },
    "Somerset": {
        "nation": "england",
        # Somerset UA (from April 2023) + Bath & North East Somerset UA
        # + North Somerset UA.
        # IMD 2019 uses the predecessor Somerset districts plus the existing
        # BANES and North Somerset UAs.
        "pop_codes": ["E06000066", "E06000022", "E06000025"],
        "imd_codes": [
            "E07000187","E07000188","E07000246","E07000189",
            "E06000022","E06000025",
        ],
    },
    "Derbyshire": {
        "nation": "england",
        # Derbyshire county + Derby UA
        "pop_codes": ["E10000007", "E06000015"],
        "imd_codes": ["E10000007", "E06000015"],
    },
    "East Riding of Yorkshire": {
        "nation": "england",
        # East Riding UA + Kingston upon Hull UA
        "pop_codes": ["E06000011", "E06000010"],
        "imd_codes": ["E06000011", "E06000010"],
    },
    "Suffolk": {
        "nation": "england",
        "pop_codes": ["E10000029"],
        "imd_codes": ["E10000029"],
    },
    "Wiltshire": {
        "nation": "england",
        # Wiltshire UA + Swindon UA
        "pop_codes": ["E06000054", "E06000030"],
        "imd_codes": ["E06000054", "E06000030"],
    },
    "Surrey": {
        "nation": "england",
        "pop_codes": ["E10000030"],
        "imd_codes": ["E10000030"],
    },
    "Northumberland": {
        "nation": "england",
        "pop_codes": ["E06000057"],
        "imd_codes": ["E06000057"],
    },
    "Cumbria": {
        "nation": "england",
        # Post-April 2023: Cumberland UA + Westmorland and Furness UA.
        # IMD 2019 uses the 6 predecessor Cumbria districts.
        "pop_codes": ["E06000063", "E06000064"],
        "imd_codes": [
            "E07000026","E07000028","E07000029",
            "E07000030","E07000031","E07000027",
        ],
    },
    "Worcestershire": {
        "nation": "england",
        "pop_codes": ["E10000034"],
        "imd_codes": ["E10000034"],
    },
    "Herefordshire": {
        "nation": "england",
        "pop_codes": ["E06000019"],
        "imd_codes": ["E06000019"],
    },
    "Rutland": {
        "nation": "england",
        "pop_codes": ["E06000017"],
        "imd_codes": ["E06000017"],
    },

    # =========================================================
    # WALES  (pre-1996 historic counties → modern UAs)
    # =========================================================
    # WIMD 2019 identifies areas by "Local Authority name".
    # Deprivation is aggregated from LSOA level to LA, then to NRM county.
    # Population from the same ONS file as England.

    "South Glamorgan": {
        "nation": "wales",
        "pop_codes": ["W06000015", "W06000014"],   # Cardiff + Vale of Glamorgan
        "imd_codes": ["Cardiff", "Vale of Glamorgan"],
    },
    "Clwyd": {
        "nation": "wales",
        # Denbighshire + Flintshire + Wrexham + Conwy (eastern part; all Conwy assigned here)
        "pop_codes": ["W06000004","W06000005","W06000006","W06000003"],
        "imd_codes": ["Denbighshire","Flintshire","Wrexham","Conwy"],
    },
    "Mid Glamorgan": {
        "nation": "wales",
        # Bridgend + Rhondda Cynon Taf + Merthyr Tydfil + Caerphilly
        "pop_codes": ["W06000013","W06000016","W06000024","W06000018"],
        "imd_codes": [
            "Bridgend","Rhondda Cynon Taf",
            "Merthyr Tydfil","Caerphilly",
        ],
    },
    "Gwent": {
        "nation": "wales",
        # Blaenau Gwent + Torfaen + Monmouthshire + Newport
        "pop_codes": ["W06000019","W06000020","W06000021","W06000022"],
        "imd_codes": [
            "Blaenau Gwent","Torfaen","Monmouthshire","Newport",
        ],
    },
    "West Glamorgan": {
        "nation": "wales",
        # Swansea + Neath Port Talbot
        "pop_codes": ["W06000011","W06000012"],
        "imd_codes": ["Swansea","Neath Port Talbot"],
    },
    "Dyfed": {
        "nation": "wales",
        # Ceredigion + Pembrokeshire + Carmarthenshire
        "pop_codes": ["W06000008","W06000009","W06000010"],
        "imd_codes": ["Ceredigion","Pembrokeshire","Carmarthenshire"],
    },
    "Gwynedd": {
        "nation": "wales",
        # Gwynedd UA + Isle of Anglesey UA (Conwy assigned to Clwyd above)
        "pop_codes": ["W06000002","W06000001"],
        "imd_codes": ["Gwynedd","Isle of Anglesey"],
    },
    "Powys": {
        "nation": "wales",
        "pop_codes": ["W06000023"],
        "imd_codes": ["Powys"],
    },

    # =========================================================
    # SCOTLAND  (SIMD council area codes + NRS 2023 pop embedded below)
    # =========================================================

    "City of Glasgow": {
        "nation": "scotland",
        "pop_codes": ["S12000049"],       # Glasgow City (post-2019 code)
    },
    "City of Edinburgh": {
        "nation": "scotland",
        "pop_codes": ["S12000036"],
    },
    "City of Aberdeen": {
        "nation": "scotland",
        "pop_codes": ["S12000033"],       # Aberdeen City
    },
    "City of Dundee": {
        "nation": "scotland",
        "pop_codes": ["S12000042"],       # Dundee City
    },
    "Lanarkshire": {
        "nation": "scotland",
        # North Lanarkshire (post-2019 code) + South Lanarkshire
        "pop_codes": ["S12000050","S12000029"],
    },
    "Renfrewshire": {
        "nation": "scotland",
        # Renfrewshire + East Renfrewshire + Inverclyde
        "pop_codes": ["S12000038","S12000011","S12000018"],
    },
    "Dunbartonshire": {
        "nation": "scotland",
        # West Dunbartonshire + East Dunbartonshire
        "pop_codes": ["S12000039","S12000045"],
    },
    "Fife": {
        "nation": "scotland",
        "pop_codes": ["S12000015"],
    },
    "Stirling and Falkirk": {
        "nation": "scotland",
        "pop_codes": ["S12000030","S12000014"],   # Stirling + Falkirk
    },
    "West Lothian": {
        "nation": "scotland",
        "pop_codes": ["S12000040"],
    },
    "Angus": {
        "nation": "scotland",
        "pop_codes": ["S12000041"],
    },
    "Moray": {
        "nation": "scotland",
        "pop_codes": ["S12000020"],
    },
    "Aberdeenshire": {
        "nation": "scotland",
        "pop_codes": ["S12000034"],
    },
    "Inverness": {
        "nation": "scotland",
        # Inverness city is in Highland council area
        "pop_codes": ["S12000017"],       # Highland
    },
    "Perth and Kinross": {
        "nation": "scotland",
        "pop_codes": ["S12000048"],       # Perth and Kinross (post-2019 code)
    },
    "Ross and Cromarty": {
        "nation": "scotland",
        # Also in Highland council area — same codes as Inverness above.
        # Both NRM counties get Highland's deprivation profile; population
        # of Highland is assigned to each (conservative over-estimate).
        "pop_codes": ["S12000017"],       # Highland
    },
    "Clackmannan": {
        "nation": "scotland",
        "pop_codes": ["S12000005"],       # Clackmannanshire
    },
    "Wigtown": {
        "nation": "scotland",
        # Historic Wigtown county is now part of Dumfries and Galloway
        "pop_codes": ["S12000006"],       # Dumfries and Galloway
    },

    # =========================================================
    # NORTHERN IRELAND
    # =========================================================

    "County Antrim": {
        "nation": "ni",
        # Antrim and Newtownabbey + Mid and East Antrim + Causeway Coast and Glens
        # (Belfast N09000003 straddles County Antrim/Down; excluded here)
        "pop_codes": ["N09000001","N09000008","N09000004"],
    },
    "County Derry / Londonderry": {
        "nation": "ni",
        "pop_codes": ["N09000005"],       # Derry City and Strabane
    },
}


# ============================================================================
# NRS mid-2023 Scotland population by council area code
# Source: National Records of Scotland, Mid-2023 Population Estimates
# https://www.nrscotland.gov.uk
# ============================================================================

SCOTLAND_POP_2023 = {
    "S12000005":  52030,   # Clackmannanshire
    "S12000006": 151030,   # Dumfries and Galloway
    "S12000008": 122440,   # East Ayrshire
    "S12000010":  107680,  # East Lothian
    "S12000011":  96400,   # East Renfrewshire
    "S12000013":  26610,   # Na h-Eileanan Siar
    "S12000014": 163080,   # Falkirk
    "S12000015": 376500,   # Fife
    "S12000017": 237970,   # Highland
    "S12000018":  78330,   # Inverclyde
    "S12000019":  91470,   # Midlothian
    "S12000020":  96810,   # Moray
    "S12000021": 138640,   # North Ayrshire (old code; may appear as S12000047)
    "S12000047": 138640,   # North Ayrshire (post-2019 code)
    "S12000023":  22290,   # Orkney Islands
    "S12000024": 158010,   # Perth and Kinross (old code)
    "S12000048": 158010,   # Perth and Kinross (post-2019 code)
    "S12000026": 114720,   # Scottish Borders
    "S12000027":  23040,   # Shetland Islands
    "S12000028":  112580,  # South Ayrshire
    "S12000029": 320050,   # South Lanarkshire
    "S12000030":  99580,   # Stirling
    "S12000033": 228440,   # Aberdeen City
    "S12000034": 264360,   # Aberdeenshire
    "S12000035":  84560,   # Argyll and Bute
    "S12000036": 524930,   # City of Edinburgh
    "S12000038": 178650,   # Renfrewshire
    "S12000039":  91330,   # West Dunbartonshire
    "S12000040": 179940,   # West Lothian
    "S12000041": 116630,   # Angus
    "S12000042": 150560,   # Dundee City
    "S12000044": 344270,   # North Lanarkshire (old code)
    "S12000050": 344270,   # North Lanarkshire (post-2019 code)
    "S12000045": 108650,   # East Dunbartonshire
    "S12000046": 620000,   # Glasgow City (old code)
    "S12000049": 620000,   # Glasgow City (post-2019 code)
}


# ============================================================================
# Helper: download with caching
# ============================================================================

def download_cached(url: str, filename: str) -> bytes:
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path):
        print(f"  [cache] {filename}")
        with open(cache_path, "rb") as f:
            return f.read()
    print(f"  [download] {filename} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(r.content)
    return r.content


# ============================================================================
# Step 1: Load NRM Table_12
# ============================================================================

def load_nrm_data() -> pd.DataFrame:
    print("Loading NRM Table_12 ...")
    df = pd.read_excel(ODS_FILE, sheet_name="Table_12", engine="odf", header=None)
    # Rows 5 onwards: row 5 is header ("UK County", "Total"), rows 6-79 are data
    data = df.iloc[6:, :2].copy()
    data.columns = ["county", "referrals"]
    data = data.dropna(subset=["county","referrals"])
    data["county"] = data["county"].astype(str).str.strip()
    data["referrals"] = data["referrals"].astype(int)
    print(f"  {len(data)} counties loaded.")
    return data.reset_index(drop=True)


# ============================================================================
# Step 2: England + Wales population (ONS mid-2023)
# ============================================================================

def load_ons_population() -> dict:
    """Return dict {ONS_code: population} for England and Wales."""
    content = download_cached(URLS["ons_pop"], "mye23tablesew.xlsx")
    xl = pd.ExcelFile(io.BytesIO(content))
    df = pd.read_excel(xl, sheet_name="MYE2 - Persons", header=7)
    df = df.rename(columns={"Code": "code", "All ages": "population"})
    df = df[["code","population"]].dropna(subset=["code","population"])
    return dict(zip(df["code"].astype(str), df["population"].astype(int)))


# ============================================================================
# Step 3: England IMD 2019 deprivation domain scores (File 10)
# ============================================================================

def load_england_imd() -> pd.DataFrame:
    """
    Return DataFrame with columns:
        lad_code, income, employment, education, health, crime
    One row per 2019 LAD.
    """
    content = download_cached(URLS["imd2019_file10"], "imd2019_file10.xlsx")
    xl = pd.ExcelFile(io.BytesIO(content))

    domain_map = {
        "income":     ("Income",     "Income - Average score "),
        "employment": ("Employment", "Employment - Average score "),
        "education":  ("Education",  "Education, Skills and Training - Average score "),
        "health":     ("Health",     "Health Deprivation and Disability - Average score "),
        "crime":      ("Crime",      "Crime - Average score "),
    }

    base = None
    for domain, (sheet, score_col) in domain_map.items():
        df = pd.read_excel(xl, sheet_name=sheet)
        df = df.rename(columns={
            "Local Authority District code (2019)": "lad_code",
            score_col: domain,
        })
        df = df[["lad_code", domain]].dropna()
        if base is None:
            base = df
        else:
            base = base.merge(df, on="lad_code", how="outer")

    # Some domain sheets may use slightly different column names
    if base is None or base.empty:
        raise RuntimeError("Failed to parse IMD 2019 File 10.")

    return base.reset_index(drop=True)


# ============================================================================
# Step 4: Scotland SIMD 2020 deprivation domain ranks → scores
# ============================================================================

def load_scotland_simd() -> pd.DataFrame:
    """
    Aggregate SIMD 2020v2 data zone domain ranks to council area level.
    Returns DataFrame with columns:
        la_code, income, employment, education, health, crime
    Domain values are population-weighted mean of (max_rank + 1 - rank),
    i.e. higher = more deprived.
    """
    content = download_cached(URLS["simd2020"], "simd2020v2.xlsx")
    xl = pd.ExcelFile(io.BytesIO(content))
    df = pd.read_excel(xl, sheet_name="SIMD 2020v2 DZ lookup data")

    # Standardise column names to lowercase for robustness
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify domain rank columns (may vary slightly between 2020 and 2020v2)
    rank_cols = {
        "income":     next(c for c in df.columns if "income_domain" in c and "rank" in c),
        "employment": next(c for c in df.columns if "employment_domain" in c and "rank" in c),
        "education":  next(c for c in df.columns if "education_domain" in c and "rank" in c),
        "health":     next(c for c in df.columns if "health_domain" in c and "rank" in c),
        "crime":      next(c for c in df.columns if "crime_domain" in c and "rank" in c),
    }

    la_col  = next(c for c in df.columns if "lacode" in c.replace("_","").replace(" ",""))
    pop_col = next(c for c in df.columns if c == "population" or c == "total_population")

    df = df[[la_col, pop_col] + list(rank_cols.values())].dropna()
    df = df.rename(columns={la_col: "la_code", pop_col: "pop"})

    # Invert ranks: rank 1 = most deprived; max_rank = least deprived.
    # Inverted: most deprived → highest value.
    for domain, rc in rank_cols.items():
        max_r = df[rc].max()
        df[domain] = max_r + 1 - df[rc]

    df["pop"] = pd.to_numeric(df["pop"], errors="coerce").fillna(0)

    # Population-weighted average within each council area
    records = []
    for la_code, grp in df.groupby("la_code"):
        w = grp["pop"].values
        row = {"la_code": la_code}
        for domain in rank_cols:
            row[domain] = np.average(grp[domain].values, weights=w) if w.sum() > 0 else np.nan
        records.append(row)

    return pd.DataFrame(records)


# ============================================================================
# Step 5: Wales WIMD 2019 deprivation domain scores
# ============================================================================

def load_wales_wimd() -> pd.DataFrame:
    """
    Aggregate WIMD 2019 LSOA-level scores to local authority level.
    Returns DataFrame with columns:
        la_name, income, employment, education, health, crime
    (la_name matches the WIMD 'Local Authority name' strings)
    """
    content = download_cached(URLS["wimd2019"], "wimd2019.ods")
    xl = pd.ExcelFile(io.BytesIO(content), engine="odf")
    df = pd.read_excel(xl, sheet_name="Data", header=3)

    # Column names from the ODS: LSOA code, LSOA name, Local Authority name,
    # WIMD 2019, Income, Employment, Health, Education,
    # Access to Services, Housing, Community Safety, Physical Environment
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Local Authority name": "la_name",
        "Income": "income",
        "Employment": "employment",
        "Education": "education",
        "Health": "health",
        "Community Safety": "crime",
    })

    keep = ["la_name","income","employment","education","health","crime"]
    df = df[keep].dropna()

    # Simple mean across LSOAs within each LA (each LSOA ≈ 1,500 residents)
    return df.groupby("la_name")[
        ["income","employment","education","health","crime"]
    ].mean().reset_index()


# ============================================================================
# Step 6: Northern Ireland NIMDM 2017 domain indicators
# ============================================================================

def load_ni_nimdm() -> pd.DataFrame:
    """
    Load NIMDM 2017 LGD-level domain indicator data.
    Returns DataFrame with columns:
        lgd_code, income, employment, education, health, crime
    Uses the first/primary indicator for each domain.
    """
    content = download_cached(URLS["nimdm2017_lgd"], "nimdm2017_lgd.xls")
    xl = pd.ExcelFile(io.BytesIO(content), engine="xlrd")

    def read_domain(sheet: str, col: int) -> pd.Series:
        df = pd.read_excel(xl, sheet_name=sheet, header=0)
        df = df.iloc[:11]   # 11 LGDs; row 12 is NI total
        codes = df.iloc[:, 0].astype(str)
        vals  = pd.to_numeric(df.iloc[:, col], errors="coerce")
        return pd.Series(vals.values, index=codes.values)

    income     = read_domain("Income and Employment", 2)   # income deprivation %
    employment = read_domain("Income and Employment", 5)   # employment deprivation %
    # Health: standardised preventable death ratio (col 2)
    health     = read_domain("Health and Disability", 2)
    # Education: proportion in special schools/SEN (col 2)
    education  = read_domain("Education, Skills and Training ", 2)
    # Crime: rate of violence/robbery/public order per 1,000 (col 2)
    crime      = read_domain("Crime and Disorder", 2)

    df_out = pd.DataFrame({
        "lgd_code":   income.index,
        "income":     income.values,
        "employment": employment.values,
        "education":  education.values,
        "health":     health.values,
        "crime":      crime.values,
    })
    return df_out.dropna()


# ============================================================================
# Step 7: Northern Ireland population (NISRA mid-2023)
# ============================================================================

def load_ni_population() -> dict:
    """Return dict {N09xxxxxx: population} for NI LGDs, mid-2023."""
    content = download_cached(URLS["nisra_pop"], "nisra_pop2023.xlsx")
    xl = pd.ExcelFile(io.BytesIO(content))
    df = pd.read_excel(xl, sheet_name="Flat")
    df = df[
        (df["area"].astype(str).str.startswith("2.")) &  # LGD rows
        (df["year"] == 2023) &
        (df["type"] == "Rounded")
    ]
    return dict(zip(df["area_code"].astype(str), df["MYE"].astype(int)))


# ============================================================================
# Step 8: Normalise domain scores within each nation to [0, 1]
# ============================================================================

def minmax_normalise(df: pd.DataFrame, domain_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in domain_cols:
        lo, hi = df[col].min(), df[col].max()
        if hi > lo:
            df[col] = (df[col] - lo) / (hi - lo)
        else:
            df[col] = 0.0
    return df


# ============================================================================
# Geographic code mappings
# ============================================================================

# IMD 2019 File 10 only contains lower-tier LAD codes (E06, E07, E08, E09).
# County council (E10) codes are upper-tier and must be expanded to their
# constituent district (E07) codes.  Mapping extracted from ONS MYE 2023
# population file hierarchical structure.
E10_TO_E07 = {
    "E10000003": ["E07000008","E07000009","E07000010","E07000011","E07000012"],
    "E10000007": ["E07000032","E07000033","E07000034","E07000035","E07000036","E07000037","E07000038","E07000039"],
    "E10000008": ["E07000040","E07000041","E07000042","E07000043","E07000044","E07000045","E07000046","E07000047"],
    "E10000011": ["E07000061","E07000062","E07000063","E07000064","E07000065"],
    "E10000012": ["E07000066","E07000067","E07000068","E07000069","E07000070","E07000071","E07000072","E07000073","E07000074","E07000075","E07000076","E07000077"],
    "E10000013": ["E07000078","E07000079","E07000080","E07000081","E07000082","E07000083"],
    "E10000014": ["E07000084","E07000085","E07000086","E07000087","E07000088","E07000089","E07000090","E07000091","E07000092","E07000093","E07000094"],
    "E10000015": ["E07000095","E07000096","E07000242","E07000098","E07000099","E07000240","E07000243","E07000102","E07000103","E07000241"],
    "E10000016": ["E07000105","E07000106","E07000107","E07000108","E07000112","E07000109","E07000110","E07000111","E07000113","E07000114","E07000115","E07000116"],
    "E10000017": ["E07000117","E07000118","E07000119","E07000120","E07000121","E07000122","E07000123","E07000124","E07000125","E07000126","E07000127","E07000128"],
    "E10000018": ["E07000129","E07000130","E07000131","E07000132","E07000133","E07000134","E07000135"],
    "E10000019": ["E07000136","E07000137","E07000138","E07000139","E07000140","E07000141","E07000142"],
    "E10000020": ["E07000143","E07000144","E07000145","E07000146","E07000147","E07000148","E07000149"],
    "E10000024": ["E07000170","E07000171","E07000172","E07000173","E07000174","E07000175","E07000176"],
    "E10000025": ["E07000177","E07000178","E07000179","E07000180","E07000181"],
    "E10000028": ["E07000192","E07000193","E07000194","E07000195","E07000196","E07000197","E07000198","E07000199"],
    "E10000029": ["E07000200","E07000244","E07000202","E07000203","E07000245"],
    "E10000030": ["E07000207","E07000208","E07000209","E07000210","E07000211","E07000212","E07000213","E07000214","E07000215","E07000216","E07000217"],
    "E10000031": ["E07000218","E07000219","E07000220","E07000221","E07000222"],
    "E10000032": ["E07000223","E07000224","E07000225","E07000226","E07000227","E07000228","E07000229"],
    "E10000034": ["E07000234","E07000235","E07000236","E07000237","E07000238","E07000239"],
}

# SIMD 2020v2 "updated 2025" uses S12000047 for Fife (instead of the
# standard NRS code S12000015).  Population lookup still uses S12000015
# from SCOTLAND_POP_2023.
SIMD_CODE_REMAP = {
    "S12000015": "S12000047",
}


# ============================================================================
# Main aggregation
# ============================================================================

DOMAINS = ["income","employment","education","health","crime"]


def aggregate_england(nrm_counties, ons_pop, imd_df):
    records = []
    imd_by_code = imd_df.set_index("lad_code").to_dict("index")

    for county, info in nrm_counties.items():
        if info["nation"] != "england":
            continue

        # --- population ---
        pop = 0
        missing_pop = []
        for code in info["pop_codes"]:
            if code in ons_pop:
                pop += ons_pop[code]
            else:
                missing_pop.append(code)
        if missing_pop:
            print(f"  WARNING: pop code(s) not found for {county}: {missing_pop}")

        # --- deprivation (unweighted average across constituent LADs) ---
        # Expand any E10 county council codes to constituent E07 district codes
        imd_codes = []
        for code in info["imd_codes"]:
            if code in E10_TO_E07:
                imd_codes.extend(E10_TO_E07[code])
            else:
                imd_codes.append(code)

        domain_vals = {d: [] for d in DOMAINS}
        missing_imd = []
        for code in imd_codes:
            if code in imd_by_code:
                row = imd_by_code[code]
                for d in DOMAINS:
                    if d in row and not np.isnan(row[d]):
                        domain_vals[d].append(row[d])
            else:
                missing_imd.append(code)
        if missing_imd:
            print(f"  WARNING: IMD code(s) not found for {county}: {missing_imd}")

        dep = {d: np.mean(v) if v else np.nan for d, v in domain_vals.items()}
        records.append({"county": county, "population": pop, **dep})

    return pd.DataFrame(records)


def aggregate_wales(nrm_counties, ons_pop, wimd_df):
    records = []
    wimd_by_name = wimd_df.set_index("la_name").to_dict("index")

    for county, info in nrm_counties.items():
        if info["nation"] != "wales":
            continue

        pop = sum(ons_pop.get(c, 0) for c in info["pop_codes"])

        domain_vals = {d: [] for d in DOMAINS}
        for la_name in info["imd_codes"]:
            if la_name in wimd_by_name:
                row = wimd_by_name[la_name]
                for d in DOMAINS:
                    v = row.get(d, np.nan)
                    if not np.isnan(v):
                        domain_vals[d].append(v)
            else:
                print(f"  WARNING: WIMD LA name not found for {county}: '{la_name}'")
                # Try case-insensitive match
                matches = [k for k in wimd_by_name if k.strip().lower() == la_name.lower()]
                if matches:
                    row = wimd_by_name[matches[0]]
                    for d in DOMAINS:
                        v = row.get(d, np.nan)
                        if not np.isnan(v):
                            domain_vals[d].append(v)

        dep = {d: np.mean(v) if v else np.nan for d, v in domain_vals.items()}
        records.append({"county": county, "population": pop, **dep})

    return pd.DataFrame(records)


def aggregate_scotland(nrm_counties, simd_df):
    records = []
    simd_by_code = simd_df.set_index("la_code").to_dict("index")

    for county, info in nrm_counties.items():
        if info["nation"] != "scotland":
            continue

        pop = sum(SCOTLAND_POP_2023.get(c, 0) for c in info["pop_codes"])
        if pop == 0:
            missing = [c for c in info["pop_codes"] if c not in SCOTLAND_POP_2023]
            print(f"  WARNING: Scottish pop code(s) not found for {county}: {missing}")

        domain_vals = {d: [] for d in DOMAINS}
        for code in info["pop_codes"]:
            simd_code = SIMD_CODE_REMAP.get(code, code)
            if simd_code in simd_by_code:
                row = simd_by_code[simd_code]
                for d in DOMAINS:
                    v = row.get(d, np.nan)
                    if not np.isnan(v):
                        domain_vals[d].append(v)
            else:
                print(f"  WARNING: SIMD code not found for {county}: '{code}' (remapped: '{simd_code}')")

        dep = {d: np.mean(v) if v else np.nan for d, v in domain_vals.items()}
        records.append({"county": county, "population": pop, **dep})

    return pd.DataFrame(records)


def aggregate_ni(nrm_counties, ni_pop, nimdm_df):
    records = []
    nimdm_by_code = nimdm_df.set_index("lgd_code").to_dict("index")

    for county, info in nrm_counties.items():
        if info["nation"] != "ni":
            continue

        pop = sum(ni_pop.get(c, 0) for c in info["pop_codes"])
        if pop == 0:
            print(f"  WARNING: NI pop not found for {county}: {info['pop_codes']}")

        domain_vals = {d: [] for d in DOMAINS}
        for code in info["pop_codes"]:
            if code in nimdm_by_code:
                row = nimdm_by_code[code]
                for d in DOMAINS:
                    v = row.get(d, np.nan)
                    if not np.isnan(v):
                        domain_vals[d].append(v)
            else:
                print(f"  WARNING: NIMDM code not found for {county}: '{code}'")

        dep = {d: np.mean(v) if v else np.nan for d, v in domain_vals.items()}
        records.append({"county": county, "population": pop, **dep})

    return pd.DataFrame(records)


# ============================================================================
# Entry point
# ============================================================================

def main():
    print("=" * 60)
    print("NRM County Data Preparation")
    print("=" * 60)

    # 1. NRM referrals
    nrm = load_nrm_data()

    # 2. Verify all NRM county names are in the lookup
    unknown = [c for c in nrm["county"] if c not in COUNTY_LOOKUP]
    if unknown:
        print(f"WARNING: {len(unknown)} NRM counties not in lookup: {unknown}")

    # 3. Load all data sources
    print("\nLoading population data ...")
    ons_pop = load_ons_population()
    ni_pop  = load_ni_population()

    print("\nLoading deprivation data ...")
    imd_df   = load_england_imd()
    simd_df  = load_scotland_simd()
    wimd_df  = load_wales_wimd()
    nimdm_df = load_ni_nimdm()

    # 4. Aggregate to NRM county level
    print("\nAggregating to county level ...")
    df_eng  = aggregate_england(COUNTY_LOOKUP, ons_pop, imd_df)
    df_wal  = aggregate_wales(COUNTY_LOOKUP, ons_pop, wimd_df)
    df_sco  = aggregate_scotland(COUNTY_LOOKUP, simd_df)
    df_ni   = aggregate_ni(COUNTY_LOOKUP, ni_pop, nimdm_df)

    dep_all = pd.concat([df_eng, df_wal, df_sco, df_ni], ignore_index=True)

    # 5. Normalise domain scores within each nation to [0, 1]
    print("\nNormalising domain scores within each nation ...")
    for county, info in COUNTY_LOOKUP.items():
        dep_all.loc[dep_all["county"] == county, "nation"] = info["nation"]

    normed_parts = []
    for nation, grp in dep_all.groupby("nation"):
        normed_parts.append(minmax_normalise(grp, DOMAINS))
    dep_norm = pd.concat(normed_parts, ignore_index=True)
    dep_norm = dep_norm.drop(columns=["nation"])

    # 6. Merge with NRM referral counts
    result = nrm.merge(dep_norm, on="county", how="left")

    # 7. Check for missing values
    n_missing = result[DOMAINS + ["population"]].isna().any(axis=1).sum()
    if n_missing:
        miss = result[result[DOMAINS + ["population"]].isna().any(axis=1)]["county"].tolist()
        print(f"\nWARNING: {n_missing} counties have missing data: {miss}")

    # 8. Report summary
    print(f"\nResult shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(f"\nFirst 5 rows:")
    print(result.head().to_string())
    print(f"\nPopulation range: {result['population'].min():,.0f} – {result['population'].max():,.0f}")
    print(f"Referral range:   {result['referrals'].min()} – {result['referrals'].max()}")

    # 9. Save
    result.to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
