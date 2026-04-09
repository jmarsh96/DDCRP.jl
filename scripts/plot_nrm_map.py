#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas>=2.0",
#   "geopandas>=0.14",
#   "matplotlib>=3.8",
#   "shapely>=2.0",
#   "requests>=2.28",
# ]
# ///
"""
plot_nrm_map.py
===============
Reads MAP cluster assignments from results/nrm_county_analysis/map_assignments.csv
and plots a choropleth of NRM counties coloured by MAP cluster for each proposal.

Called from nrm_county_analysis.jl via:
    run(`uv run scripts/plot_nrm_map.py`)
"""

import os, sys, io, requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from shapely.ops import unary_union

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
CACHE_DIR   = os.path.join(DATA_DIR, ".cache")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "nrm_county_analysis")
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from prepare_nrm_county_data import COUNTY_LOOKUP, E10_TO_E07

# ============================================================================
# Boundary source URLs
# ============================================================================

GB_LAD_URL = ("https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master"
              "/json/administrative/gb/lad.json")
NI_LGD_URL = ("https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master"
              "/json/administrative/ni/lgd.json")

# ============================================================================
# Code expansion mappings
# ============================================================================

# Metropolitan county (E11) → constituent borough (E08) codes
E11_TO_E08 = {
    "E11000001": ["E08000001","E08000002","E08000003","E08000004","E08000005",
                  "E08000006","E08000007","E08000008","E08000009","E08000010"],
    "E11000002": ["E08000011","E08000012","E08000013","E08000014","E08000015"],
    "E11000003": ["E08000016","E08000017","E08000018","E08000019"],
    "E11000005": ["E08000025","E08000026","E08000027","E08000028",
                  "E08000029","E08000030","E08000031"],
    "E11000006": ["E08000032","E08000033","E08000034","E08000035","E08000036"],
    "E11000007": ["E08000021","E08000022","E08000023","E08000024","E08000037"],
}

# Post-2013 England codes → 2013 predecessor codes
NEW_CODE_TO_OLD = {
    # Dorset 2019
    "E06000058": ["E06000028","E06000029","E07000048"],           # BCP
    "E06000059": ["E07000049","E07000050","E07000051",
                  "E07000052","E07000053"],                        # Dorset UA
    # Buckinghamshire 2020
    "E06000060": ["E07000004","E07000005","E07000006","E07000007"],
    # Northamptonshire 2021
    "E06000061": ["E07000150","E07000152","E07000153","E07000156"],
    "E06000062": ["E07000151","E07000154","E07000155"],
    # Cumbria 2023
    "E06000063": ["E07000026","E07000027","E07000028"],
    "E06000064": ["E07000029","E07000030","E07000031"],
    # North Yorkshire 2023
    "E06000065": ["E07000163","E07000164","E07000165",
                  "E07000166","E07000167","E07000168","E07000169"],
    # Somerset 2023
    "E06000066": ["E07000187","E07000188","E07000189","E07000190"],
    # Renamed district (Somerset West & Taunton → old Taunton Deane)
    "E07000246": ["E07000189"],
    # Post-2019 Scotland council area codes → old 2013 codes
    "S12000047": ["S12000021"],  # North Ayrshire
    "S12000048": ["S12000024"],  # Perth and Kinross
    "S12000049": ["S12000046"],  # Glasgow City
    "S12000050": ["S12000044"],  # North Lanarkshire
}

# ============================================================================
# Download and cache boundaries
# ============================================================================

def download_cached(url: str, filename: str) -> bytes:
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()
    print(f"  Downloading {filename} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(r.content)
    return r.content


def load_boundaries():
    gb_bytes = download_cached(GB_LAD_URL, "martinjc_gb_lad.json")
    ni_bytes = download_cached(NI_LGD_URL, "martinjc_ni_lgd.json")
    gdf_gb = gpd.read_file(io.BytesIO(gb_bytes)).set_crs("EPSG:4326", allow_override=True)
    gdf_ni = gpd.read_file(io.BytesIO(ni_bytes)).set_crs("EPSG:4326", allow_override=True)
    # Uniform code column
    gdf_gb = gdf_gb.rename(columns={"LAD13CD": "code"})
    gdf_ni = gdf_ni.rename(columns={"LGDCode": "code"})
    return gdf_gb[["code","geometry"]], gdf_ni[["code","geometry"]]


# ============================================================================
# Build per-county geometry
# ============================================================================

def expand_codes(raw_codes, imd_codes=None):
    """Expand E10/E11/E12 upper-tier codes, then remap post-2013 codes to 2013 equivalents."""
    step1 = []
    for code in raw_codes:
        if code.startswith("E10"):
            step1.extend(E10_TO_E07.get(code, [code]))
        elif code.startswith("E11"):
            step1.extend(E11_TO_E08.get(code, [code]))
        elif code.startswith("E12"):
            # E12 = English region code; use imd_codes (leaf-level) instead
            if imd_codes:
                step1.extend([c for c in imd_codes if not c.startswith("E10")])
        else:
            step1.append(code)

    step2 = []
    for code in step1:
        if code in NEW_CODE_TO_OLD:
            step2.extend(NEW_CODE_TO_OLD[code])
        else:
            step2.append(code)
    return step2


def build_county_geodataframe(gdf_gb, gdf_ni):
    gb_idx = gdf_gb.set_index("code")["geometry"]
    ni_idx = gdf_ni.set_index("code")["geometry"]

    records = []
    for name, entry in COUNTY_LOOKUP.items():
        nation = entry["nation"]
        raw_codes = entry["pop_codes"]

        if nation == "ni":
            geom_list = [ni_idx[c] for c in raw_codes if c in ni_idx.index]
        else:
            imd_codes = entry.get("imd_codes", [])
            leaf_codes = expand_codes(raw_codes, imd_codes=imd_codes)
            geom_list = [gb_idx[c] for c in leaf_codes if c in gb_idx.index]

        if not geom_list:
            print(f"  Warning: no geometry found for '{name}'")
            continue

        merged = unary_union(geom_list)
        records.append({"county": name, "geometry": merged})

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


# ============================================================================
# Main
# ============================================================================

def main():
    assignments_path = os.path.join(RESULTS_DIR, "map_assignments.csv")
    if not os.path.exists(assignments_path):
        print(f"ERROR: {assignments_path} not found. Run nrm_county_analysis.jl first.")
        sys.exit(1)

    df_asgn = pd.read_csv(assignments_path)
    proposals = df_asgn["proposal"].unique().tolist()
    n_props   = len(proposals)

    print("Loading boundary data ...")
    gdf_gb, gdf_ni = load_boundaries()

    print("Building NRM county polygons ...")
    gdf_counties = build_county_geodataframe(gdf_gb, gdf_ni)

    # Colour palette: tab10 for up to 10 clusters, extend if needed
    tab10 = plt.cm.tab10.colors
    max_k = df_asgn["cluster"].max()
    palette = (list(tab10) * ((max_k // 10) + 1))[:max_k]

    ncols = min(n_props, 3)
    nrows = (n_props + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 6.5 * nrows),
                             squeeze=False)

    for idx, proposal in enumerate(proposals):
        ax = axes[idx // ncols][idx % ncols]
        sub = df_asgn[df_asgn["proposal"] == proposal].set_index("county")

        # Merge cluster labels into GeoDataFrame
        gdf_plot = gdf_counties.copy()
        gdf_plot["cluster"] = gdf_plot["county"].map(sub["cluster"])

        # Background: all counties grey (handles missing)
        gdf_plot.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3)

        # Colour by cluster
        k = int(gdf_plot["cluster"].max(skipna=True)) if gdf_plot["cluster"].notna().any() else 1
        for cl in range(1, k + 1):
            mask = gdf_plot["cluster"] == cl
            if mask.any():
                gdf_plot[mask].plot(ax=ax,
                                    color=palette[cl - 1],
                                    edgecolor="white",
                                    linewidth=0.3)

        # Legend
        handles = [mpatches.Patch(color=palette[cl - 1], label=f"Cluster {cl}")
                   for cl in range(1, k + 1)]
        ax.legend(handles=handles, loc="lower left", fontsize=6, framealpha=0.7)

        ax.set_title(proposal, fontsize=9, fontweight="bold")
        ax.set_axis_off()

    # Hide unused axes
    for idx in range(n_props, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("MAP cluster assignments — NRM county geography", fontsize=12, y=1.01)
    fig.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, "nrm_county_map_geography.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved map to {out_path}")


if __name__ == "__main__":
    main()
