# ============================================================================
# Walk Free – proposal comparison analysis
# ============================================================================
#
# Loads (thinned) chains from results/walkfree/chains_thinned/ and produces:
#   • Per-config trace plots: K, r, α, logpost
#   • Cross-config comparison plots: K distribution, ESS(K), acceptance rates,
#     mean K, mean r, run time
#   • Table of posterior number-of-clusters statistics (CSV + console)
#
# Run after walkfree_thin_chains.jl (or directly against the full chains):
#   julia --project scripts/walkfree_proposal_analysis.jl
#
# CLI options:
#   --src DIR      chain directory to read (default: results/walkfree/chains_thinned)
#   --out DIR      figure/table output root  (default: results/walkfree/proposal_analysis)
#   --burnin-frac  fraction of each chain to discard as burn-in (default: 0.25)
# ============================================================================

ENV["GKSwstype"] = "100"   # headless GR backend

using DDCRP
using JLD2
using CSV, DataFrames
using Statistics, StatsBase
using Plots, StatsPlots
using Printf

# ── CLI args ─────────────────────────────────────────────────────────────────
function get_arg(flag, default)
    idx = findfirst(==(flag), ARGS)
    isnothing(idx) ? default : ARGS[idx + 1]
end
const SRC_DIR     = get_arg("--src",    "results/walkfree/chains_thinned")
const OUT_DIR     = get_arg("--out",    "results/walkfree/proposal_analysis")
const BURNIN_FRAC = parse(Float64, get_arg("--burnin-frac", "0.25"))

mkpath(joinpath(OUT_DIR, "traces"))
mkpath(joinpath(OUT_DIR, "comparison"))
mkpath(joinpath(OUT_DIR, "tables"))

# ── colour palette ────────────────────────────────────────────────────────────
const PALETTE = palette(:tab10)
color_for(i) = PALETTE[mod1(i, length(PALETTE))]

# ============================================================================
# Load all chains
# ============================================================================

chain_files = sort(filter(f -> endswith(f, ".jld2"), readdir(SRC_DIR; join=true)))
isempty(chain_files) && error("No .jld2 files found in $SRC_DIR")

println("Loading $(length(chain_files)) chain(s) from $SRC_DIR …")

struct ChainInfo
    name     :: String
    samples  :: Any           # NBPopulationRatesSamples or …MargSamples
    diag     :: Any
    cfg      :: Any           # NamedTuple or nothing
    n_iter   :: Int
    idx      :: UnitRange{Int}  # post-burnin index
    nc       :: Vector{Int}     # K per post-burnin iteration
end

chains = ChainInfo[]
for fpath in chain_files
    data    = load(fpath)
    samples = data["samples"]
    diag    = data["diag"]
    cfg     = get(data, "cfg", nothing)

    n_iter  = size(samples.c, 1)
    burnin  = max(1, round(Int, BURNIN_FRAC * n_iter))
    idx     = (burnin + 1):n_iter

    nc = calculate_n_clusters(samples.c[idx, :])

    # derive a display name: cfg.name if available, else strip thinning suffix
    name = if !isnothing(cfg)
        string(cfg.name)
    else
        stem = basename(fpath)[1:end-5]
        replace(stem, r"_thinned$" => "")
    end

    push!(chains, ChainInfo(name, samples, diag, cfg, n_iter, idx, nc))
    @printf("  %-55s  %d post-burnin iters  mean_K=%.2f\n",
            name, length(idx), mean(nc))
end
println()

# ============================================================================
# Helper: acceptance rates from diag
# ============================================================================
function safe_ar(diag, sym)
    try
        ar = acceptance_rates(diag)
        v  = getfield(ar, sym)
        isnan(v) ? NaN : v
    catch
        NaN
    end
end

# ============================================================================
# Section 1 – Per-config trace plots
# ============================================================================

println("[Section 1] Per-config trace plots → $(joinpath(OUT_DIR, "traces"))")

for (i, ch) in enumerate(chains)
    col  = color_for(i)
    s    = ch.samples
    idx  = ch.idx
    name = ch.name
    pfx  = joinpath(OUT_DIR, "traces", replace(name, r"[^A-Za-z0-9_\-]" => "_"))

    # K trace
    p = plot(ch.nc; title="K trace — $name", xlabel="Post-burnin iteration",
             ylabel="Number of clusters", color=col, alpha=0.7, linewidth=0.8, legend=false)
    savefig(p, "$(pfx)_k_trace.png")

    # K distribution (bar chart)
    cm = countmap(ch.nc)
    ks = sort(collect(keys(cm)))
    ps = [cm[k] / length(ch.nc) for k in ks]
    p = bar(ks, ps; title="K distribution — $name", xlabel="K",
            ylabel="Posterior probability", color=col, legend=false)
    savefig(p, "$(pfx)_k_dist.png")

    # r trace (global dispersion)
    r_post = s.r[idx]
    p = plot(r_post; title="r trace — $name", xlabel="Post-burnin iteration",
             ylabel="r (dispersion)", color=col, alpha=0.7, linewidth=0.8, legend=false)
    savefig(p, "$(pfx)_r_trace.png")

    p = density(r_post; title="r density — $name", xlabel="r", ylabel="Density",
                color=col, linewidth=2, legend=false)
    savefig(p, "$(pfx)_r_density.png")

    # α trace
    α_post = s.α_ddcrp[idx]
    p = plot(α_post; title="α trace — $name", xlabel="Post-burnin iteration",
             ylabel="α (DDCRP concentration)", color=col, alpha=0.7, linewidth=0.8, legend=false)
    savefig(p, "$(pfx)_alpha_trace.png")

    p = density(α_post; title="α density — $name", xlabel="α", ylabel="Density",
                color=col, linewidth=2, legend=false)
    savefig(p, "$(pfx)_alpha_density.png")

    # log-posterior trace
    lp_post = s.logpost[idx]
    p = plot(lp_post; title="Log-posterior trace — $name", xlabel="Post-burnin iteration",
             ylabel="Log-posterior", color=col, alpha=0.7, linewidth=0.8, legend=false)
    savefig(p, "$(pfx)_logpost_trace.png")

    @printf("  %s  saved\n", name)
end
println()

# ============================================================================
# Section 2 – K distribution overlay (all configs on one plot)
# ============================================================================

println("[Section 2] K distribution overlay")

# Separate Gibbs reference from RJMCMC configs
gibbs_chains  = filter(c -> startswith(c.name, "gibbs"), chains)
rjmcmc_chains = filter(c -> !startswith(c.name, "gibbs"), chains)

# Group RJMCMC configs by fixed-dim proposal for sub-panels
fd_labels = unique([split(c.name, "_")[end] for c in rjmcmc_chains])

for fd in fd_labels
    grp = filter(c -> endswith(c.name, fd), rjmcmc_chains)
    isempty(grp) && continue

    p = plot(title="K distribution ($fd fixed-dim)", xlabel="K",
             ylabel="Posterior probability", legend=:outertopright,
             size=(900, 500))

    # Gibbs reference
    for gc in gibbs_chains
        cm = countmap(gc.nc); ks = sort(collect(keys(cm)))
        ps = [cm[k] / length(gc.nc) for k in ks]
        plot!(p, ks, ps; label=gc.name, color=:black, linestyle=:dash,
              linewidth=2, markershape=:none)
    end

    for (j, ch) in enumerate(grp)
        cm = countmap(ch.nc); ks = sort(collect(keys(cm)))
        ps = [cm[k] / length(ch.nc) for k in ks]
        blabel = replace(ch.name, "_$(fd)" => "")
        plot!(p, ks, ps; label=blabel, color=color_for(j),
              linewidth=1.5, markershape=:circle, markersize=4)
    end

    savefig(p, joinpath(OUT_DIR, "comparison", "k_dist_fixed_$(fd).png"))
    println("  K distribution (fixed=$fd) saved")
end

# Single overlay of all configs coloured by birth proposal
birth_types = unique([join(split(c.name, "_")[2:end-1], "_") for c in rjmcmc_chains])
birth_color = Dict(bt => color_for(i) for (i, bt) in enumerate(birth_types))

for fd in fd_labels
    grp = filter(c -> endswith(c.name, fd), rjmcmc_chains)
    isempty(grp) && continue

    p = plot(title="K distribution overlay ($fd)", xlabel="K",
             ylabel="Posterior probability", legend=:outertopright, size=(1000, 520))

    for gc in gibbs_chains
        cm = countmap(gc.nc); ks = sort(collect(keys(cm)))
        ps = [cm[k] / length(gc.nc) for k in ks]
        plot!(p, ks, ps; label="Gibbs ref", color=:black, linestyle=:dash, linewidth=2)
    end

    for ch in grp
        bt  = join(split(ch.name, "_")[2:end-1], "_")
        col = get(birth_color, bt, :gray)
        cm  = countmap(ch.nc); ks = sort(collect(keys(cm)))
        ps  = [cm[k] / length(ch.nc) for k in ks]
        plot!(p, ks, ps; label=ch.name, color=col, linewidth=1.2, alpha=0.8)
    end

    savefig(p, joinpath(OUT_DIR, "comparison", "k_dist_all_birth_$(fd).png"))
end
println()

# ============================================================================
# Section 3 – Comparison bar charts: ESS(K), acceptance rates, mean K, mean r
# ============================================================================

println("[Section 3] Comparison bar charts")

names_rj  = [ch.name for ch in rjmcmc_chains]
ess_vals  = [effective_sample_size(Float64.(ch.nc))            for ch in rjmcmc_chains]
mean_K    = [mean(ch.nc)                                        for ch in rjmcmc_chains]
std_K     = [std(Float64.(ch.nc))                               for ch in rjmcmc_chains]
mean_r    = [mean(ch.samples.r[ch.idx])                         for ch in rjmcmc_chains]
ar_birth  = [safe_ar(ch.diag, :birth)                           for ch in rjmcmc_chains]
ar_death  = [safe_ar(ch.diag, :death)                           for ch in rjmcmc_chains]
ar_fixed  = [safe_ar(ch.diag, :fixed)                           for ch in rjmcmc_chains]

n_rj = length(rjmcmc_chains)
xs   = 1:n_rj
rot  = 60

# ESS(K)
p = bar(xs, ess_vals; xticks=(xs, names_rj), xrotation=rot,
        title="ESS(K) by proposal config", ylabel="ESS(K)",
        legend=false, size=(max(900, 25*n_rj), 500), bottom_margin=80Plots.px)
savefig(p, joinpath(OUT_DIR, "comparison", "ess_K.png"))

# Mean K ± std (with Gibbs reference lines)
p = bar(xs, mean_K; xticks=(xs, names_rj), xrotation=rot,
        title="Mean K by proposal config", ylabel="Mean K",
        legend=:outertopright, size=(max(900, 25*n_rj), 500), bottom_margin=80Plots.px,
        label="RJMCMC")
for gc in gibbs_chains
    hline!(p, [mean(gc.nc)]; linestyle=:dash, color=:black,
           linewidth=2, label="$(gc.name) mean")
end
savefig(p, joinpath(OUT_DIR, "comparison", "mean_K.png"))

# Birth acceptance rate
valid_birth = .!isnan.(ar_birth)
if any(valid_birth)
    p = bar(xs[valid_birth], ar_birth[valid_birth];
            xticks=(xs[valid_birth], names_rj[valid_birth]), xrotation=rot,
            title="Birth acceptance rate", ylabel="Acceptance rate",
            legend=false, size=(max(900, 25*n_rj), 500), bottom_margin=80Plots.px)
    savefig(p, joinpath(OUT_DIR, "comparison", "ar_birth.png"))
end

# Death acceptance rate
valid_death = .!isnan.(ar_death)
if any(valid_death)
    p = bar(xs[valid_death], ar_death[valid_death];
            xticks=(xs[valid_death], names_rj[valid_death]), xrotation=rot,
            title="Death acceptance rate", ylabel="Acceptance rate",
            legend=false, size=(max(900, 25*n_rj), 500), bottom_margin=80Plots.px)
    savefig(p, joinpath(OUT_DIR, "comparison", "ar_death.png"))
end

# Fixed acceptance rate
valid_fixed = .!isnan.(ar_fixed)
if any(valid_fixed)
    p = bar(xs[valid_fixed], ar_fixed[valid_fixed];
            xticks=(xs[valid_fixed], names_rj[valid_fixed]), xrotation=rot,
            title="Fixed-dim acceptance rate", ylabel="Acceptance rate",
            legend=false, size=(max(900, 25*n_rj), 500), bottom_margin=80Plots.px)
    savefig(p, joinpath(OUT_DIR, "comparison", "ar_fixed.png"))
end

# Mean r
p = bar(xs, mean_r; xticks=(xs, names_rj), xrotation=rot,
        title="Mean r (dispersion) by proposal config", ylabel="Mean r",
        legend=:outertopright, size=(max(900, 25*n_rj), 500), bottom_margin=80Plots.px,
        label="RJMCMC")
for gc in gibbs_chains
    hline!(p, [mean(gc.samples.r[gc.idx])]; linestyle=:dash, color=:black,
           linewidth=2, label="$(gc.name) mean r")
end
savefig(p, joinpath(OUT_DIR, "comparison", "mean_r.png"))

println("  Comparison bar charts saved")
println()

# ============================================================================
# Section 4 – Heatmap: mean K over (birth proposal × fixed-dim proposal)
# ============================================================================

println("[Section 4] Proposal grid heatmaps")

# Extract unique birth / fixed-dim labels from config names
# Name pattern: rjmcmc_<birth_label>_<fixed_dim_label>
function parse_birth_fd(name)
    # drop the "rjmcmc_" prefix then split on last "_" for fd
    s = replace(name, r"^rjmcmc_" => "")
    # fd is the last component
    parts = split(s, "_")
    # fixed-dim labels known from proposal comparison script
    fd_known = ["noupdate", "weighted", "resample"]
    for fd in fd_known
        if parts[end] == fd
            birth = join(parts[1:end-1], "_")
            return birth, fd
        end
    end
    return s, "unknown"
end

birth_set = String[]
fd_set    = String[]
for ch in rjmcmc_chains
    b, fd = parse_birth_fd(ch.name)
    push!(birth_set, b); push!(fd_set, fd)
end
births = unique(birth_set)
fds    = unique(fd_set)

mk_grid(vals) = [
    let ch_list = filter(c -> begin b,fd = parse_birth_fd(c.name); b==b_ && fd==fd_; end,
                         rjmcmc_chains)
        isempty(ch_list) ? NaN : vals[findfirst(c -> c==ch_list[1], rjmcmc_chains)]
    end
    for b_ in births, fd_ in fds
]

for (metric, vals, title_str) in [
    ("mean_K",  mean_K,  "Mean K"),
    ("ess_K",   ess_vals, "ESS(K)"),
    ("ar_birth",ar_birth, "Birth acceptance rate"),
    ("ar_fixed",ar_fixed, "Fixed acceptance rate"),
    ("mean_r",  mean_r,  "Mean r"),
]
    grid = mk_grid(vals)
    finite_vals = filter(!isnan, grid)
    isempty(finite_vals) && continue

    clims = (minimum(finite_vals), maximum(finite_vals))
    p = heatmap(fds, births, grid;
                title=title_str, xlabel="Fixed-dim proposal",
                ylabel="Birth proposal", color=:viridis,
                clims=clims, aspect_ratio=:none,
                size=(300 + 120*length(fds), 150 + 55*length(births)))
    savefig(p, joinpath(OUT_DIR, "comparison", "grid_$(metric).png"))
    println("  Grid heatmap ($title_str) saved")
end
println()

# ============================================================================
# Section 5 – K trace overlay (all RJMCMC configs, one panel per fixed-dim)
# ============================================================================

println("[Section 5] K trace overlays")

for fd in fd_labels
    grp = filter(c -> endswith(c.name, fd), rjmcmc_chains)
    isempty(grp) && continue

    p = plot(title="K traces ($fd)", xlabel="Post-burnin iteration",
             ylabel="K", legend=:outertopright, size=(1100, 500), alpha=0.5)

    for (j, ch) in enumerate(grp)
        blabel = replace(ch.name, "_$(fd)" => "")
        plot!(p, ch.nc; label=blabel, color=color_for(j), linewidth=0.7)
    end

    savefig(p, joinpath(OUT_DIR, "comparison", "k_trace_overlay_$(fd).png"))
    println("  K trace overlay ($fd) saved")
end
println()

# ============================================================================
# Section 6 – Summary table: posterior K statistics for all configs
# ============================================================================

println("[Section 6] Summary table")

all_chains = vcat(gibbs_chains, rjmcmc_chains)

df_summary = DataFrame(
    config    = [ch.name             for ch in all_chains],
    n_iters   = [length(ch.idx)      for ch in all_chains],
    mean_K    = [mean(ch.nc)         for ch in all_chains],
    median_K  = [median(ch.nc)       for ch in all_chains],
    mode_K    = [mode(ch.nc)         for ch in all_chains],
    std_K     = [std(Float64.(ch.nc)) for ch in all_chains],
    q05_K     = [quantile(Float64.(ch.nc), 0.05) for ch in all_chains],
    q95_K     = [quantile(Float64.(ch.nc), 0.95) for ch in all_chains],
    ess_K     = [effective_sample_size(Float64.(ch.nc)) for ch in all_chains],
    mean_r    = [mean(ch.samples.r[ch.idx])   for ch in all_chains],
    mean_α    = [mean(ch.samples.α_ddcrp[ch.idx]) for ch in all_chains],
    ar_birth  = [safe_ar(ch.diag, :birth) for ch in all_chains],
    ar_death  = [safe_ar(ch.diag, :death) for ch in all_chains],
    ar_fixed  = [safe_ar(ch.diag, :fixed) for ch in all_chains],
)

csv_path = joinpath(OUT_DIR, "tables", "posterior_K_summary.csv")
CSV.write(csv_path, df_summary)
println("  Table saved to $csv_path\n")

# Console print
@printf("%-50s %8s %8s %8s %8s %8s %8s %8s %8s\n",
        "Config", "mean_K", "med_K", "mode_K", "std_K",
        "ESS_K", "mean_r", "ar_birth", "ar_death")
println("-" ^ 120)
for r in eachrow(df_summary)
    @printf("%-50s %8.2f %8.1f %8d %8.2f %8.1f %8.4f %8.4f %8.4f\n",
            r.config, r.mean_K, r.median_K, r.mode_K, r.std_K, r.ess_K,
            r.mean_r,
            isnan(r.ar_birth) ? 0.0 : r.ar_birth,
            isnan(r.ar_death) ? 0.0 : r.ar_death)
end

# ============================================================================
# Section 7 – K distribution comparison: all configs as stacked panel
# ============================================================================

println("\n[Section 7] Combined K-distribution panel (all configs)")

n_all = length(all_chains)
ncols = 5
nrows = cld(n_all, ncols)
plts  = map(enumerate(all_chains)) do (i, ch)
    cm = countmap(ch.nc); ks = sort(collect(keys(cm)))
    ps = [cm[k] / length(ch.nc) for k in ks]
    bar(ks, ps; title=ch.name, xlabel="K", ylabel="Pr",
        legend=false, titlefontsize=6, tickfontsize=5,
        color=startswith(ch.name, "gibbs") ? :black : color_for(i))
end

p_all = plot(plts...; layout=(nrows, ncols),
             size=(ncols*220, nrows*200),
             plot_title="Posterior K — all configs")
savefig(p_all, joinpath(OUT_DIR, "comparison", "k_dist_panel_all.png"))
println("  Combined panel saved")

println("\nAnalysis complete. All outputs in $OUT_DIR")
