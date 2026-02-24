# ============================================================================
# Proposal Comparison Study: dd-CRP with Skew-Normal Likelihood
#
# Design:
#   Two fixed datasets (same seed=42):
#     - "distinct":    3 well-separated clusters (ξ = [−10, 0, 10])
#     - "overlapping": 3 overlapping clusters    (ξ = [−4,  0,  4])
#
#   x covariates are drawn from cluster-specific normals:
#     - "distinct":    centers = [−8, 0, 8], σ = 2.0
#     - "overlapping": centers = [−3, 0, 3], σ = 1.5
#
#   2 birth proposals × 6 fixed-dim combinations = 12 MCMC runs per scenario.
#
#   ξ and α birth: NormalMomentMatch (always)
#   ω birth:       InverseGammaMomentMatch  |  LogNormalMomentMatch
#
#   Fixed-dim (MixedFixedDim):
#     ξ and α: no_update | resample | weighted_mean   (3 choices)
#     ω:       no_update | resample                   (2 choices)
#     → 3 × 2 = 6 combinations per birth proposal
#
# Metrics per run (post-burnin):
#   - Acceptance rates: birth, death, overall
#   - Clustering: mean ARI, final ARI, mean VI, mean K, P(K=K_true)
#   - ESS: ESS(K), ESS(logpost), ESS/sec
#   - Wall-clock time
#
# Outputs: results/skew_norm_proposal_study/{distinct,overlapping}/
#   summary_table.csv, table.tex,
#   figures/{k_trace, ari_trace, acceptance_rates, ess_comparison}.png
#   results/skew_norm_proposal_study/data_scenarios.{png,pdf}
# ============================================================================

using DDCRP, StatsPlots, DataFrames, CSV, Random, Statistics, Distributions, Printf

# ── Output directory ──────────────────────────────────────────────────────────

const OUTDIR = joinpath(@__DIR__, "..", "results", "skew_norm_proposal_study")

# ── Constants ─────────────────────────────────────────────────────────────────

const N_SAMPLES = 50_000
const BURNIN    = 10_000
const K_TRUE    = 3

# ── X covariate generation ────────────────────────────────────────────────────
# Draw n observations from K cluster-specific normals.
# Each cluster gets n÷K observations (remainder added to last cluster).

function make_cluster_x(n::Int, centers::Vector{Float64}, σ::Float64; seed::Int=0)
    rng   = MersenneTwister(seed)
    K     = length(centers)
    sizes = fill(n ÷ K, K)
    sizes[end] += n - sum(sizes)
    vcat([centers[k] .+ σ * randn(rng, sizes[k]) for k in 1:K]...)
end

# ── Datasets (generated once, reused across all runs) ─────────────────────────

Random.seed!(42)

const X_DISTINCT    = make_cluster_x(150, [-8.0, 0.0, 8.0], 2.0; seed=0)
const X_OVERLAPPING = make_cluster_x(150, [-3.0, 0.0, 3.0], 1.5; seed=0)

const SIM_DISTINCT = simulate_skewnormal_data(
    150, [-10.0, 0.0, 10.0], [2.0, 1.0, 2.0], [4.0, 0.0, -4.0];
    α=0.5, scale=5.0, x=X_DISTINCT
)

const SIM_OVERLAPPING = simulate_skewnormal_data(
    150, [-4.0, 0.0, 4.0], [2.0, 2.0, 2.0], [2.0, 0.0, -2.0];
    α=0.5, scale=5.0, x=X_OVERLAPPING
)

# ── Shared model / sampler configuration ─────────────────────────────────────

const MODEL        = SkewNormalCluster()
const DDCRP_PARAMS = DDCRPParams(0.5, 5.0)

# Slightly uninformative priors
const PRIORS = SkewNormalClusterPriors(
    ξ_μ=0.0, ξ_σ=10.0,
    ω_a=2.0, ω_b=1.0,
    α_μ=0.0, α_σ=5.0,
)

const OPTS = MCMCOptions(
    n_samples         = N_SAMPLES,
    track_diagnostics = true,
)

# ── Per-parameter birth proposals ────────────────────────────────────────────
# ξ and α always use NormalMomentMatch; ω is varied.

const ξ_BIRTH = NormalMomentMatch([1.0])
const α_BIRTH = NormalMomentMatch([1.0])

const ω_METHODS = [
    (label="invgamma",  ω_birth=InverseGammaMomentMatch()),
    (label="lognormal", ω_birth=LogNormalMomentMatch([0.3])),
]

# ── Fixed-dim configurations ──────────────────────────────────────────────────

const ξα_FD_MODES = ["none", "resample", "wmean"]
const ω_FD_MODES  = ["none", "resample"]

function make_fixed_dim(ξα_mode::String, ω_fd_mode::String, ω_birth)
    if ξα_mode == "resample"
        ξ_fd = Resample(ξ_BIRTH)
        α_fd = Resample(α_BIRTH)
    elseif ξα_mode == "wmean"
        ξ_fd = WeightedMean()
        α_fd = WeightedMean()
    else
        ξ_fd = NoUpdate()
        α_fd = NoUpdate()
    end
    ω_fd = ω_fd_mode == "resample" ? Resample(ω_birth) : NoUpdate()
    return MixedFixedDim(ξ=ξ_fd, ω=ω_fd, α=α_fd)
end

# ── Colour palette ────────────────────────────────────────────────────────────

const N_FDIM       = length(ξα_FD_MODES) * length(ω_FD_MODES)
const COLOURS      = palette(:tab10)[1:N_FDIM]

# ── Manuscript: data-scenarios figure ────────────────────────────────────────
# Converts the link/table representation sim.tables to integer cluster labels.

function tables_to_labels(tables, n)
    labels = zeros(Int, n)
    for (k, t) in enumerate(tables)
        labels[t] .= k
    end
    return labels
end

function plot_data_scenarios()
    mkpath(OUTDIR)
    scenarios = [
        ("Distinct (ξ = [−10, 0, 10])",  SIM_DISTINCT),
        ("Overlapping (ξ = [−4, 0, 4])", SIM_OVERLAPPING),
    ]
    panels = map(scenarios) do (title_str, sim)
        n   = length(sim.y)
        lbl = tables_to_labels(sim.tables, n)
        scatter(sim.x, sim.y;
            group=lbl, palette=:tab10, markersize=3, alpha=0.7,
            xlabel="Spatial covariate x", ylabel="Observed y",
            title=title_str, legend=:topright)
    end
    fig = plot(panels...; layout=(1, 2), size=(1100, 420),
               plot_title="Simulated Data Scenarios")
    for ext in ("png", "pdf")
        savefig(fig, joinpath(OUTDIR, "data_scenarios.$ext"))
    end
    println("Data scenarios figure → $(OUTDIR)/data_scenarios.{png,pdf}")
end

# ── LaTeX table generator ─────────────────────────────────────────────────────

function latex_fmt(x::Float64, decimals::Int)
    Printf.format(Printf.Format("%.$(decimals)f"), x)
end

function generate_latex_table(scenario_name::String, results_df::DataFrame, outdir::String)
    tex_path = joinpath(outdir, "table.tex")

    birth_groups = unique(results_df.birth_proposal)
    n_per_group  = count(==(birth_groups[1]), results_df.birth_proposal)

    lines = String[]
    push!(lines, "% Required packages: booktabs, multirow, xcolor, colortbl")
    push!(lines, "% Generated automatically — re-run script to update")
    push!(lines, "")
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "  \\centering")
    push!(lines, "  \\caption{Proposal comparison --- $scenario_name scenario}")
    push!(lines, "  \\label{tab:$(scenario_name)}")
    push!(lines, "  \\small")
    push!(lines, "  \\begin{tabular}{l >{\\columncolor{blue!10}}c >{\\columncolor{orange!10}}c c c c r r r}")
    push!(lines, "    \\toprule")
    push!(lines, "    \\multirow{2}{*}{\$\\omega\$ birth}")
    push!(lines, "      & \\multicolumn{2}{c}{Fixed-dim updates}")
    push!(lines, "      & Mean ARI & Mean VI & \$P(K{=}K^*)\$ & ESS\$(K)\$ & ESS/sec & Time (s) \\\\")
    push!(lines, "    \\cmidrule(lr){2-3}")
    push!(lines, "      & \$(\\xi,\\alpha)\$ & \$\\omega\$ & & & & & & \\\\")
    push!(lines, "    \\midrule")

    for (g, blabel) in enumerate(birth_groups)
        sub = filter(r -> r.birth_proposal == blabel, results_df)
        for (i, row) in enumerate(eachrow(sub))
            birth_cell = if i == 1
                "    \\multirow{$(n_per_group)}{*}{$(blabel)}"
            else
                "    "
            end
            ari_s  = latex_fmt(row.mean_ari,    3)
            vi_s   = latex_fmt(row.mean_vi,     2)
            pk_s   = latex_fmt(row.prob_K_true, 3)
            essk_s = string(round(Int, row.ess_K))
            essp_s = latex_fmt(row.ess_per_sec, 1)
            time_s = latex_fmt(row.total_time,  1)
            data_cols = "$(row.fd_xialpha) & $(row.fd_omega) & $(ari_s) & $(vi_s) & $(pk_s) & $(essk_s) & $(essp_s) & $(time_s) \\\\"
            push!(lines, "$(birth_cell) & $(data_cols)")
        end
        if g < length(birth_groups)
            push!(lines, "    \\midrule")
        end
    end

    push!(lines, "    \\bottomrule")
    push!(lines, "  \\end{tabular}")
    push!(lines, "\\end{table}")

    open(tex_path, "w") do io
        println(io, join(lines, "\n"))
    end
    println("LaTeX table → $tex_path")
end

# ── Single Experiment ─────────────────────────────────────────────────────────

function run_experiment(birth_label::String, birth_proposal,
                        ξα_mode::String,     ω_fd_mode::String,
                        fixed_dim_proposal,
                        sim, data)
    total_time = @elapsed begin
        samples, diag = mcmc(MODEL, data, DDCRP_PARAMS, PRIORS, birth_proposal;
                             fixed_dim_proposal=fixed_dim_proposal, opts=OPTS)
    end

    post_c  = samples.c[(BURNIN+1):end, :]
    post_lp = samples.logpost[(BURNIN+1):end]

    ari_trace = compute_ari_trace(post_c, sim.c)
    vi_trace  = compute_vi_trace(post_c, sim.c)
    k_trace   = calculate_n_clusters(post_c)

    acc = acceptance_rates(diag)

    ess_K       = effective_sample_size(Float64.(k_trace))
    ess_logpost = effective_sample_size(post_lp)
    ess_per_sec = ess_K / total_time

    row = (
        birth_proposal = birth_label,
        fd_xialpha     = ξα_mode,
        fd_omega       = ω_fd_mode,
        K_simulated    = length(unique(sim.c)),
        acc_birth      = acc.birth,
        acc_death      = acc.death,
        acc_overall    = acc.overall,
        mean_ari       = mean(ari_trace),
        final_ari      = ari_trace[end],
        mean_vi        = mean(vi_trace),
        mean_K         = mean(k_trace),
        prob_K_true    = mean(k_trace .== K_TRUE),
        ess_K          = ess_K,
        ess_logpost    = ess_logpost,
        ess_per_sec    = ess_per_sec,
        total_time     = total_time,
    )

    return row, samples
end

# ── Full Scenario (12 runs + 4 figures) ───────────────────────────────────────

function run_study(scenario_name::String, sim)
    scenario_outdir = joinpath(OUTDIR, scenario_name)
    scenario_figdir = joinpath(scenario_outdir, "figures")
    mkpath(scenario_outdir)
    mkpath(scenario_figdir)

    data        = ContinuousData(sim.y, sim.D)
    k_simulated = length(unique(sim.c))

    println("=" ^ 65)
    println("Scenario: $scenario_name")
    println("  n=$(length(sim.y)), K_true=$K_TRUE, K_simulated=$k_simulated")
    println("  ω birth:      $(join([m.label for m in ω_METHODS], ", "))")
    println("  ξα fixed-dim: $(join(ξα_FD_MODES, ", "))")
    println("  ω  fixed-dim: $(join(ω_FD_MODES, ", "))")
    println("  Samples: $N_SAMPLES  |  Burnin: $BURNIN")
    println("=" ^ 65)
    println()

    results_df  = DataFrame()
    all_samples = Dict{String, SkewNormalClusterSamples}()

    for ω_method in ω_METHODS
        birth_proposal = MixedProposal(ξ=ξ_BIRTH, ω=ω_method.ω_birth, α=α_BIRTH)
        for ξα_mode in ξα_FD_MODES, ω_fd_mode in ω_FD_MODES
            run_label = "$(ω_method.label)/$(ξα_mode)/$(ω_fd_mode)"
            print("  $run_label ... ")
            flush(stdout)

            fixed_dim = make_fixed_dim(ξα_mode, ω_fd_mode, ω_method.ω_birth)
            row, samples = run_experiment(
                ω_method.label, birth_proposal, ξα_mode, ω_fd_mode, fixed_dim, sim, data
            )
            push!(results_df, row; promote=true)
            all_samples[run_label] = samples

            t = round(row.total_time; digits=1)
            a = round(row.mean_ari;   digits=3)
            k = round(row.mean_K;     digits=1)
            println("done ($(t)s, ARI=$(a), K̄=$(k))")
        end
        println()
    end

    # ── Save Summary Table ──────────────────────────────────────────────────

    CSV.write(joinpath(scenario_outdir, "summary_table.csv"), results_df)
    println("Summary table → $(joinpath(scenario_outdir, "summary_table.csv")) ($(nrow(results_df)) rows)")
    println()

    # ── LaTeX Table ─────────────────────────────────────────────────────────

    generate_latex_table(scenario_name, results_df, scenario_outdir)
    println()

    subset_birth(df, blabel) = filter(r -> r.birth_proposal == blabel, df)

    # fd legend label for trace plots
    fdim_legend(ξα, ωfd) = "ξα=$ξα, ω=$ωfd"

    # ── Figure 1: K Trace ──────────────────────────────────────────────────

    println("  Generating K trace figure...")
    let
        panels = []
        for ω_method in ω_METHODS
            p = plot(xlabel="Iteration", ylabel="K",
                     title="ω birth = $(ω_method.label)", legend=:topright)
            hline!(p, [K_TRUE]; color=:black, linestyle=:dash, linewidth=1.5,
                   label="K_true=$K_TRUE")
            colour_idx = 0
            for ξα_mode in ξα_FD_MODES, ω_fd_mode in ω_FD_MODES
                colour_idx += 1
                key    = "$(ω_method.label)/$(ξα_mode)/$(ω_fd_mode)"
                samps  = all_samples[key]
                k_full = calculate_n_clusters(samps.c)
                plot!(p, k_full; label=fdim_legend(ξα_mode, ω_fd_mode),
                      color=COLOURS[colour_idx], alpha=0.7, linewidth=1.0)
            end
            push!(panels, p)
        end
        fig = plot(panels...; layout=(1, 2), size=(1200, 420),
                   plot_title="K Trace — $scenario_name")
        savefig(fig, joinpath(scenario_figdir, "k_trace.png"))
        println("    Saved k_trace.png")
    end

    # ── Figure 2: ARI Trace (post-burnin) ─────────────────────────────────

    println("  Generating ARI trace figure...")
    let
        panels = []
        for ω_method in ω_METHODS
            p = plot(xlabel="Post-burnin iteration", ylabel="ARI",
                     title="ω birth = $(ω_method.label)",
                     legend=:bottomright, ylims=(-0.1, 1.05))
            hline!(p, [1.0]; color=:black, linestyle=:dash, linewidth=1.5,
                   label="Perfect (1.0)")
            colour_idx = 0
            for ξα_mode in ξα_FD_MODES, ω_fd_mode in ω_FD_MODES
                colour_idx += 1
                key    = "$(ω_method.label)/$(ξα_mode)/$(ω_fd_mode)"
                samps  = all_samples[key]
                post_c = samps.c[(BURNIN+1):end, :]
                ari_tr = compute_ari_trace(post_c, sim.c)
                plot!(p, ari_tr; label=fdim_legend(ξα_mode, ω_fd_mode),
                      color=COLOURS[colour_idx], alpha=0.7, linewidth=1.0)
            end
            push!(panels, p)
        end
        fig = plot(panels...; layout=(1, 2), size=(1200, 420),
                   plot_title="ARI Trace — $scenario_name")
        savefig(fig, joinpath(scenario_figdir, "ari_trace.png"))
        println("    Saved ari_trace.png")
    end

    # ── Figure 3: Acceptance Rates ─────────────────────────────────────────

    println("  Generating acceptance rates figure...")
    let
        panels = []
        for ω_method in ω_METHODS
            df_sub   = subset_birth(results_df, ω_method.label)
            births   = df_sub.acc_birth   .* 100
            deaths   = df_sub.acc_death   .* 100
            overalls = df_sub.acc_overall .* 100
            xlabels  = string.(df_sub.fd_xialpha, "/", df_sub.fd_omega)

            p = groupedbar(
                hcat(births, deaths, overalls);
                bar_position = :dodge,
                xticks       = (1:nrow(df_sub), xlabels),
                xrotation    = 30,
                label        = ["Birth" "Death" "Overall"],
                ylabel       = "Acceptance rate (%)",
                title        = "ω birth = $(ω_method.label)",
                legend       = :topright,
                ylims        = (0, 100),
            )
            push!(panels, p)
        end
        fig = plot(panels...; layout=(1, 2), size=(1200, 450),
                   plot_title="Acceptance Rates — $scenario_name")
        savefig(fig, joinpath(scenario_figdir, "acceptance_rates.png"))
        println("    Saved acceptance_rates.png")
    end

    # ── Figure 4: ESS Comparison ───────────────────────────────────────────

    println("  Generating ESS comparison figure...")
    let
        panels = []
        for ω_method in ω_METHODS
            df_sub   = subset_birth(results_df, ω_method.label)
            ess_K_v  = df_sub.ess_K
            ess_ps_v = df_sub.ess_per_sec
            xlabels  = string.(df_sub.fd_xialpha, "/", df_sub.fd_omega)

            p = groupedbar(
                hcat(ess_K_v, ess_ps_v);
                bar_position = :dodge,
                xticks       = (1:nrow(df_sub), xlabels),
                xrotation    = 30,
                label        = ["ESS(K)" "ESS/sec"],
                ylabel       = "ESS",
                title        = "ω birth = $(ω_method.label)",
                legend       = :topright,
            )
            push!(panels, p)
        end
        fig = plot(panels...; layout=(1, 2), size=(1200, 450),
                   plot_title="ESS Comparison — $scenario_name")
        savefig(fig, joinpath(scenario_figdir, "ess_comparison.png"))
        println("    Saved ess_comparison.png")
    end

    println()
    return results_df
end

# ── Generate manuscript data-scenarios figure ────────────────────────────────

plot_data_scenarios()

# ── Run both scenarios ────────────────────────────────────────────────────────

df_distinct    = run_study("distinct",    SIM_DISTINCT)
df_overlapping = run_study("overlapping", SIM_OVERLAPPING)

println("Study complete. Results in: $OUTDIR")
