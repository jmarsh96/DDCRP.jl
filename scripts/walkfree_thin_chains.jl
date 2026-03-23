# ============================================================================
# Walk Free – thin proposal-comparison chains by factor 10
# ============================================================================
#
# Reads every .jld2 file from results/walkfree/chains/, keeps every 10th
# sample (post-load thinning), and writes <name>_thinned.jld2 to
# results/walkfree/chains_thinned/.
#
# Run:
#   julia --project scripts/walkfree_thin_chains.jl
#
# The thinned files are intended for download / offline analysis.
# ============================================================================

using DDCRP          # must be loaded so JLD2 can deserialise DDCRP types
using JLD2
using Printf

const THIN    = 10
const SRC_DIR = "results/walkfree/chains"
const DST_DIR = "results/walkfree/chains_thinned"

mkpath(DST_DIR)

chain_files = filter(f -> endswith(f, ".jld2"), readdir(SRC_DIR; join=true))

if isempty(chain_files)
    error("No .jld2 files found in $SRC_DIR — run walkfree_proposal_comparison.jl first.")
end

println("Thinning $(length(chain_files)) chain(s) by factor $THIN → $DST_DIR\n")

for src_path in chain_files
    fname    = basename(src_path)
    stem     = fname[1:end-5]              # strip ".jld2"
    dst_path = joinpath(DST_DIR, "$(stem)_thinned.jld2")

    data = load(src_path)
    samples = data["samples"]
    diag    = data["diag"]
    cfg     = get(data, "cfg", nothing)

    n_iter = size(samples.c, 1)
    idx    = 1:THIN:n_iter                 # every THIN-th row

    # ── thin every matrix/vector field in the samples struct ──────────────────
    thinned_fields = map(fieldnames(typeof(samples))) do f
        v = getfield(samples, f)
        if v isa Matrix
            return v[idx, :]
        elseif v isa Vector
            return v[idx]
        else
            return v
        end
    end
    samples_thin = typeof(samples)(thinned_fields...)

    n_thin = length(idx)
    @printf("  %-55s  %6d → %6d samples\n", fname, n_iter, n_thin)

    if isnothing(cfg)
        jldsave(dst_path; samples=samples_thin, diag)
    else
        jldsave(dst_path; samples=samples_thin, diag, cfg)
    end
end

println("\nDone. Thinned chains saved to $DST_DIR")
