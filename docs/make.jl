using DistanceDependentCRP
using Documenter

DocMeta.setdocmeta!(DistanceDependentCRP, :DocTestSetup, :(using DistanceDependentCRP); recursive=true)

makedocs(;
    modules=[DistanceDependentCRP],
    authors="Joseph Marsh <joe.s.marsh@gmail.com> and contributors",
    sitename="DistanceDependentCRP.jl",
    warnonly = [:missing_docs],
    format=Documenter.HTML(;
        canonical="https://jmarsh96.github.io/DistanceDependentCRP.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"            => "index.md",
        "Getting Started" => "getting_started.md",
        "Models" => [
            "Poisson"  => "models/poisson.md",
            "Binomial" => "models/binomial.md",
            "Gamma"    => "models/gamma.md",
        ],
        "Adding Your Own Model" => "extending.md",
        "API Reference"         => "api.md",
    ],
)

deploydocs(;
    repo="github.com/jmarsh96/DistanceDependentCRP.jl",
    devbranch="main",
)
