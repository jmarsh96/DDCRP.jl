using DDCRP
using Documenter

DocMeta.setdocmeta!(DDCRP, :DocTestSetup, :(using DDCRP); recursive=true)

makedocs(;
    modules=[DDCRP],
    authors="Joseph Marsh <joe.s.marsh@gmail.com> and contributors",
    sitename="DDCRP.jl",
    format=Documenter.HTML(;
        canonical="https://jmarsh96.github.io/DDCRP.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jmarsh96/DDCRP.jl",
    devbranch="main",
)
