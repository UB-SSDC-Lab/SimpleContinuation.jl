using SimpleContinuation
using Documenter

DocMeta.setdocmeta!(SimpleContinuation, :DocTestSetup, :(using SimpleContinuation); recursive=true)

makedocs(;
    modules=[SimpleContinuation],
    authors="Grant Hecht",
    sitename="SimpleContinuation.jl",
    format=Documenter.HTML(;
        canonical="https://UB-SSDC-Lab.github.io/SimpleContinuation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/UB-SSDC-Lab/SimpleContinuation.jl",
    devbranch="main",
)
