module SimpleContinuation

using FunctionWrappersWrappers
using LinearAlgebra
using FastClosures
using Printf

# Numerical methods
using LinearSolve
using NonlinearSolve

include("type_flags.jl")

# Problem interface
include("function.jl")
include("problem.jl")

# PALC algorithm
include("PALC/palc.jl")
include("PALC/callback.jl")
include("PALC/initialization.jl")
include("PALC/prediction.jl")
include("PALC/correction.jl")
include("PALC/nm_cache.jl")
include("PALC/continuation.jl")

export Silent, ContinuaitonSteps, ContinuationAndNewtonSteps
export Bordered, Secant

export ContinuationFunction
export ContinuationProblem

export TerminateContinuationCallback, AnalysisContinuationCallback

export PALC
export continuation

end
