module SimpleContinuation

using FunctionWrappersWrappers
using SparseArrays
using LinearAlgebra
using FastClosures
using Printf

# Numerical methods
using LinearSolve
using NonlinearSolve

using ForwardDiff: ForwardDiff

include("type_flags.jl")
include("inner_products.jl")

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
include("PALC/special_callbacks.jl")

export Silent, ContinuaitonSteps, ContinuationAndNewtonSteps
export Bordered, Secant

export ContinuationFunction, SparseContinuationFunction
export ContinuationProblem

export TerminateContinuationCallback, AnalysisContinuationCallback
export FoldBifurcationTerminationCallback

export PALC
export StandardDotProduct
export ScaledInnerProduct
export DoubleScaledInnerProduct
export BifurcationKitInnerProduct
export continuation

end
