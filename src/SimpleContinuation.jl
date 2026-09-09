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

export Silent, ContinuationSteps, ContinuationAndNewtonSteps
export Bordered, Secant

export ContinuationFunction, SparseContinuationFunction
export ContinuationProblem

export TerminateContinuationCallback, AnalysisContinuationCallback, TerminateContinuationCallbackSet
export FoldBifurcationTerminationCallback, FoldBifurcationDetectionCallback

export CorrectionStepLimiter

export PALC
export StandardDotProduct
export ScaledInnerProduct
export DoubleScaledInnerProduct
export BifurcationKitInnerProduct
export SecantInitialTangent, BorderedInitialTangent, UserInitialTangent
export continuation

end
