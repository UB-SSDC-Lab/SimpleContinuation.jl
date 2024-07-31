# Predictor type flags
abstract type AbstractPredictor end
struct Bordered <: AbstractPredictor end

# Struct for storing all information for the PALC algorithm
# (includes algorithm linear and nonlinear solve dependancies)
struct PALC{P, LS, NLS, NTC}
    # Continuation parameters
    θ::Float64

    # Numerical method options
    linsolve::LS
    nlsolve::NLS
    termcond::NTC

    function PALC(; 
        predicter   = Bordered(),
        θ           = 0.5,
        linesearch  = LiFukushimaLineSearch(), 
        linsolve    = SVDFactorization(), 
        termcond    = AbsSafeBestTerminationMode(),
    )
        if !(predicter isa AbstractPredictor)
            error("Predictor type not recognized")
        end
        if !(linsolve isa LinearSolve.SciMLLinearSolveAlgorithm)
            error("Linear solver not recognized")
        end

        # Form newton solver
        nls = NewtonRaphson(;
            linsolve    = linsolve,
            linesearch  = linesearch,
            autodiff    = nothing, # Are functions are currently not differentiable
        )

        # Construct and return PALC
        new{typeof(predicter), typeof(linsolve), typeof(nls), typeof(termcond)}(
            θ, linsolve, nls, termcond,
        )
    end
end

# A Cache for the PALC algorithm
mutable struct PALCCache
    # Algorithm parameters
    θ::Float64
    ds::Float64

    # Continuation curve
    br::Vector{Tuple{Vector{Float64}, Float64}}

    # Current iterate
    uλ0::Vector{Float64}
    u0::Vector{Float64}
    λ0::Float64

    # Natural Continuation
    λn::Float64 # Predicted parameter in natural continuation

    # PALC Prediction
    bordered_mat::Matrix{Float64}
    bordered_b::Vector{Float64}
    δuλ0::Vector{Float64} 
    δu0::Vector{Float64}
    δλ0::Float64

    # Store the initial tangent so we can avoid recomputing
    δuλ0_initial::Vector{Float64}

    # PALC Correction
    uλpred::Vector{Float64}
    δu::Vector{Float64}     # The change in u for the corrent iteration during correction
    Ffun::Vector{Float64}   # Storage for the function residuals (not including hyperplane constraint)
    Jfun::Matrix{Float64}   # Storage for the function Jacobian (not including hyperplane constraint)
end

function PALCCache(p::ContinuationProblem, alg::PALC, ds0)
    # Get initial solution and problem size
    u0 = p.u0
    λ0 = p.λ0
    n  = length(u0)

    # Allocate memory for current iterate
    u0c         = copy(u0)
    uλ0         = Vector{Float64}(undef, n + 1)
    uλ0[1:n]   .= u0c
    uλ0[n+1]    = λ0

    # Allocate memory for storing curve
    br      = Vector{Tuple{Vector{Float64}, Float64}}(undef, 0)

    # Allocate memory for prediction
    δu0     = similar(u0)
    δuλ0    = Vector{Float64}(undef, n + 1)
    bm      = Matrix{Float64}(undef, n + 1, n + 1)
    bb      = zeros(n + 1); bb[end] = 1.0

    # Allocate memory for initial tangent (obtained with secant method)
    δuλ0_i  = similar(δuλ0)

    # Allocate memory for correction
    δu      = similar(u0)
    uλpred  = Vector{Float64}(undef, n + 1)
    Ffun    = similar(u0)
    Jfun    = Matrix{Float64}(undef, n, n + 1)

    PALCCache(alg.θ, ds0, br, uλ0, u0c, λ0, λ0, bm, bb, δuλ0, δu0, 0.0, δuλ0_i, uλpred, δu, Ffun, Jfun)
end

# Set natural continuation parameter as perturbed current parameter λ0 
function set_natural_continuation_parameter!(cache::PALCCache, λ)
    cache.λn = λ
    return nothing
end
function perturb_natural_continuation_parameter!(cache::PALCCache, δλ)
    cache.λn = cache.λ0 + δλ
    return nothing
end

function set_successful_iterate!(cache::PALCCache, u::Vector{Float64}, λ::Float64, push_point::Bool=true)
    # Set the current iterate
    cache.u0 .= u
    cache.λ0 = λ

    n = length(u)
    cache.uλ0[1:n] .= u
    cache.uλ0[end]  = λ

    # Push to continuation curve
    if push_point
        push!(cache.br, (copy(cache.u0), cache.λ0))
    end
    return nothing
end
function set_successful_iterate!(cache::PALCCache, uλ::Vector{Float64}, push_point::Bool=true)
    # Set the current iterate
    n = length(uλ) - 1
    cache.u0 .= view(uλ, 1:n)
    cache.λ0 = uλ[end]

    cache.uλ0 .= uλ

    # Push to continuation curve
    if push_point
        push!(cache.br, (copy(cache.u0), cache.λ0))
    end
    return nothing
end

function update_tangent!(cache::PALCCache, δuλ::Vector{Float64})
    # Update the tangent
    n = length(δuλ) - 1
    cache.δuλ0 .= δuλ
    cache.δu0 .= view(δuλ, 1:n)
    cache.δλ0 = δuλ[end]
    return nothing
end
function update_tangent!(cache::PALCCache, δu, δλ)
    # Update the tangent
    n = length(δu)
    cache.δu0 .= δu
    cache.δλ0 = δλ
    cache.δuλ0[1:n] .= δu
    cache.δuλ0[end] = δλ
    return nothing
end