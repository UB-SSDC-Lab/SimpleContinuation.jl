# Struct for storing all information for the PALC algorithm
# (includes algorithm linear and nonlinear solve dependancies)
"""
    PALC

Data type for storing all information for PALC algorithm.
"""
struct PALC{P,D<:AbstractInnerProduct,LS,NLS,NTC}
    # PALC normalization
    inner_prod::D

    # Numerical method options
    linsolve::LS
    nlsolve::NLS
    termcond::NTC

    @doc"""
        PALC(; kwargs...)
    
    Contructor for PALC algorithm.
    
    # Kwargs
    - `predicter::AbstractPredictor`: Prediction method. Default (recommended): `Bordered()`
    - `inner_product::AbstractInnerProduct`: Inner product to use throughout. Default: `StandardDotProduct()`
    - `linesearch`: Line search method provided by `LineSearch.jl`. Default: `LiFukushimaLineSearch()`
    - `linsolve`: Linear solve algorithm provided by `LinearSolve.jl`. Default `SVDFactorization()`
    - `termcond`: Termination condition for solvers. Default: `termcond=NonlinearSolve.AbsTerminationMode()`

    # Examples
    ```julia
    pred = Secant() # secant predictor
    alg = PALC(; predicter=pred) # all defaults except prediction method
    ```
    """
    function PALC(;
        predicter=Bordered(),
        inner_prod=StandardDotProduct(),
        linesearch=LiFukushimaLineSearch(),
        linsolve=SVDFactorization(),
        termcond=NonlinearSolve.AbsTerminationMode(),
    )
        if !(predicter isa AbstractPredictor)
            error("Predictor type not recognized")
        end
        if !(linsolve isa LinearSolve.SciMLLinearSolveAlgorithm)
            error("Linear solver not recognized")
        end

        # Form newton solver
        nls = NewtonRaphson(;
            linsolve=linsolve,
            linesearch=linesearch,
            autodiff=nothing, # Our functions are currently not differentiable
        )

        # Construct and return PALC
        return new{
            typeof(predicter),
            typeof(inner_prod),
            typeof(linsolve),
            typeof(nls),
            typeof(termcond),
        }(
            inner_prod, linsolve, nls, termcond
        )
    end
end

"""
    PALCCache{MT<:Union{Matrix{Float64},SparseMatrixCSC{Float64,Int}}}

A cache for the PALC algorithm.

This cache includes all preallocated storage required for PALC. It also contains the zero curve (solution), return code, and any special detected points.

# Fields
- `ds::Float64`: PALC step size
- `br::Vector{Tuple{Vector{Float64},Float64}}`: Zero curve storage, in format `[(u0,λ0); (u1,λ1); ...; (un,λn)]` for `n` iterations.
- `detected_points::Vector{Tuple{Vector{Float64},Float64, Symbol}}`: Storage for any points of interest detected by `FoldBifurcationDetectionCallback()` or any other detection callbacks (once implemented).
- `uλ0::Vector{Float64}`: Full `[u;λ]` current iterate
- `u0::Vector{Float64}`: Unknowns `u` current iterate.
- `λ0::Float64`: Continuation parameter `λ` current iterate.
- `λn::Float64`: Natural continuation parameter. Only used in select instances when required.
- `bordered_mat::MT: Bordered matrix storage`
- `bordered_b::Vector{Float64}`: Storage for RHS of bordered system.
- `δuλ0::Vector{Float64}`: Storage for full tangent `[δu; δλ]`
- `δu0::Vector{Float64}`: Storage for tangent components `δu`
- `δλ::Float64`: Storage for tangent component `δλ`
- `δuλ0_initial::Vector{Float64}`: Storage for initalized tangent computed before first iteration.
- `uλpred::Vector{Float64}`: Storage for predicted point
- `Ffun::Vector{Float64}`: Storage for continuation function residuals (not including hyperplane constraint)
- `Jfun::MT`: Storage for non-square jacobian of system ∂F(u;λ)/∂[u;λ]. (not including hyperplane constraint)
- `u_0::Vector{Float64}`: Storage for regula-falsi solver
- `u_1::Vector{Float64}`: Storage for regula-falsi solver
- `u_t::Vector{Float64}`: Storage for regula-falsi solver
-  `ret::Symbol`: Exit condition return code.
"""
mutable struct PALCCache{MT<:Union{Matrix{Float64},SparseMatrixCSC{Float64,Int}}}
    # Algorithm parameters
    ds::Float64

    # Continuation curve
    br::Vector{Tuple{Vector{Float64},Float64}}

    # Special points (found with Detection Callbacks)
    detected_points::Vector{Tuple{Vector{Float64},Float64, Symbol}}

    # Current iterate
    uλ0::Vector{Float64}
    u0::Vector{Float64}
    λ0::Float64

    # Natural Continuation
    λn::Float64 # Predicted parameter in natural continuation

    # PALC Prediction
    bordered_mat::MT
    bordered_b::Vector{Float64}
    δuλ0::Vector{Float64}
    δu0::Vector{Float64}
    δλ0::Float64

    # Store the initial tangent so we can avoid recomputing
    δuλ0_initial::Vector{Float64}

    # PALC Correction
    uλpred::Vector{Float64}
    Ffun::Vector{Float64}   # Storage for the function residuals (not including hyperplane constraint)
    Jfun::MT                # Storage for the function Jacobian (not including hyperplane constraint)

    # PALC Callback Root-Find
    u_0::Vector{Float64}
    u_1::Vector{Float64}
    u_t::Vector{Float64}

    # Verbose step info
    tangents::Vector{Vector{Float64}}
    predictions::Vector{Vector{Float64}}
    dss::Vector{Float64}

    # Return code
    ret::Symbol # could also do a custom type for pretty viewing like sciml but symbols should work just fine
    # current implemented retcodes:
    # :HitBound; successful, hit boundary
    # :Callback; successful, terminated due to callback
    # :Callbacki; successful, terminated due to callback i in callback set
    # :Maxiters; unsuccessful (generally), terminated due to max iterations
    # :MinimumStepSize; unsuccessful, terminated due to shrinking stepsize
    # :None; this is what the retcode is initialized to, so seeing this after running continuation signifies an unhandled exit condition
end

function PALCCache(
    p::ContinuationProblem{F}, alg::PALC, ds0
) where {F<:ContinuationFunction}
    # Get initial solution and problem size
    u0 = p.u0
    λ0 = p.λ0
    n = length(u0)

    # Allocate memory for current iterate
    u0c = copy(u0)
    uλ0 = Vector{Float64}(undef, n + 1)
    uλ0[1:n] .= u0c
    uλ0[n + 1] = λ0

    # Allocate memory for storing curve
    br = Vector{Tuple{Vector{Float64},Float64}}(undef, 0)
    detected_points = Vector{Tuple{Vector{Float64},Float64, Symbol}}(undef, 0)

    # Allocate memory for prediction
    δu0 = similar(u0)
    δuλ0 = Vector{Float64}(undef, n + 1)
    bm = Matrix{Float64}(undef, n + 1, n + 1)
    bb = zeros(n + 1)
    bb[end] = 1.0

    # Allocate memory for initial tangent (obtained with secant method)
    δuλ0_i = similar(δuλ0)

    # Allocate memory for correction
    uλpred = Vector{Float64}(undef, n + 1)
    Ffun = similar(u0)
    Jfun = Matrix{Float64}(undef, n, n + 1)

    # Allocate memory for regula-falsi root-find
    u_0 = similar(u0)
    u_1 = similar(u0)
    u_t = similar(u0)

    # Allocate memory for verbose info
    tangents = Vector{Vector{Float64}}(undef, 0)
    predictions = similar(tangents)
    dss = Vector{Float64}(undef, 0)

    ret = :None # init return code (should always be overwritten)
    return PALCCache{Matrix{Float64}}(
        ds0,
        br,
        detected_points,
        uλ0,
        u0c,
        λ0,
        λ0,
        bm,
        bb,
        δuλ0,
        δu0,
        0.0,
        δuλ0_i,
        uλpred,
        Ffun,
        Jfun,
        u_0,
        u_1,
        u_t,
        tangents,
        predictions,
        dss,
        ret
    )
end
function PALCCache(
    p::ContinuationProblem{F}, alg::PALC, ds0
) where {F<:SparseContinuationFunction}
    # Get Jacobian prototypes
    Ju_prototype = p.f.Ju_prototype
    J_prototype = p.f.J_prototype

    # Get initial solution and problem size
    u0 = p.u0
    λ0 = p.λ0
    n = length(u0)

    # Allocate memory for current iterate
    u0c = copy(u0)
    uλ0 = Vector{Float64}(undef, n + 1)
    uλ0[1:n] .= u0c
    uλ0[n + 1] = λ0

    # Allocate memory for storing curve
    br = Vector{Tuple{Vector{Float64},Float64}}(undef, 0)
    detected_points = Vector{Tuple{Vector{Float64},Float64, Symbol}}(undef, 0)

    # Allocate memory for prediction
    δu0 = similar(u0)
    δuλ0 = Vector{Float64}(undef, n + 1)
    bm = vcat(J_prototype, sparse(ones(1, n + 1)))
    bb = zeros(n + 1)
    bb[end] = 1.0

    # Allocate memory for initial tangent (obtained with secant method)
    δuλ0_i = similar(δuλ0)

    # Allocate memory for correction
    uλpred = Vector{Float64}(undef, n + 1)
    Ffun = similar(u0)
    Jfun = copy(J_prototype)

    # Allocate memory for regula-falsi root-find
    u_0 = similar(u0)
    u_1 = similar(u0)
    u_t = similar(u0)

    # Allocate memory for verbose info
    tangents = Vector{Vector{Float64}}(undef, 0)
    predictions = similar(tangents)
    dss = Vector{Float64}(undef, 0)

    ret = :None # init return code

    return PALCCache{SparseMatrixCSC{Float64,Int}}(
        ds0,
        br,
        detected_points,
        uλ0,
        u0c,
        λ0,
        λ0,
        bm,
        bb,
        δuλ0,
        δu0,
        0.0,
        δuλ0_i,
        uλpred,
        Ffun,
        Jfun,
        u_0,
        u_1,
        u_t,
        tangents,
        predictions,
        dss,
        ret
    )
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

function set_successful_iterate!(
    cache::PALCCache, u::Vector{Float64}, λ::Float64, push_point::Bool=true
)
    # Set the current iterate
    cache.u0 .= u
    cache.λ0 = λ

    n = length(u)
    cache.uλ0[1:n] .= u
    cache.uλ0[end] = λ

    # Push to continuation curve
    if push_point
        push!(cache.br, (copy(cache.u0), cache.λ0))
    end
    return nothing
end
function set_successful_iterate!(
    cache::PALCCache, uλ::Vector{Float64}, push_point::Bool=true
)
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
