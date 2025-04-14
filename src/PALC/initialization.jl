# Initial tangent options
abstract type AbstractInitialTangent end

# Constructs initial tangent using the secant method with a single natural continuation step
struct SecantInitialTangent <: AbstractInitialTangent
    step_scale_factor::Float64
    function SecantInitialTangent(; step_scale_factor=1e-6)
        new(step_scale_factor)
    end
end

# Constructs initial tangent using the bordered method, using every base-vector
struct BorderedInitialTangent{I} <: AbstractInitialTangent
    positive_search_order::Bool
    search_order::I
    function BorderedInitialTangent(; positive_search_order=true)
        new{Nothing}(positive_search_order, nothing)
    end
    function BorderedInitialTangent(search_order::I) where {I<:AbstractVector}
        if !(eltype(search_order) <: Integer)
            throw(ArgumentError("The search order must be an integer vector!"))
        end
        new{I}(true, search_order)
    end
end
function get_search_order(bit::BorderedInitialTangent{Nothing}, n)
    return bit.positive_search_order ? (1:(n + 1)) : ((n + 1):-1:1)
end
function get_search_order(bit::BorderedInitialTangent, n)
    return bit.search_order
end

struct UserInitialTangent <: AbstractInitialTangent
    initial_tangent::Vector{Float64}
    function UserInitialTangent(user_initial_tangent)
        initial_tangent = Vector{Float64}(undef, length(user_initial_tangent))
        initial_tangent .= user_initial_tangent
        return new(initial_tangent)
    end
end

function compute_initial_tangent(
    method::SecantInitialTangent,
    cache::PALCCache,
    alg::PALC,
    prob::ContinuationProblem,
    solvers,
    trace::AbstractTraceLevel,
)
    # Perturb the natural continuation parameter to construct initial tangent
    λ_pert = method.step_scale_factor * cache.ds
    perturb_natural_continuation_parameter!(cache, λ_pert)

    # Print trace if desired
    print_initialization_trace(cache, trace, 2)

    # Resolve the problem with perturbed parameter
    u_sol, retcode = solve_natural_nlp!(solvers, cache.u0, trace)

    if SciMLBase.successful_retcode(retcode)
        # Compute and set the initial tangent
        u0 = cache.u0                       # The current solution (from first solve)
        n = length(u0)
        δuλ0 = cache.δuλ0                   # The predicted tangent direction

        # Set secant direction
        δuλ0[1:n] .= u_sol .- cache.u0
        δuλ0[end] = λ_pert

        # Normalize the tangent direction and ensure it is in the positive λ direction
        sf = λ_pert < 0 ? -1.0 : 1.0
        n_inv = sf / norm(δuλ0)
        δuλ0 .*= n_inv

        update_tangent!(cache, δuλ0)        # Update the tangent direction in cache
        cache.δuλ0_initial .= δuλ0          # Save initial tangent
    else
        error(
            "Solve to compute initial tangent failed! Consider reducing perturbation size."
        )
    end
    return nothing
end

function compute_initial_tangent(
    method::UserInitialTangent,
    cache::PALCCache,
    alg::PALC,
    prob::ContinuationProblem,
    solvers,
    trace::AbstractTraceLevel,
)
    # Get user tangent
    n = length(cache.u0)
    initial_tangent = method.initial_tangent
    if length(initial_tangent) != length(cache.u0) + 1
        error("The provided initial tangent must be of length $(n + 1)!")
    end

    # Scale user tangent
    sf = initial_tangent[end] < 0 ? -1.0 : 1.0
    n_inv = sf / norm(initial_tangent)
    initial_tangent .*= n_inv

    # Update tangent and save the initial tangent
    update_tangent!(cache, initial_tangent)
    cache.δuλ0_initial .= initial_tangent
end

function compute_initial_tangent(
    method::BorderedInitialTangent,
    cache::PALCCache,
    alg::PALC,
    prob::ContinuationProblem,
    solvers,
    trace::AbstractTraceLevel,
)
    n = length(cache.u0)
    uλpred = cache.uλpred
    uλpred[1:n] .= cache.u0
    uλpred[end] = cache.λ0

    A = cache.bordered_mat
    δuλ0 = cache.δuλ0
    δuλ0 .= 0.0

    iterates = get_search_order(method, n)
    if maximum(iterates) > n + 1
        error(
            "The user provided base direction search order contains an integer that is " *
            "greater than n + 1!",
        )
    end

    for i in iterates
        # Set the tangent direction
        δuλ0[i] = 1.0
        update_tangent!(cache, δuλ0)

        # Solve PALC nonlinear problem
        uλ, retcode = solve_palc_nlp!(solvers, uλpred, trace)

        # If successful update the tangent direction and return
        if SciMLBase.successful_retcode(retcode)
            # Set bordered matrix
            eval_J!(cache.Jfun, cache.Ffun, uλ, prob.f)
            A[1:n, 1:(n + 1)] .= cache.Jfun
            ddotdu1!(view(A, n + 1, 1:n), cache.δu0, alg.inner_prod)
            A[n + 1, n + 1] = ddotdλ1(cache.δλ0, alg.inner_prod)

            # Solve the linear system
            set_lp_matrix!(solvers, A)
            x = solve_lp!(solvers)

            # Scale the new tangent
            xn = sqrt(alg.inner_prod(view(x, 1:n), x[n + 1]))
            α = x[end] < 0 ? -1.0 / xn : 1.0 / xn
            x .*= α

            # Update the tangent direction
            update_tangent!(cache, x)
            return nothing
        end

        # Null element of the tangent direction
        δuλ0[i] = 0.0
    end

    # If we reach here, then we failed to compute the tangent
    error("Solve to compute initial tangent failed!")

    return nothing
end

function initialize_palc!(
    initial_tangent::AbstractInitialTangent,
    cache::PALCCache,
    alg::PALC,
    p::ContinuationProblem,
    solvers,
    trace::AbstractTraceLevel,
)
    # Get information from the problem
    u0 = p.u0
    λ0 = p.λ0

    # Make sure our current iterate is the initial guess provided by user
    # NOTE: We pass the final argument of false so that this initial point is not
    # added to the points on the zero-curve
    set_successful_iterate!(cache, u0, λ0, false)

    # Set natural continuation parameter
    set_natural_continuation_parameter!(cache, λ0)

    # Print trace if desired
    print_initialization_trace(cache, trace, 1)

    # Solve initial problem with user provided guess
    u_sol, retcode = solve_natural_nlp!(solvers, u0, trace)

    # Update current iterate if solve successful, otherwise error
    if SciMLBase.successful_retcode(retcode)
        set_successful_iterate!(cache, u_sol, λ0, true)
    else
        error("Initial solve failed with user provided guess!")
    end

    # Compute the initial tangent
    compute_initial_tangent(initial_tangent, cache, alg, p, solvers, trace)

    return nothing
end

function print_initialization_trace(cache::PALCCache, trace::Silent, stage::Int)
    return nothing
end
function print_initialization_trace(
    cache::PALCCache, trace::NonSilentTraceLevel, stage::Int
)
    if stage == 1
        println("Initializing PALC Algorithm:")
        println("  Solving with provided guess: λ = $(cache.λn)")
    elseif stage == 2
        println("  Computing initial tangent:   λ = $(cache.λn)")
    end
    return nothing
end
