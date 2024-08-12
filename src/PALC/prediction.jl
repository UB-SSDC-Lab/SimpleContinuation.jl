
# ===== Bordered Predictor
function palc_prediction!(cache::PALCCache, alg::PALC{Bordered}, p::ContinuationProblem, solvers, trace)
    # Fill bordered matrix
    set_boardered_matrix!(cache, p)

    # Update lp cache with new matrix
    set_lp_matrix!(solvers, cache.bordered_mat)

    # Solve the linear problem
    x = solve_lp!(solvers)

    # Set the new tangent, making sure we keep moving in the same direction
    scale_predicted_tangent!(x, cache)
    update_tangent!(cache, x)

    # Print trace if desired
    print_prediction_trace(cache, trace)

    return nothing
end

function scale_predicted_tangent!(x, cache::PALCCache)
    α  = 1.0 / norm(x)
    α *= sign(dot(x, cache.δuλ0))
    x *= α
    return nothing
end

function set_boardered_matrix!(cache::PALCCache, p::ContinuationProblem)
    # Get parameters
    fun     = p.f
    θ       = cache.θ
    u0      = cache.u0
    λ0      = cache.λ0
    uλ0     = cache.uλ0
    F       = cache.Ffun
    J       = cache.Jfun
    A       = cache.bordered_mat

    # Evaluate the function jacobian and set in bm
    n = length(uλ0) - 1
    eval_J!(J, F, uλ0, fun)
    A[1:n, 1:n+1] .= J

    # Remaining
    sf1 = 1.0   #θ / n
    sf2 = 1.0   #1.0 - θ
    #A[n+1, 1:n] .= sf1.*u0
    #A[n+1, n+1] = sf2*λ0
    A[n+1, :] .= cache.δuλ0

    return nothing
end

# Just doing nothing for now in all caes as I'm not really sure if there's any relevant 
# information to print here (we're already going to print the predicted update to λ
# in the correction step)
function print_prediction_trace(cache::PALCCache, trace::Silent)
    return nothing
end
function print_prediction_trace(cache::PALCCache, trace::NonSilentTraceLevel)
    println("Computed boardered prediction.")
    return nothing
end

