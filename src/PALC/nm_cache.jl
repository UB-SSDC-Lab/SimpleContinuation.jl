
# Storage for Linear and Nonlinear Problems
struct PALCSolverCache{LP,NNLP,PNLP}
    # The linear solver cache
    lp::LP

    # The nonlinear solve caches
    n_nlp::NNLP     # For natural continuation solves
    palc_nlp::PNLP  # For PALC continuation solves

    nlp_iters::Int
    nlp_tol::Float64

    function PALCSolverCache(p::ContinuationProblem, alg::PALC, cache::PALCCache, newton_iter, newton_tol)
        # Create linear problem cache
        lp = construct_lp_cache(alg, cache)

        # Create nonlinear problem caches
        n_nlp, palc_nlp = construct_nlp_caches(
            p, alg, cache, newton_iter, newton_tol,
        )

        new{typeof(lp), typeof(n_nlp), typeof(palc_nlp)}(lp, n_nlp, palc_nlp, newton_iter, newton_tol)
    end
end

# ===== Linear Problem Cache Construction
function construct_lp_cache(alg::PALC{<:Bordered}, cache::PALCCache)
    return init(LinearProblem(cache.bordered_mat, cache.bordered_b), alg.linsolve)
    return nothing
end
function construct_lp_cache(alg::PALC, cache::PALCCache)
    return nothing
end

# ===== Nonlinear Problem Cache Construction
function construct_nlp_caches(p::ContinuationProblem, alg::PALC, cache::PALCCache, newton_iter, newton_tol)
    # Nonlinear problem parameters (define this to avoid needing closures)
    nlp_params  = (p.f, cache)

    # Initialize caches
    n_nlp       = init(
        NonlinearProblem{true}(
            NonlinearFunction{true, SciMLBase.FullSpecialize}(
                (du,u,p) -> eval_f!(du,u,p[2].λn,p[1]);
                jac = (J,u,p) -> eval_Ju!(J,p[2].Ffun,u,p[2].λn,p[1]),
            ),
            cache.u0, nlp_params,
        ),
        alg.nlsolve;
        abstol = newton_tol,
        reltol = newton_tol,
        maxiters = newton_iter,
        termination_condition = alg.termcond,
    )
    palc_nlp    = init(
        NonlinearProblem{true}(
            NonlinearFunction{true, SciMLBase.FullSpecialize}(
                palc_correction_function!; jac = palc_correction_jacobian!),
            cache.uλ0, nlp_params,
        ),
        alg.nlsolve;
        abstol = newton_tol,
        reltol = newton_tol,
        maxiters = newton_iter,
        termination_condition = alg.termcond,
    )

    return n_nlp, palc_nlp
end

# ===== Getter methods 
get_lp_cache(solvers::PALCSolverCache) = solvers.lp
get_natural_nlp_cache(solvers::PALCSolverCache) = solvers.n_nlp
get_palc_nlp_cache(solvers::PALCSolverCache) = solvers.palc_nlp

# ===== Linear Problem Utilities
function solve_lp!(solvers::PALCSolverCache)
    sol = solve!(solvers.lp)
    return sol.u
end
function set_lp_matrix!(solvers::PALCSolverCache, A)
    solvers.lp.A = A
    return nothing
end

# ===== Nonlinear Problem Utilities
function solve_natural_nlp!(solvers::PALCSolverCache, u0, trace)
    # Get parameters that are getting reset when calling reinit!
    reltol = solvers.n_nlp.termination_cache.reltol
    abstol = solvers.n_nlp.termination_cache.abstol
    maxiters = solvers.n_nlp.maxiters

    # Reinitialize the nonlinear problem
    reinit!(
        solvers.n_nlp, u0; 
        maxiters = maxiters,
        abstol = abstol, 
        reltol = reltol,
    )

    # Solve
    for i in 1:solvers.nlp_iters
        # Take newton step
        step!(solvers.n_nlp)

        # Print trace if desired
        print_natural_solve_trace(solvers, trace)

        if !NonlinearSolve.not_terminated(solvers.n_nlp)
            break
        end
    end
    return NonlinearSolve.get_u(solvers.n_nlp), solvers.n_nlp.retcode
end
function solve_palc_nlp!(solvers::PALCSolverCache, uλ0, trace)
    # Get parameters that are getting reset when calling reinit!
    reltol = solvers.n_nlp.termination_cache.reltol
    abstol = solvers.n_nlp.termination_cache.abstol
    maxiters = solvers.n_nlp.maxiters

    # Reinitialize the nonlinear problem
    reinit!(
        solvers.palc_nlp, uλ0; 
        maxiters = maxiters,
        abstol = abstol, 
        reltol = reltol,
    )

    # Solve
    for i in 1:solvers.nlp_iters
        # Take newton step
        step!(solvers.palc_nlp)

        # Print trace if desired
        print_palc_solve_trace(solvers, trace)

        if !NonlinearSolve.not_terminated(solvers.palc_nlp)
            break
        end
    end
    return NonlinearSolve.get_u(solvers.palc_nlp), solvers.palc_nlp.retcode
end

function print_natural_solve_trace(solvers::PALCSolverCache, trace)
    return nothing
end
function print_natural_solve_trace(solvers::PALCSolverCache, trace::ContinuationAndNewtonSteps)
    nsteps = solvers.n_nlp.stats.nsteps
    fnorm  = norm(solvers.n_nlp.fu)
    println("  $nsteps\t$fnorm")
    return nothing
end

function print_palc_solve_trace(solvers::PALCSolverCache, trace)
    return nothing
end
function print_palc_solve_trace(solvers::PALCSolverCache, trace::ContinuationAndNewtonSteps)
    nsteps = solvers.palc_nlp.stats.nsteps
    fnorm  = norm(solvers.palc_nlp.fu)
    println("  $nsteps\t$fnorm")
    return nothing
end