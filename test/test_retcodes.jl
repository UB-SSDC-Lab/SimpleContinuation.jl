using SimpleContinuation
using Test

const SC = SimpleContinuation

function scalar_test_fun!(F, x, λ)
    F[1] = (λ + x[1] - x[1]^3 / 3)
end

function scalar_fun_ujac!(J, F, x, λ)
    scalar_test_fun!(F, x, λ)
    J[1,1] = 1 - x[1]^2
end

function scalar_fun_jac!(J, F, x, λ)
    scalar_test_fun!(F, x, λ)
    J[1,1] = 1 - x[1]^2
    J[1,2] = 1
end

u0 = [-2.]
λ0 = -1.

f_min_time = (F, u, s) -> scalar_test_fun!(F, u, s)
Jz_min_time = (J, F, u, s) -> scalar_fun_ujac!(J, F, u, s)
J_min_time = (J, F, u, s) -> scalar_fun_jac!(J, F, u, s)

# Should term due to callback
cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=0.01,
    dsmin=1e-1,
    dsmax=0.01,
    max_cont_steps=10e3,
    #trace=ContinuationSteps(),
    term_callback=FoldBifurcationTerminationCallback(),
)
@test cache.ret == :Callback

# Should term due to hitting boundary
cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=0.01,
    dsmin=1e-1,
    dsmax=0.01,
    max_cont_steps=10e3,
    #trace=ContinuationSteps(),
)
@test cache.ret == :HitBound

# Should term due to maxiters
cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=0.01,
    dsmin=1e-1,
    dsmax=0.01,
    max_cont_steps=10,
    #trace=ContinuationSteps(),
)
@test cache.ret == :Maxiters

# and stepsize
cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=1.,
    dsmin=1.,
    dsmax=1.,
    max_cont_steps=10,
    trace=ContinuationSteps(),
)
@test cache.ret == :MinimumStepSize