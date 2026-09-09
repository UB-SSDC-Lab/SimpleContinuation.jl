
using Test, SimpleContinuation

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

f_min_time = (F, u, s) -> scalar_test_fun!(F, u, s)
Jz_min_time = (J, F, u, s) -> scalar_fun_ujac!(J, F, u, s)
J_min_time = (J, F, u, s) -> scalar_fun_jac!(J, F, u, s)

u0 = [-2.0]
λ0 = -1.0

prob = ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
)
alg = PALC(; predicter=Bordered())
cache = continuation(
    prob,
    alg;
    both_sides=false,
    ds0=1e-2,
    dsmin=1e-2,
    dsmax=0.1,
    max_cont_steps=1000,
    detection_callback=FoldBifurcationDetectionCallback(),
    #term_callback = FoldBifurcationTerminationCallback(),
    save_verbose=true
)

# A few tests to make sure everything works as expected
@test length(cache.detected_points)==2
lt = length(cache.tangents)
@test lt==length(cache.br)-1 # should have one fewer tangent, since alg ends after correction
@test lt==length(cache.dss)
@test lt==length(cache.predictions)

cache = continuation(
    prob,
    alg;
    both_sides=false,
    ds0=1e-2,
    dsmin=1e-2,
    dsmax=0.1,
    max_cont_steps=1000,
    detection_callback=FoldBifurcationDetectionCallback(),
    term_callback = FoldBifurcationTerminationCallback(),
)
ld = length(cache.detected_points)
@test ld == 1
det = cache.detected_points[1]
dp = (det[1], det[2])
@test isapprox(dp[1], cache.br[end][1])
@test isapprox(dp[2], cache.br[end][2])