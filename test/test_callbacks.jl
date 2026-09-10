using SimpleContinuation
using Test

const SC = SimpleContinuation

# The scalar test function is from https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/gettingstarted/

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


f1 = (x,λ) -> -1.5-x[1] # this should trigger BEFORE first fold bifurcation
f2 = (x,λ) -> x[1] # this would trigger AFTER first fold bifurcation
f3 = (x,λ) -> x[1] - 100 # this should never trigger
f4 = (x,λ) -> λ

fold_bifurcation_cb = SC.FoldBifurcationTerminationCallback()
cb1 = SC.TerminateContinuationCallback(f1)
cb2 = SC.TerminateContinuationCallback(f2)
cb3 = SC.TerminateContinuationCallback(f3)

# now, lets test a couple sets

u0 = [-2.0]
λ0 = -1.0

# ===== Detection Callbacks
cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=1e-2,
    dsmin=1e-2,
    dsmax=0.1,
    max_cont_steps=1000,
    detection_callback=FoldBifurcationDetectionCallback(),
)
#parse and test detected points
p1 = [cache.detected_points[1][1][1], cache.detected_points[1][2]]
p2 = [cache.detected_points[2][1][1], cache.detected_points[2][2]]
@test isapprox(p1, [-1.0, 2/3])
@test isapprox(p2, [1.0, -2/3])

cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=1e-2,
    dsmin=1e-2,
    dsmax=0.1,
    max_cont_steps=1000,
    detection_callback=ContinuationDetectionCallback(f4), # checking for y-axis crossings
    #trace=ContinuationAndNewtonSteps()
)
@test length(cache.detected_points)==3
@test cache.ret==:HitBound # make sure we didn't erroneously stop somewhere else


# ========== Callback Sets
# Set 1: should terminate at fold and have retcode :Callback1
set_1 = SC.TerminateContinuationCallbackSet(fold_bifurcation_cb, cb2) 
cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=1e-2,
    dsmin=1e-2,
    dsmax=0.1,
    max_cont_steps=1000,
    term_callback=set_1,
)
@test cache.ret == :Callback1

# set 2 should have retcode :Callback2
set_2 = SC.TerminateContinuationCallbackSet(cb3, cb2) 
cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=1e-2,
    dsmin=1e-2,
    dsmax=0.1,
    max_cont_steps=1000,
    term_callback=set_2,
)
@test cache.ret == :Callback2

# test detection with a callback set
set_1 = SC.TerminateContinuationCallbackSet(fold_bifurcation_cb, cb2) 
cache = continuation(
    ContinuationProblem(
        ContinuationFunction{Val{true}}(f_min_time, Jz_min_time, J_min_time),
        u0,
        λ0,
        (λ0, 1.0),
    ),
    PALC(; predicter=Bordered());
    both_sides=false,
    ds0=1e-2,
    dsmin=1e-2,
    dsmax=0.1,
    max_cont_steps=1000,
    term_callback=set_1,
    detection_callback = FoldBifurcationDetectionCallback()
)
p1 = [cache.detected_points[1][1][1], cache.detected_points[1][2]]
@test isapprox(p1, [-1.0, 2/3])
@test cache.ret == :Callback1