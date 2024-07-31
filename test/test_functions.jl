using SimpleContinuation
using Test

const SC = SimpleContinuation

function test_fun(du, u, λ)
    du[1] = u[1]^2 + u[1]*λ
    du[2] = u[2]^2 + u[2]*λ
    return nothing
end
function test_Ju(Ju, Fu, u, λ)
    Ju[1,1] = 2*u[1] + λ
    Ju[1,2] = 0
    Ju[2,1] = 0
    Ju[2,2] = 2*u[2] + λ
    return nothing
end
function test_Jλ(Jλ, Fu, u, λ)
    Jλ[1] = u[1]
    Jλ[2] = u[2]
    return nothing
end
function test_J(J, Fu, u, λ)
    J[1,1] = 2*u[1] + λ
    J[1,2] = 0
    J[1,3] = u[1]
    J[2,1] = 0
    J[2,2] = 2*u[2] + λ
    J[2,3] = u[2]
    return nothing
end

# ==== Truth
u   = rand(2)
λ   = rand()
uλ  = [u; λ]
Ft  = zeros(2)
cF  = similar(Ft)
Jut = zeros(2,2)
Jλt = zeros(2)
Jt  = zeros(2,3)

test_fun(Ft, u, λ)
test_Ju(Jut, cF, u, λ)
test_Jλ(Jλt, cF, u, λ)
test_J(Jt, cF, u, λ)

# ==== Test Ju and Jλ provided
fun1 = ContinuationFunction(test_fun, test_Ju, test_Jλ)
F1 = zeros(2); Ju1 = zeros(2,2); Jλ1 = zeros(2); J1 = zeros(2,3)

# F evaluation
SC.eval_f!(F1, u, λ, fun1)
@test all(F1 .≈ Ft)

SC.eval_f!(F1, uλ, fun1)
@test all(F1 .≈ Ft)

# Ju evaluation
SC.eval_Ju!(Ju1, F1, u, λ, fun1)
@test all(Ju1 .≈ Jut)

SC.eval_Ju!(Ju1, F1, uλ, fun1)
@test all(Ju1 .≈ Jut)

# J evaluation
SC.eval_J!(J1, F1, u, λ, fun1)
@test all(J1 .≈ Jt)

SC.eval_J!(J1, F1, uλ, fun1)
@test all(J1 .≈ Jt)

# ==== Test only J provided
fun2 = ContinuationFunction(test_fun, test_J)
F2 = zeros(2); Ju2 = zeros(2,2); Jλ2 = zeros(2); J2 = zeros(2,3)

# F evaluation
SC.eval_f!(F2, u, λ, fun2)
@test all(F2 .≈ Ft)

SC.eval_f!(F2, uλ, fun2)
@test all(F2 .≈ Ft)

# Ju evaluation
SC.eval_Ju!(Ju2, F2, u, λ, fun2)
@test all(Ju2 .≈ Jut)

SC.eval_Ju!(Ju2, F2, uλ, fun2)
@test all(Ju2 .≈ Jut)

# J evaluation
SC.eval_J!(J2, F2, u, λ, fun2)
@test all(J2 .≈ Jt)

SC.eval_J!(J2, F2, uλ, fun2)
@test all(J2 .≈ Jt)

# ==== Test Ju and J provided (no Jλ)
fun3 = ContinuationFunction{Val{true}}(test_fun, test_Ju, test_J)
F3 = zeros(2); Ju3 = zeros(2,2); Jλ3 = zeros(2); J3 = zeros(2,3)

# F evaluation
SC.eval_f!(F3, u, λ, fun3)
@test all(F3 .≈ Ft)

SC.eval_f!(F3, uλ, fun3)
@test all(F3 .≈ Ft)

# Ju evaluation
SC.eval_Ju!(Ju3, F3, u, λ, fun3)
@test all(Ju3 .≈ Jut)

SC.eval_Ju!(Ju3, F3, uλ, fun3)
@test all(Ju3 .≈ Jut)

# J evaluation
SC.eval_J!(J3, F3, u, λ, fun3)
@test all(J3 .≈ Jt)

SC.eval_J!(J3, F3, uλ, fun3)
@test all(J3 .≈ Jt)