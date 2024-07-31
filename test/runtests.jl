using SimpleContinuation
using Test
using SafeTestsets

@time begin
    @time @safetestset "Functions" begin
        include("test_functions.jl")
    end
end