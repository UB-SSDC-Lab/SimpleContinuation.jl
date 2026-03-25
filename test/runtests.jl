using SimpleContinuation
using Test
using SafeTestsets

@time begin
    @time @safetestset "Functions" begin
        include("test_functions.jl")
        include("test_retcodes.jl")
    end
end
