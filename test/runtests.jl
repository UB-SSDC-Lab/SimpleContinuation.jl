using SimpleContinuation
using Test
using SafeTestsets

@time begin
    @time @safetestset "Functions" begin
        include("test_functions.jl")
        include("test_retcodes.jl")
        include("test_callbacks.jl")
        include("test_example.jl")
    end
end
