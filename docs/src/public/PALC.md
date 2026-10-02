# Pseudo-arclength continuation (PALC)
## PALC
This contains information regarding pseudo-arclength continuation. 
```@autodocs
Modules = [SimpleContinuation]
Pages = ["PALC/palc.jl"]
```

# Predictors
```@docs
SimpleContinuation.Bordered
SimpleContinuation.Secant
```
# Tangent Initialization Methods
```@autodocs
Modules = [SimpleContinuation]
Pages = ["PALC/initialization.jl"]
```


# Step Limiters
Methods to limit the length of the correction step with respect to the size of the tangent. Helps to avoid erroneous curve switching.
```@autodocs
Modules = [SimpleContinuation]
Pages = ["PALC/correction.jl"]
```

# Inner Products
Types of inner products. 
```@autodocs
Modules = [SimpleContinuation]
Pages = ["inner_products.jl"]
```

# Trace
Options for REPL printing during continuation runs
```@docs
Silent
ContinuationSteps
ContinuationAndNewtonSteps
```