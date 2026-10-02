
# Verbosity levels
abstract type AbstractTraceLevel end
abstract type NonSilentTraceLevel <: AbstractTraceLevel end

"""
    Silent <: AbstractTraceLevel

No output is printed to the console during continuation. This is the default for PALC.
"""
struct Silent <: AbstractTraceLevel end

"""
    ContinuationSteps <: AbstractTraceLevel
    
Prints basic information at the end of each step.
"""
struct ContinuationSteps <: NonSilentTraceLevel end

"""
    ContinuationAndNewtonSteps <: AbstractTraceLevel

Prints the same information as `ContinuationSteps`, but also prints information about the Newton iterations at each step.
"""
struct ContinuationAndNewtonSteps <: NonSilentTraceLevel end

# Continuation predictor types
abstract type AbstractPredictor end

"""
    Bordered <: AbstractPredictor

Bordered prediction method. Uses the bordered matrix to compute predicted updates. This is the default for PALC.
"""
struct Bordered <: AbstractPredictor end

"""
    Secant <: AbstractPredictor

Secant prediction method. Uses previous values of iterates to compute predicted updates. 
"""
struct Secant <: AbstractPredictor end
