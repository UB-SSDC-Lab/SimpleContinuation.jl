
# Verbosity levels
abstract type AbstractTraceLevel end
abstract type NonSilentTraceLevel <: AbstractTraceLevel end

struct Silent <: AbstractTraceLevel end
struct ContinuationSteps <: NonSilentTraceLevel end
struct ContinuationAndNewtonSteps <: NonSilentTraceLevel end

# Continuation predictor types
abstract type AbstractPredictor end
struct Bordered <: AbstractPredictor end
struct Secant <: AbstractPredictor end