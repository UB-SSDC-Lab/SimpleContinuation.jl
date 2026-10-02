# Callbacks
Functions used to detect or terminate at specific points which meet certain conditions. There are several classes of callbacks. Analysis callbacks are simple, and have no root finding capability, but allow you to interact with the `cache` throughout the continuation process. Termination and Detection callbacks use a regula-falsi method to find precise points where they equal zero, terminating at or saving the found point, respectively.
```@autodocs
Modules = [SimpleContinuation]
Pages = ["PALC/callback.jl", "PALC/special_callbacks.jl"]
```