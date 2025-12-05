module MLPBasis

    # using BlackBoxOptim
    using DifferentiationInterface
    using Distributions
    # using ForwardNeuralNetworks
    using LinearAlgebra
    import Mooncake
    using Printf
    using Random
    using StatsFuns
    using Optim
    import Zygote
    import Mooncake
    
    include("fitbasis_send.jl"); export fitbasis, fit

end
