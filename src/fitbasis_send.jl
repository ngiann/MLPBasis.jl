############################################################################
function fitbasis(y, σ; K = 3, iterations = 1)
############################################################################

    N, D = size(y); @assert(size(σ) == size(y))

    param = [0.1*randn(D*K); invsoftplus.(ones(K*N))]

    fitbasis(y, σ, param; K = K, iterations = iterations)

end

############################################################################
function fitbasis(y, σ, param; K = 3, iterations = 1)
############################################################################

    N, D = size(y); @assert(size(σ) == size(y))

    makepos(x) = softplus(x)

    @printf("There are %d number of spectra of dimension %d\n", N, D)
    @printf("Learn %d basis functions\n\n", K)
    @printf("Optimising %d number of free parameters\n\n", D*K + K*N)


    #--------------------------------------------
    function unpack(p)
    #--------------------------------------------

        local MARK = 0

        local B = makepos.(reshape(p[MARK+1:MARK+D*K], K, D)); MARK += D*K

        local α = makepos.(reshape(p[MARK+1:MARK+K*N], N, K)); MARK += K*N

        @assert(MARK == length(p))

        return B, α, p

    end


    #--------------------------------------------
    function logl(B, α, p)
    #--------------------------------------------

        return -0.5*sum(abs2, (α*B - y)./σ) - 1e-6*sum(abs2,p)
        
    end


    helper(p) = -logl(unpack(p)...)

    @show helper(param)


    #--------------------------------------------
    # Setup optimisation
    #--------------------------------------------

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 100, g_tol=1e-9)

    result = optimize(helper, param, ConjugateGradient(), opt, autodiff = AutoZygote())


    #--------------------------------------------
    # Instantiate basis
    #--------------------------------------------

    B, α = unpack(result.minimizer)

    
    #--------------------------------------------
    # Fitting 
    #--------------------------------------------

    return  B, α, result.minimizer
    
end