function fitgp_1(t, x; iterations = 10, JITTER = 1e-6)

    # t are the target outputs (here one-dimensional)
    # x are the inputs (here one-dimensional)

    # Initialise hyperparameters
    logℓ, logα, logσ = 0.0, 0.0, 0.0
   
    rbf(x, y, logℓ, logα) = exp(-abs2(x - y)*exp(logℓ))*exp(logα)

    function calculatecovariance!(K, logℓ, logα)
        # Can be done better, but point here it to understand where
        # problems may arise with allocations and performance
        # when writing own code

        local N = length(x)

        for i in 1:N
            for j in 1:N
                K[i, j] = rbf(x[i], x[j], logℓ, logα)
            end
        end

    end

    function unpack_parameters(params)
        local logℓ, logα, logσ = params
        return logℓ, logα, logσ
    end

    function marginal_loglikelihood_MvNormal(logℓ, logα, logσ)

        # calculate the covariance matrix for the passed hyperparameters
        local K = zeros(length(x), length(x))
        calculatecovariance!(K, logℓ, logα)

        # add noise variance to the diagonal and jitter for numerical stability   
        K += exp(logσ) * I + JITTER * I 

        # return log marginal likelihood of a Gaussian process for Gaussian likelihood
        return logpdf(MvNormal(zeros(length(x)), K), t)

    end


    function marginal_loglikelihood_explicit(logℓ, logα, logσ)

        # calculate the covariance matrix for the passed hyperparameters
        local K = zeros(length(x), length(x))
        calculatecovariance!(K, logℓ, logα)

        # add noise variance to the diagonal and jitter for numerical stability   
        K += exp(logσ) * I + JITTER * I 

        # return log marginal likelihood of a Gaussian process for Gaussian likelihood
        local L = cholesky(Symmetric(K)).L
        return -0.5 * sum(abs2, L \ t) - sum(log, diag(L)) - 0.5*length(x)*log(2π)
    end

    # define negative log-likelihood for optimization
    nll_MvNormal(params) = -marginal_loglikelihood_MvNormal(unpack_parameters(params)...)
    nll_explicit(params) = -marginal_loglikelihood_explicit(unpack_parameters(params)...)

    # Sanity check:
    # show that the two implementations of the marginal log-likelihood 
    # differ only very slightly for the same hyperparameters.
    let
        initparam = randn(3)
        @show nll_MvNormal(initparam)
        @show nll_explicit(initparam)
    end

    # Optimise hyperparameters using LBFGS
    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    # optimise using the MvNormal implementation
    optimize(nll_MvNormal, [logℓ, logα, logσ], LBFGS(), opt, autodiff=Mooncake.AutoMooncake())

    # optimise using the explicit implementation
    optimize(nll_explicit, [logℓ, logα, logσ], LBFGS(), opt, autodiff=Mooncake.AutoMooncake())
    
end





