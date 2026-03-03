function fitgp(t, x; iterations = 10)

    # t are the target outputs (here one-dimensional)
    # x are the inputs (here one-dimensional)

    # Initialise hyperparameters
    logℓ, logα = 0.0, 0.0
   
    rbf(x, y, logℓ, logα) = exp(-abs2(x - y)*exp(logℓ))*exp(logα)

    function calculatecovariance!(K, logℓ, logα)
        # Can be done better, but point here it to understand where
        # problems may arise with allocations and performance
        # when writing own code

        N = length(x)

        for i in 1:N
            for j in 1:N
                K[i, j] = rbf(x[i], x[j], logℓ, logα)
            end
        end

    end

    function unpack_parameters(params)
        logℓ, logα = params
        return logℓ, logα
    end

    function marginal_loglikelihood(logℓ, logα)
        K = zeros(length(x), length(x))
        calculatecovariance!(K, logℓ, logα)
        L = cholesky(Symmetric(K + 1e-6I))
        # log marginal likelihood of a Gaussian process for Gaussian likelihood
        return -0.5 * t' * (L \ (L' \ t)) - sum(log.(diag(L))) - length(x)/2*log(2π)
    end

    helper_objective(params) = -marginal_loglikelihood(unpack_parameters(params)...)

    # Optimise hyperparameters using LBFGS
    optimize(helper_objective, [logℓ, logα], LBFGS(); iterations = iterations, autodiff=AutoMooncake())
    
end




