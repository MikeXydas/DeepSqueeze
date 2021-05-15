from bayes_opt import BayesianOptimization


def minimize_comp_ratio(func, params):
    # The constant parameters that are not optimized but our func needs
    const_params = {
        "data_path": params["data_path"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "lr": params["lr"],
        "error_threshold": params["error_threshold"],
        "compression_path": params["compression_path"],
        "binning_strategy": params["binning_strategy"]
    }

    # The func parameters we optimize
    optimized_params = {
        "ae_depth": params["ae_depth"],
        "width_multiplier": params["width_multiplier"],
        "code_size": params["code_size"]
    }

    def param_wrapper(ae_depth, width_multiplier, code_size):
        # Through picked_params we also perform discretization if needed
        # and any other transformations
        picked_params = {
            "ae_depth": int(ae_depth),
            "width_multiplier": int(width_multiplier),
            "code_size": int(code_size)
        }

        # Union both the constant and the optimized parameters and pass them on our
        # compression pipeline
        union_params = dict(const_params, **picked_params)

        # Since bayesian optimization maximizes a function we return the negative compression ratio
        return -func(union_params)[0]

    optimizer = BayesianOptimization(
        f=param_wrapper,
        pbounds=optimized_params
    )

    # Guide the optimizer by suggesting values that are empirically correct as initial point
    optimizer.probe(
        params={"ae_depth": 2, "width_multiplier": 2, "code_size": 1},
        lazy=True,
    )

    # Start the maximization of -1 * compression_ratio
    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )

    return optimizer.max
