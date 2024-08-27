def set_global_seed(seed):
    import random as random
    import torch
    import numpy as np

    # Set the random seed for the random, numpy, and torch libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def add_seed_to_hyperparameters(hyperparameters, seed):
    # Define the random seed argument names for each model type
    random_seed_args = {
        'GBM': 'random_state',
        'CAT': 'random_seed',
        'XGB': 'random_state',
        'FASTAI': 'seed',
        'RF': 'random_state',
        'XT': 'random_state',
        'LR': 'random_state'
    }
    
    # New dictionary to store the updated hyperparameters
    updated_hyperparameters = {}

    # Loop through each key in the hyperparameters dictionary
    for model, params in hyperparameters.items():
        seed_arg = random_seed_args.get(model)
        if isinstance(params, list):
            new_params = []
            for param in params:
                new_param = param.copy()
                new_param[seed_arg] = seed
                new_params.append(new_param)
            updated_hyperparameters[model] = new_params
        elif isinstance(params, dict):
            new_param = params.copy()
            new_param[seed_arg] = seed
            updated_hyperparameters[model] = new_param
                
    return updated_hyperparameters