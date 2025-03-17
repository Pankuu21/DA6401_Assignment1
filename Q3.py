import numpy as np
from optimizers import update_parameters_nadam, update_parameters_rmsprop, update_parameters_nesterov, update_parameters_adam, update_parameters_sgd, update_parameters_momentum
from Q1 import WANDB_ENTITY, WANDB_PROJECT

def update_parameters(parameters, grads, optimizer="sgd", optimizer_state=None,
                      eta=0.01, t=1, hyperparams=None):
    
    # Initialize dictionaries if None
    optimizer_state = {} if optimizer_state is None else optimizer_state
    hyperparams = {} if hyperparams is None else hyperparams
    
    # Define a dictionary mapping optimizer names to their update functions
    optimizer_functions = {
        "sgd": lambda p, g, s, e, h: (update_parameters_sgd(p, g, e), s),
        
        "momentum": lambda p, g, s, e, h: update_parameters_momentum(
            p, g, s, e, h.get("beta", 0.9)),
            
        "nag": lambda p, g, s, e, h: update_parameters_nesterov(
            p, g, s, e, h.get("beta", 0.9)),
            
        "rmsprop": lambda p, g, s, e, h: update_parameters_rmsprop(
            p, g, s, e, h.get("beta", 0.9), h.get("epsilon", 1e-8)),
            
        "adam": lambda p, g, s, e, h: update_parameters_adam(
            p, g, s, e, h.get("beta1", 0.9), h.get("beta2", 0.999), 
            h.get("epsilon", 1e-8), t),
            
        "nadam": lambda p, g, s, e, h: update_parameters_nadam(
            p, g, s, e, h.get("beta1", 0.9), h.get("beta2", 0.999),
            h.get("epsilon", 1e-8), t)
    }
    
    update_function = optimizer_functions.get(optimizer.lower())
    if update_function is None:
        raise ValueError(f"Unknown Optimizer: {optimizer}")
        
    return update_function(parameters, grads, optimizer_state, eta, hyperparams)


def der_hid(a):
    return (a > 0).astype(float)


def backprop(Y,act_preact,parameters):
    grad_w_b={}
    L=len(parameters)//2
    m=Y.shape[1]
    dZ=act_preact[f"h{L}"]-Y
    grad_w_b[f"dW{L}"]=(1/m)*np.dot(dZ,act_preact[f"h{L-1}"].T)
    grad_w_b[f"db{L}"]=(1/m)*np.sum(dZ,axis=1,keepdims=True)

    for i in range(L-1,0,-1):
        dA=np.dot(parameters[f"W{i+1}"].T,dZ)

        dZ=dA*der_hid(act_preact[f"a{i}"])
        grad_w_b[f"dW{i}"] = (1/m) * np.dot(dZ, act_preact[f"h{i-1}"].T)
        grad_w_b[f"db{i}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    return grad_w_b
