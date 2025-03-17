import numpy as np

def get_layer_count(parameters):
    return len(parameters) // 2

def init_if_missing(state, key, shape):
    if key not in state:
        state[key] = np.zeros(shape)
    return state[key]

def update_parameters_nadam(parameters, grad_w_b, state, eta, beta1, beta2, epsilon, t):
    layers = list(range(1, get_layer_count(parameters) + 1))
    layers.reverse()
    
    for l in layers:
        w_shape = parameters[f"W{l}"].shape
        b_shape = parameters[f"b{l}"].shape
        
        moment_keys = [
            (f"vW{l}", w_shape), (f"vb{l}", b_shape),
            (f"sW{l}", w_shape), (f"sb{l}", b_shape)
        ]
        for key, shape in moment_keys:
            init_if_missing(state, key, shape)
        
        for param_type in ["W", "b"]:
            state[f"v{param_type}{l}"] = (
                beta1 * state[f"v{param_type}{l}"] + 
                (1-beta1) * grad_w_b[f"d{param_type}{l}"]
            )
        
        for param_type in ["W", "b"]:
            state[f"s{param_type}{l}"] = (
                beta2 * state[f"s{param_type}{l}"] + 
                (1-beta2) * np.square(grad_w_b[f"d{param_type}{l}"])
            )
        
        bias_corrections = {
            "first": 1 - beta1**t,
            "second": 1 - beta2**t
        }
        
        for param_type in ["W", "b"]:
            v_corrected = state[f"v{param_type}{l}"] / bias_corrections["first"]
            s_corrected = state[f"s{param_type}{l}"] / bias_corrections["second"]
            
            next_bias = 1 - beta1**(t+1)
            nesterov_term = (
                beta1 * v_corrected + 
                (1-beta1) * grad_w_b[f"d{param_type}{l}"] / next_bias
            )
            
            denominator = np.sqrt(s_corrected) + epsilon
            parameters[f"{param_type}{l}"] -= eta * nesterov_term / denominator
            
    return parameters, state

def update_parameters_rmsprop(parameters, grad_w_b, state, eta, beta, epsilon=1e-8):
    layer_count = get_layer_count(parameters)
    layer_idx = 1
    
    while layer_idx <= layer_count:
        param_groups = [
            {"param": "W", "grad": "dW"}, 
            {"param": "b", "grad": "db"}
        ]
        
        for pg in param_groups:
            param_key = f"{pg['param']}{layer_idx}"
            grad_key = f"{pg['grad']}{layer_idx}"
            state_key = f"s{param_key}"
            
            current_state = init_if_missing(
                state, state_key, parameters[param_key].shape
            )
            
            grad_squared = np.power(grad_w_b[grad_key], 2)
            new_state = beta * current_state + (1-beta) * grad_squared
            state[state_key] = new_state
            
            denom = np.sqrt(state[state_key]) + epsilon
            update = eta * grad_w_b[grad_key] / denom
            parameters[param_key] -= update
            
        layer_idx += 1
        
    return parameters, state

def update_parameters_nesterov(parameters, grad_w_b, state, eta, beta):
    layer_count = get_layer_count(parameters)
    
    velocities = {}
    param_refs = {}
    
    for l in range(1, layer_count + 1):
        if f"vW{l}" not in state:
            state[f"vW{l}"] = np.zeros_like(parameters[f"W{l}"])
        velocities[f"W{l}"] = state[f"vW{l}"]
        param_refs[f"W{l}"] = parameters[f"W{l}"]
        
        if f"vb{l}" not in state:
            state[f"vb{l}"] = np.zeros_like(parameters[f"b{l}"])
        velocities[f"b{l}"] = state[f"vb{l}"]
        param_refs[f"b{l}"] = parameters[f"b{l}"]
    
    for l in range(1, layer_count + 1):
        for param_type in ["W", "b"]:
            param_key = f"{param_type}{l}"
            v_key = f"v{param_key}"
            
            state[v_key] = beta * velocities[param_key] + eta * grad_w_b[f"d{param_key}"]
    
    for l in range(1, layer_count + 1):
        for param_type in ["W", "b"]:
            param_key = f"{param_type}{l}"
            v_key = f"v{param_key}"
            parameters[param_key] -= state[v_key]
            
    return parameters, state

def update_parameters_momentum(parameters, grad_w_b, state, eta, beta):
    layer_count = get_layer_count(parameters)
    updates = {}
    
    for param_type in ["W", "b"]:
        for l in range(1, layer_count + 1):
            param_key = f"{param_type}{l}"
            v_key = f"v{param_key}"
            grad_key = f"d{param_key}"
            
            if v_key not in state:
                state[v_key] = np.zeros_like(parameters[param_key])
            
            state[v_key] = beta * state[v_key] + eta * grad_w_b[grad_key]
            updates[param_key] = state[v_key]
    
    for param_key, update in updates.items():
        parameters[param_key] -= update
        
    return parameters, state

def update_parameters_sgd(parameters, grad_w_b, eta):
    layer_count = get_layer_count(parameters)
    
    param_keys = []
    for l in range(1, layer_count + 1):
        param_keys.extend([f"W{l}", f"b{l}"])
        
    for key in param_keys:
        grad_key = "d" + key
        
        parameters[key] = parameters[key] - eta * grad_w_b[grad_key]
        
    return parameters

def update_parameters_adam(parameters, grad_w_b, state, eta, beta1, beta2, epsilon, t):
    layer_count = get_layer_count(parameters)
    
    correction_factor1 = 1 - beta1**t
    correction_factor2 = 1 - beta2**t
    
    param_types = ["W", "b"]
    
    for l in range(1, layer_count + 1):
        for p_type in param_types:
            param_key = f"{p_type}{l}"
            grad_key = f"d{param_key}"
            
            v_key = f"v{param_key}"
            s_key = f"s{param_key}"
            
            for state_key in [v_key, s_key]:
                if state_key not in state:
                    state[state_key] = np.zeros_like(parameters[param_key])
            
            gradient = grad_w_b[grad_key]
            
            state[v_key] = beta1 * state[v_key] + (1 - beta1) * gradient
            
            squared_grad = np.multiply(gradient, gradient)
            state[s_key] = beta2 * state[s_key] + (1 - beta2) * squared_grad
            
            v_corrected = state[v_key] / correction_factor1
            s_corrected = state[s_key] / correction_factor2
            
            update = eta * (v_corrected / (np.sqrt(s_corrected) + epsilon))
            
            parameters[param_key] -= update
            
    return parameters, state
