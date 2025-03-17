import numpy as np

def act_hid(x, activation="ReLU"):
    """Our activation function for hidden layers."""
    if activation == "ReLU":
        return np.maximum(0, x)  
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-x))  
    elif activation == "tanh":
        return np.tanh(x)  
    elif activation == "identity":
        return x  
    else:
        raise ValueError("I don't know this activation function.")

def act_out(x):
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    denominator = np.sum(numerator, axis=0, keepdims=True)
    numerator = np.exp(x_shifted)
    return numerator / denominator

def initialize_parameters(layer_dims):
    
    L = len(layer_dims) - 1
    
    weights = {f"W{l}": np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01 
               for l in range(1, L+1)}
    biases = {f"b{l}": np.zeros((layer_dims[l], 1)) 
              for l in range(1, L+1)}
    parameters = {**weights, **biases}
    
    return parameters

def forward_propagation(X, parameters):
    act_preact = {"h0": X}  # Store input as first activation
    L = len(parameters) // 2
    for i in range(1, L+1):
        W = parameters[f"W{i}"]
        b = parameters[f"b{i}"]
        a = np.dot(W, act_preact[f"h{i-1}"]) + b  # Linear part: WX + b
        h = act_out(a) if i == L else act_hid(a)  # Apply activation - softmax for output layer
        act_preact[f"a{i}"] = a  # Save pre-activation
        act_preact[f"h{i}"] = h  # Save activation
    return act_preact[f"h{L}"], act_preact  # Return output and cache
