import wandb
import numpy as np
from Q1 import load_data
from Q2 import initialize_parameters, forward_propagation
from Q3 import backprop, update_parameters
from Q1 import WANDB_ENTITY, WANDB_PROJECT

def train_mnist(config):
    
    # Initialize wandb with descriptive group name
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config, group="Q10_mnist_comparison")
    config = wandb.config
    
    # Create descriptive run name using components pattern
    run_components = [
        f"hl_{config.num_layers}",
        f"bs_{config.batch_size}", 
        f"ac_{config.activation}",
        f"wd_{getattr(config, 'weight_decay', 0)}",
        f"lr_{config.learning_rate}",
        f"opt_{config.optimizer}",
        f"wi_{config.weight_init}"
    ]
    wandb.run.name = "_".join(run_components)
    wandb.run.save()

    # Load MNIST data with validation split
    try:
        data_package = load_data("mnist", val_split=0.2)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_package
        print(f"Data loaded: {X_train.shape[1]} training, {X_val.shape[1]} validation, {X_test.shape[1]} test samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Define network architecture
    network_dims = [784] + [config.hidden_size] * config.num_layers + [10]
    print(f"Network architecture: {network_dims}")

    try:
        if config.weight_init.lower() == "xavier":
            parameters = {}
            for l in range(1, len(network_dims)):
                # Xavier initialization scales weights by sqrt(1/fan_in)
                scale_factor = np.sqrt(1 / network_dims[l-1])
                parameters[f"W{l}"] = np.random.randn(network_dims[l], network_dims[l-1]) * scale_factor
                parameters[f"b{l}"] = np.zeros((network_dims[l], 1))
        else:
            # Default to random initialization
            parameters = initialize_parameters(network_dims)
        
        print(f"Parameters initialized using {config.weight_init} method")
    except Exception as e:
        print(f"Error during parameter initialization: {e}")
        return

    # Setup training state
    optimizer_state = {}
    iteration = 0
    total_samples = X_train.shape[1]
    batch_size = config.batch_size
    batches_per_epoch = total_samples // batch_size
    
    # Training loop
    for epoch in range(config.epochs):
        # Shuffle data at the beginning of each epoch
        permutation = np.random.permutation(total_samples)
        X_shuffled = X_train[:, permutation]
        Y_shuffled = Y_train[:, permutation]
        
        # Track epoch metrics
        epoch_loss = 0
        correct_predictions = 0
        
        # Process each mini-batch
        for batch_idx in range(batches_per_epoch):
            # Get current batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            X_batch = X_shuffled[:, start_idx:end_idx]
            Y_batch = Y_shuffled[:, start_idx:end_idx]
            current_batch_size = X_batch.shape[1]
            
            # Forward propagation
            predictions, cache = forward_propagation(X_batch, parameters)
            
            # Compute loss based on specified loss function
            if config.loss == "cross_entropy":
                batch_loss = -np.sum(Y_batch * np.log(predictions + 1e-8)) / current_batch_size
            else:  # mean_squared_error
                batch_loss = np.sum((predictions - Y_batch) ** 2) / (2 * current_batch_size)
            
            # Add L2 regularization if weight_decay > 0
            weight_decay = getattr(config, "weight_decay", 0)
            if weight_decay > 0:
                l2_penalty = sum(np.sum(np.square(parameters[f"W{l}"])) 
                               for l in range(1, len(network_dims)))
                batch_loss += (weight_decay / (2 * current_batch_size)) * l2_penalty
                
            epoch_loss += batch_loss * current_batch_size
            
            # Calculate accuracy
            predicted_classes = np.argmax(predictions, axis=0)
            true_classes = np.argmax(Y_batch, axis=0)
            correct_predictions += np.sum(predicted_classes == true_classes)
            
            # Backpropagation
            gradients = backprop(Y_batch, cache, parameters)
            iteration += 1
            
            # Configure optimizer hyperparameters
            hyperparams = {
                "beta": getattr(config, "momentum", 0.9),
                "beta1": getattr(config, "beta1", 0.9),
                "beta2": getattr(config, "beta2", 0.999),
                "epsilon": getattr(config, "epsilon", 1e-8)
            }
            
            # Update parameters
            parameters, optimizer_state = update_parameters(
                parameters, gradients, 
                optimizer=config.optimizer,
                optimizer_state=optimizer_state,
                eta=config.learning_rate, 
                t=iteration,
                hyperparams=hyperparams
            )
        
        # Calculate epoch metrics
        train_loss = epoch_loss / total_samples
        train_accuracy = correct_predictions / total_samples
        
        # Validate model performance
        val_predictions, _ = forward_propagation(X_val, parameters)
        val_predicted_classes = np.argmax(val_predictions, axis=0)
        val_true_classes = np.argmax(Y_val, axis=0)
        val_accuracy = np.mean(val_predicted_classes == val_true_classes)
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })
        
        print(f"Epoch {epoch+1}/{config.epochs}: "
              f"Loss={train_loss:.4f}, "
              f"Train Acc={train_accuracy:.4f}, "
              f"Val Acc={val_accuracy:.4f}")
    
    # Final evaluation
    test_predictions, _ = forward_propagation(X_test, parameters)
    test_predicted_classes = np.argmax(test_predictions, axis=0)
    test_true_classes = np.argmax(Y_test, axis=0)
    test_accuracy = np.mean(test_predicted_classes == test_true_classes)
    
    # Log final metrics and confusion matrix
    wandb.log({
        "test_accuracy": test_accuracy,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_true_classes,
            preds=test_predicted_classes,
            class_names=[str(i) for i in range(10)]  # MNIST class names are digits 0-9
        )
    })
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

if __name__ == "__main__":
    # Configuration examples - now with all required hyperparameters
    
    configs_to_try = [
        {
            "num_layers": 2, 
            "hidden_size": 128, 
            "activation": "ReLU", 
            "optimizer": "adam", 
            "learning_rate": 0.001, 
            "weight_init": "Xavier", 
            "loss": "cross_entropy", 
            "epochs": 10, 
            "batch_size": 32,
            "momentum": 0.9,  # Added missing hyperparameters
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
        
        {
            "num_layers": 3, 
            "hidden_size": 64, 
            "activation": "tanh", 
            "optimizer": "rmsprop", 
            "learning_rate": 0.0005, 
            "weight_init": "random", 
            "loss": "cross_entropy", 
            "epochs": 10, 
            "batch_size": 32,
            "momentum": 0.9,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
        
        {
            "num_layers": 1, 
            "hidden_size": 256, 
            "activation": "sigmoid", 
            "optimizer": "momentum", 
            "learning_rate": 0.01, 
            "weight_init": "Xavier", 
            "loss": "mean_squared_error", 
            "epochs": 10,
            "batch_size": 32,
            "momentum": 0.9,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        }
    ]
    
    # Run each configuration
    for config in configs_to_try:
        print(f"Training with configuration: {config}")
        train_mnist(config)
