import numpy as np
import wandb
from Q2 import initialize_parameters, forward_propagation
from Q3 import backprop, update_parameters
from Q1 import load_data
from Q7 import get_best_config, SWEEP_ID
from Q1 import WANDB_ENTITY, WANDB_PROJECT


# Define class names for Fashion MNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]



def compute_loss(predictions, targets, loss_type="cross_entropy"):
    
    num_samples = targets.shape[1]
    
    # Return loss based on specified type using ternary-like structure
    if loss_type == "cross_entropy":
        # Added epsilon inside log for numerical stability
        return -np.sum(targets * np.log(predictions + 1e-8)) / num_samples
    elif loss_type == "mean_squared_error":
        # Using element-wise subtraction and power
        squared_diff = np.power(predictions - targets, 2)
        return np.sum(squared_diff) / (2 * num_samples)
    else:
        # More descriptive error message
        valid_types = "'cross_entropy' or 'mean_squared_error'"
        raise ValueError(f"Loss type '{loss_type}' not recognized. Must be {valid_types}.")

def train_with_loss_function(criterion_type, configuration):
    # Start tracking with descriptive group name
    run_id = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, 
                      group=f"Q8_{criterion_type}_loss_comparison")

    # Document configuration parameters
    for param, value in configuration.items():
        wandb.config[param] = value
    
    # Construct a descriptive run name using list comprehension
    run_components = [
        f"hl_{configuration['num_layers']}",
        f"bs_{configuration['batch_size']}",
        f"ac_{configuration['activation']}",
        f"wd_{configuration['weight_decay']}",
        f"lr_{configuration['learning_rate']}",
        f"opt_{configuration['optimizer']}",
        f"wi_{configuration['weight_init']}"
    ]
    wandb.run.name = "_".join(run_components)
    wandb.run.save()

    # Prepare dataset with validation
    data_package = load_data(dataset="fashion_mnist", val_split=0.2)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_package

    # Define network structure
    depths = [784] + [configuration["hidden_size"]] * configuration["num_layers"] + [10]

    # Initialize network parameters with conditional assignment
    if configuration["weight_init"].lower() == 'random':
        network_params = initialize_parameters(depths)
    else:  
        # Xavier initialization with dictionary comprehension
        network_params = {
            f"W{i}": np.random.randn(depths[i], depths[i-1]) * np.sqrt(1 / depths[i-1])
            for i in range(1, len(depths))
        }
        network_params.update({
            f"b{i}": np.zeros((depths[i], 1)) 
            for i in range(1, len(depths))
        })

    opt_memory = {}
    iteration = 0

    sample_count = X_train.shape[1]
    chunk_size = configuration["batch_size"]
    chunks_per_epoch = sample_count // chunk_size

    training_metrics = []
    validation_scores = []

    for epoch_idx in range(configuration["epochs"]):
        # Shuffle dataset each epoch
        shuffle_idx = np.random.permutation(sample_count)
        X_randomized, Y_randomized = X_train[:, shuffle_idx], Y_train[:, shuffle_idx]

        epoch_error = 0
        epoch_correct = 0

        batch_idx = 0
        while batch_idx < chunks_per_epoch:
            start_pos = batch_idx * chunk_size
            end_pos = min((batch_idx + 1) * chunk_size, sample_count)
            
            X_chunk = X_randomized[:, start_pos:end_pos]
            Y_chunk = Y_randomized[:, start_pos:end_pos]

            # Forward pass
            activations, cache = forward_propagation(X_chunk, network_params)

            # Compute batch loss using function
            chunk_loss = compute_loss(activations, Y_chunk, criterion_type)
            epoch_error += chunk_loss

            # Compute accuracy with vectorized operations
            predictions_idx = np.argmax(activations, axis=0)
            truth_idx = np.argmax(Y_chunk, axis=0)
            batch_accuracy = np.mean(predictions_idx == truth_idx)
            epoch_correct += batch_accuracy

            # Gradient computation
            gradients = backprop(Y_chunk, cache, network_params)
            iteration += 1

            # Configure optimizer with dictionary
            optimizer_config = {
                "beta": configuration.get("momentum", 0.9),
                "beta1": configuration.get("beta1", 0.9),
                "beta2": configuration.get("beta2", 0.999),
                "epsilon": configuration.get("epsilon", 1e-8)
            }

            # Update model parameters
            network_params, opt_memory = update_parameters(
                network_params, gradients, 
                optimizer=configuration["optimizer"],
                optimizer_state=opt_memory,
                eta=configuration["learning_rate"],
                t=iteration, hyperparams=optimizer_config
            )
            
            # Move to next batch
            batch_idx += 1

        # Calculate and store training metrics
        training_metrics.append(epoch_error / chunks_per_epoch)

        # Validation metrics calculation
        val_outputs, _ = forward_propagation(X_val, network_params)
        val_pred = np.argmax(val_outputs, axis=0)
        val_true = np.argmax(Y_val, axis=0)
        val_accuracy = np.mean(val_pred == val_true)
        validation_scores.append(val_accuracy)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch_idx,
            f"{criterion_type}_train_loss": epoch_error / chunks_per_epoch,
            f"{criterion_type}_val_accuracy": val_accuracy,
        })

    # Final evaluation and metrics logging
    test_outputs, _ = forward_propagation(X_test, network_params)
    test_pred = np.argmax(test_outputs, axis=0)
    test_true = np.argmax(Y_test, axis=0)
    test_accuracy = np.mean(test_pred == test_true)

    # Log final results to wandb
    wandb.log({
        f"{criterion_type}_test_accuracy": test_accuracy,
        f"{criterion_type}_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=test_true, preds=test_pred, class_names=CLASS_NAMES),
    })

    return training_metrics

def train_best_model_with_losses():
    # Retrieve optimal configuration
    optimal_params = get_best_config()

    # First model: cross-entropy loss
    print("Training model with cross-entropy loss...")
    ce_history = train_with_loss_function("cross_entropy", optimal_params)
    wandb.finish()  # Close the first run
    
    # Second model: mean squared error loss
    print("Training model with mean squared error loss...")
    mse_history = train_with_loss_function("mean_squared_error", optimal_params)
    wandb.finish()  # Close the second run
    
    # Create comparison visualization
    print("Creating comparison visualization...")
    vis_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="loss_comparison_analysis")
    
    # Generate time points for x-axis
    epochs = range(optimal_params["epochs"])
    
    # Plot comparison using wandb visualization tools
    wandb.log({
        "loss_function_comparison": wandb.plot.line_series(
            xs=list(epochs),
            ys=[ce_history, mse_history],
            keys=["Cross Entropy", "Mean Squared Error"],
            title="Training Loss Comparison by Loss Function",
            xname="Training Epoch",
        )
    })
    
    print("Comparison complete! Results available in Weights & Biases dashboard.")
    
if __name__ == "__main__":
    train_best_model_with_losses()
