import numpy as np
import wandb
from Q2 import initialize_parameters, forward_propagation  # Fixed import
from Q3 import backprop, update_parameters
from Q1 import load_data, CLASS_NAMES, WANDB_ENTITY, WANDB_PROJECT

SWEEP_ID = "mp5cpxc6"#baysean gave best config

def get_best_config():
    api = wandb.Api()
    sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{SWEEP_ID}")
    best_run = sweep.best_run(order="val_accuracy")  # Order by validation accuracy
    
    return best_run.config

def train_best_model():
    """
    Train the neural network using the best configuration from the sweep.
    """
    experiment = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group="Q7_best_model")

    # Retrieve optimal hyperparameters
    optimal_config = get_best_config()

    # Document configuration in wandb
    wandb.config.update(optimal_config)

    # Construct descriptive run name from config values
    run_identifier = "_".join([
        f"hl_{optimal_config['num_layers']}",
        f"bs_{optimal_config['batch_size']}",
        f"ac_{optimal_config['activation']}",
        f"wd_{optimal_config['weight_decay']}",
        f"lr_{optimal_config['learning_rate']}",
        f"opt_{optimal_config['optimizer']}",
        f"wi_{optimal_config['weight_init']}"
    ])
    wandb.run.name = run_identifier
    wandb.run.save()

    # Prepare training data with validation split
    data_splits = load_data(dataset="fashion_mnist", val_split=0.1)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_splits

    # Construct network architecture - input, hidden layers, output
    hidden_units = optimal_config.get("hidden_size", 128)
    network_structure = [784] + [hidden_units] * optimal_config.get("num_layers", 1) + [10]

    # Set up model parameters with chosen initialization strategy
    if optimal_config.get("weight_init", "").lower() == "xavier":
        # Xavier initialization - scale by sqrt(1/fan_in)
        model_params = {}
        for i in range(1, len(network_structure)):
            fan_in = network_structure[i-1]
            fan_out = network_structure[i]
            scale_factor = np.sqrt(1/fan_in)
            model_params[f"W{i}"] = np.random.randn(fan_out, fan_in) * scale_factor
            model_params[f"b{i}"] = np.zeros((fan_out, 1))
    else:
        # Default random initialization
        model_params = initialize_parameters(network_structure)

    # Training state variables
    opt_state = {}     # Optimizer's memory
    step_count = 0     # Iteration counter for optimizers like Adam
    examples_count = X_train.shape[1]
    mini_batch_size = optimal_config.get("batch_size", 32)
    batches_per_epoch = examples_count // mini_batch_size

    # Main training loop
    for training_epoch in range(optimal_config.get("epochs", 10)):
        # Shuffle data for each epoch
        shuffle_indices = np.random.permutation(examples_count)
        X_shuffled = X_train[:, shuffle_indices] 
        Y_shuffled = Y_train[:, shuffle_indices]

        # Track epoch statistics
        accumulated_loss = 0
        correct_count = 0

        # Process mini-batches
        for batch_idx in range(batches_per_epoch):
            # Extract current batch
            batch_start = batch_idx * mini_batch_size
            batch_end = min((batch_idx + 1) * mini_batch_size, examples_count)
            X_current = X_shuffled[:, batch_start:batch_end]
            Y_current = Y_shuffled[:, batch_start:batch_end]
            
            predictions, activation_cache = forward_propagation(X_current, model_params)
            
            # Compute cross-entropy loss for this batch
            batch_samples = Y_current.shape[1]
            batch_loss = -np.sum(Y_current * np.log(predictions + 1e-8)) / batch_samples
            
            # Add regularization if configured
            weight_decay = optimal_config.get("weight_decay", 0)
            if weight_decay > 0:
                # L2 regularization term
                l2_penalty = 0
                for l in range(1, len(network_structure)):
                    l2_penalty += np.sum(np.square(model_params[f"W{l}"]))
                batch_loss += (weight_decay / (2 * batch_samples)) * l2_penalty
            
            accumulated_loss += batch_loss
            
            predicted_labels = np.argmax(predictions, axis=0)
            true_labels = np.argmax(Y_current, axis=0)
            batch_accuracy = np.mean(predicted_labels == true_labels)
            correct_count += batch_accuracy
            
            # Compute gradients via backpropagation
            gradients = backprop(Y_current, activation_cache, model_params)
            step_count += 1
            
            # Configure optimizer hyperparameters
            optimizer_config = {
                "beta": optimal_config.get("momentum", 0.9),
                "beta1": optimal_config.get("beta1", 0.9),
                "beta2": optimal_config.get("beta2", 0.999),
                "epsilon": optimal_config.get("epsilon", 1e-8),
            }
            
            # Apply parameter updates
            model_params, opt_state = update_parameters(
                model_params,
                gradients,
                optimizer=optimal_config.get("optimizer", "sgd"),
                optimizer_state=opt_state,
                eta=optimal_config.get("learning_rate", 0.01),
                t=step_count,
                hyperparams=optimizer_config
            )
        
        # Evaluate on validation set
        val_outputs, _ = forward_propagation(X_val, model_params)
        val_predicted = np.argmax(val_outputs, axis=0)
        val_actual = np.argmax(Y_val, axis=0)
        val_performance = np.mean(val_predicted == val_actual)
        
        # Log metrics for this epoch
        wandb.log({
            "epoch": training_epoch + 1,
            "train_loss": accumulated_loss / batches_per_epoch,
            "train_accuracy": correct_count / batches_per_epoch,
            "val_accuracy": val_performance,
        })
    
    # Final evaluation on test set
    test_outputs, _ = forward_propagation(X_test, model_params)
    test_predicted = np.argmax(test_outputs, axis=0)
    test_actual = np.argmax(Y_test, axis=0)
    final_accuracy = np.mean(test_predicted == test_actual)
    
    # Log test results and visualizations
    wandb.log({
        "test_accuracy": final_accuracy,
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_actual,
            preds=test_predicted,
            class_names=CLASS_NAMES,
        ),
    })
    
    print(f"Final test accuracy: {final_accuracy:.4f}")
    print("Results and confusion matrix logged to Weights & Biases dashboard")

if __name__ == "__main__":
    train_best_model()
