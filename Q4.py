import numpy as np
import wandb
import yaml
import argparse
import os
from Q1 import load_data, CLASS_NAMES, WANDB_ENTITY, WANDB_PROJECT
from Q2 import initialize_parameters, forward_propagation
from Q3 import backprop, update_parameters

# Hey, here's our loss function!
def lossfn(AL, Y, parameters=None, weight_decay=0):
    
    m = Y.shape[1]
    
    cross_entropy_loss = -np.sum(Y * np.log(AL + 1e-8)) / m
    
    regularization_term = 0
    if weight_decay > 0 and parameters is not None:
        L = len(parameters) // 2
        for l in range(1, L+1):
            regularization_term += np.linalg.norm(parameters[f"W{l}"], 'fro') ** 2
        regularization_term *= (weight_decay / (2 * m))
    
    # Total cost
    cost = cross_entropy_loss + regularization_term
    
    return cost

# Training function for Wandb sweep
def train_fashion():
    """
    This is our training function that Wandb sweep will call.
    It trains the model using the hyperparameters that Wandb gives us.
    Alternative implementation with different control flow.
    """
    # Initialize wandb run differently - explicit init/finish pattern
    run = wandb.init(group="Q4_sweep", entity=WANDB_ENTITY, project=WANDB_PROJECT)
    config = run.config
    
    try:
        # Data loading with validation split
        dataset = load_data(dataset="fashion_mnist", val_split=0.1)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset
        
        # Create run name using string joining instead of f-strings
        name_components = [
            "hl_" + str(config.num_layers),
            "bs_" + str(config.batch_size),
            "ac_" + str(config.activation),
            "wd_" + str(config.weight_decay),
            "lr_" + str(config.learning_rate),
            "opt_" + str(config.optimizer),
            "wi_" + str(config.weight_init)
        ]
        run.name = "_".join(name_components)
        run.save()
        
        # Network architecture setup with list extension
        layer_dims = [784]
        layer_dims.extend([config.hidden_size] * config.num_layers)
        layer_dims.append(10)
        
        # Parameter initialization with conditional assignment
        if config.weight_init.lower() == "xavier":
            parameters = {}
            for l in range(1, len(layer_dims)):
                scale = np.sqrt(1 / layer_dims[l-1])
                parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale
                parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
        else:  # default to random
            parameters = initialize_parameters(layer_dims)
            
        # Training setup with different variable names
        state = {}  # optimizer state
        time_step = 0
        sample_count = X_train.shape[1]
        mini_batch_size = config.batch_size
        iterations_per_epoch = sample_count // mini_batch_size
        
        # Precompute batch indices for all epochs
        all_batch_indices = []
        for _ in range(config.epochs):
            indices = np.random.permutation(sample_count)
            epoch_batches = []
            idx = 0
            while idx < sample_count:
                end_idx = min(idx + mini_batch_size, sample_count)
                epoch_batches.append(indices[idx:end_idx])
                idx = end_idx
            all_batch_indices.append(epoch_batches)
        
        # Training loop using while instead of for
        epoch = 0
        metrics = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
        
        while epoch < config.epochs:
            # Track epoch metrics
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # Process each batch using the precomputed indices
            for batch_indices in all_batch_indices[epoch]:
                X_batch = X_train[:, batch_indices]
                Y_batch = Y_train[:, batch_indices]
                batch_size = X_batch.shape[1]
                
                # Run forward pass
                predictions, activations = forward_propagation(X_batch, parameters)
                
                # Compute and accumulate loss
                current_loss = lossfn(predictions, Y_batch, parameters, config.weight_decay)
                total_loss += current_loss * batch_size
                
                # Update accuracy metrics
                predicted_classes = np.argmax(predictions, axis=0)
                true_classes = np.argmax(Y_batch, axis=0)
                correct = np.sum(predicted_classes == true_classes)
                correct_predictions += correct
                total_predictions += batch_size
                
                # Compute gradients and update parameters
                gradients = backprop(Y_batch, activations, parameters)
                time_step += 1
                
                # Set up optimizer hyperparameters as a dictionary comprehension
                optimizer_params = {
                    param: config.get(param, default_val) 
                    for param, default_val in [
                        ("beta", 0.9), 
                        ("beta1", 0.9), 
                        ("beta2", 0.999), 
                        ("epsilon", 1e-8)
                    ]
                }
                
                # Update model parameters
                parameters, state = update_parameters(
                    parameters,
                    gradients,
                    optimizer=config.optimizer,
                    optimizer_state=state,
                    eta=config.learning_rate,
                    t=time_step,
                    hyperparams=optimizer_params,
                )
            
            if epoch % 1 == 0 or epoch == config.epochs - 1:
                output_val, _ = forward_propagation(X_val, parameters)
                val_loss = lossfn(output_val, Y_val)
                
                val_pred_classes = np.argmax(output_val, axis=0)
                val_true_classes = np.argmax(Y_val, axis=0)
                val_accuracy = np.mean(val_pred_classes == val_true_classes)
            
                metrics["train_loss"].append(total_loss / sample_count)
                metrics["train_accuracy"].append(correct_predictions / total_predictions)
                metrics["val_loss"].append(val_loss)
                metrics["val_accuracy"].append(val_accuracy)
                
                
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": metrics["train_loss"][-1],
                    "train_accuracy": metrics["train_accuracy"][-1],
                    "val_loss": metrics["val_loss"][-1],
                    "val_accuracy": metrics["val_accuracy"][-1],
                })
            
            epoch += 1
    
    finally:
        # Make sure to finish the wandb run
        wandb.finish()

# Our main function to run or resume Wandb sweep
def main():
    
    # Set up command line interface
    cmd_parser = argparse.ArgumentParser(
        description="Fashion MNIST hyperparameter optimization using WandB sweeps"
    )
    cmd_parser.add_argument("--continue_sweep", dest="resume", action="store_true", 
                         help="Continue a previously started sweep")
    cmd_parser.add_argument("--id", dest="sweep_id", type=str, 
                         help="Specific sweep identifier to continue")
    
    # Parse and process arguments
    cmd_args = cmd_parser.parse_args()
    
    # Try to log in to WandB first
    try:
        wandb.login()
        print(" Successfully logged in to Weights & Biases")
    except Exception as e:
        print(f"Failed to log in to Weights & Biases: {e}")
        print("Please check your credentials and try again.")
        return
        
    # Determine sweep ID based on command arguments
    sweep_identifier = None
    
    # Case: Continue existing sweep
    if cmd_args.resume:
        # First priority: Use command-line provided ID
        if cmd_args.sweep_id:
            sweep_identifier = cmd_args.sweep_id
            print(f" Continuing sweep with ID provided via command line: {sweep_identifier}")
        # Second priority: Look for saved ID in file
        else:
            sweep_id_file = "sweep_id.txt"
            if not os.path.exists(sweep_id_file):
                print(f"Cannot resume: {sweep_id_file} not found")
                print("Either provide a sweep ID with --id or start a new sweep")
                return
                
            with open(sweep_id_file, 'r') as id_file:
                sweep_identifier = id_file.read().strip()
                if not sweep_identifier:
                    print("Empty sweep ID found in file")
                    return
                print(f"Continuing sweep with ID from file: {sweep_identifier}")
    
    # Case: Create new sweep
    else:
        # Load sweep configuration
        try:
            config_file = "sweep_config.yaml"
            with open(config_file, 'r') as yaml_file:
                sweep_configuration = yaml.safe_load(yaml_file)
                
            # Initialize new sweep
            sweep_identifier = wandb.sweep(
                sweep_configuration, 
                entity=WANDB_ENTITY, 
                project=WANDB_PROJECT
            )
            
            print(f"🎉 Created new sweep with ID: {sweep_identifier}")
            
            # Save ID for future use
            with open("sweep_id.txt", 'w') as id_file:
                id_file.write(sweep_identifier)
                print("Saved sweep ID to sweep_id.txt")
                
        except FileNotFoundError:
            print(f"Sweep configuration file '{config_file}' not found")
            return
        except Exception as e:
            print(f"Error creating sweep: {e}")
            return
    
    # Start the sweep agent
    print("Launching sweep agent...")
    wandb.agent(sweep_identifier, function=train_fashion, count=None)

if __name__ == "__main__":
    main()
