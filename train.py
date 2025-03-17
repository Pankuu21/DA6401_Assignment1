import argparse
import numpy as np
import wandb
import os

from Q1 import load_data, CLASS_NAMES
from Q2 import initialize_parameters, forward_propagation, act_hid
from Q3 import backprop, update_parameters

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network with various configurations')
    
    parser.add_argument('-wp', '--wandb_project', default='DA6401_Assignment1', 
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='pankuu21-indian-institute-of-technology-madras',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for training')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size used to train neural network')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function to use for training')
    parser.add_argument('-o', '--optimizer', default='sgd', 
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer to use for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.5,
                        help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.5,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5,
                        help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5,
                        help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001,
                        help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help='Weight decay used by optimizers')
    parser.add_argument('-w_i', '--weight_init', default='random', choices=['random', 'Xavier'],
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1,
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4,
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', default='sigmoid', 
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function for hidden layers')
    
    return parser.parse_args()

def compute_loss(predictions, targets, loss_type="cross_entropy"):
    sample_count = targets.shape[1]
    
    if loss_type == "cross_entropy":
        log_probs = np.log(predictions + 1e-8)
        weighted_logs = targets * log_probs
        total_negative_log_likelihood = -np.sum(weighted_logs)
        return total_negative_log_likelihood / sample_count
    elif loss_type == "mean_squared_error":
        # Calculate MSE using vector operations
        error_matrix = predictions - targets
        squared_errors = np.multiply(error_matrix, error_matrix)
        total_squared_error = np.sum(squared_errors)
        return total_squared_error / (2 * sample_count)
    else:
        options = ["cross_entropy", "mean_squared_error"]
        raise ValueError(f"Invalid loss type '{loss_type}'. Choose from: {options}")

def train_model(args):
    run_config = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "group": "training_runs"
    }
    run = wandb.init(**run_config)
    
    name_parts = []
    name_parts.append(f"hl_{args.num_layers}")
    name_parts.append(f"bs_{args.batch_size}")
    name_parts.append(f"ac_{args.activation}")
    name_parts.append(f"wd_{args.weight_decay}")
    name_parts.append(f"lr_{args.learning_rate}")
    name_parts.append(f"opt_{args.optimizer}")
    name_parts.append(f"wi_{args.weight_init}")
    
    run_name = "_".join(name_parts)
    wandb.run.name = run_name
    wandb.run.save()
    
    # Log configuration using dictionary unpacking
    wandb.config.update({**vars(args)})
    
    # Load and process data 
    dataset_splits = load_data(dataset=args.dataset, val_split=0.1)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset_splits
    
    architecture = [784]
    for _ in range(args.num_layers):
        architecture.append(args.hidden_size)
    architecture.append(10)
    
    initialization_method = args.weight_init.lower()
    if initialization_method == "xavier":
        params = {}
        for idx in range(1, len(architecture)):
            input_size = architecture[idx-1]
            output_size = architecture[idx]
            scale = np.sqrt(1.0 / input_size)
            params[f"W{idx}"] = np.random.randn(output_size, input_size) * scale
            params[f"b{idx}"] = np.zeros((output_size, 1))
    else:
        params = initialize_parameters(architecture)
    
    training_state = {
        "optimizer_memory": {},
        "iteration": 0,
        "dataset_size": X_train.shape[1],
        "batch_count": X_train.shape[1] // args.batch_size
    }
    
    current_epoch = 0
    while current_epoch < args.epochs:
        indices = np.random.permutation(training_state["dataset_size"])
        
        # Track  with dictionary
        epoch_metrics = {
            "loss_sum": 0,
            "correct_predictions": 0,
            "processed_batches": 0
        }
        
        batch_idx = 0
        while batch_idx < training_state["batch_count"]:
            # Calculate batch bounds
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, training_state["dataset_size"])
            
            batch_indices = indices[start_idx:end_idx]
            X_batch = X_train[:, batch_indices]
            Y_batch = Y_train[:, batch_indices]
            
            # Model forward pass
            outputs, cache = forward_propagation(X_batch, params)
            
            # Compute loss value
            batch_loss = compute_loss(outputs, Y_batch, args.loss)
            
            if args.weight_decay > 0:
                l2_terms = [np.sum(params[f"W{l}"]**2) for l in range(1, len(architecture))]
                l2_penalty = (args.weight_decay / (2 * X_batch.shape[1])) * sum(l2_terms)
                batch_loss += l2_penalty
            
            epoch_metrics["loss_sum"] += batch_loss
            
            pred_classes = np.argmax(outputs, axis=0)
            true_classes = np.argmax(Y_batch, axis=0)
            correct = np.sum(pred_classes == true_classes)
            total = Y_batch.shape[1]
            epoch_metrics["correct_predictions"] += correct / total
            epoch_metrics["processed_batches"] += 1
            
            gradients = backprop(Y_batch, cache, params)
            training_state["iteration"] += 1
            
            opt_config = {
                key: getattr(args, key, default) 
                for key, default in [
                    ("beta", 0.5),
                    ("beta1", 0.5),
                    ("beta2", 0.5),
                    ("epsilon", 0.000001)
                ]
            }
            
            # Update parameters
            params, training_state["optimizer_memory"] = update_parameters(
                params,
                gradients,
                optimizer=args.optimizer,
                optimizer_state=training_state["optimizer_memory"],
                eta=args.learning_rate,
                t=training_state["iteration"],
                hyperparams=opt_config
            )
            
            # Move to next batch
            batch_idx += 1
        
        val_predictions, _ = forward_propagation(X_val, params)
        val_loss = compute_loss(val_predictions, Y_val, args.loss)
        
        # Compute validation accuracy directly
        val_classes_predicted = np.argmax(val_predictions, axis=0)
        val_classes_actual = np.argmax(Y_val, axis=0)
        val_accuracy = np.mean(val_classes_predicted == val_classes_actual)
        
        # Calculate training metrics averages
        avg_loss = epoch_metrics["loss_sum"] / epoch_metrics["processed_batches"]
        avg_accuracy = epoch_metrics["correct_predictions"] / epoch_metrics["processed_batches"]
        
        # Log metrics to wandb with step parameter
        wandb_metrics = {
            "epoch": current_epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }
        wandb.log(wandb_metrics)
        
        # Print progress with formatted string
        progress = (
            f"Epoch {current_epoch+1}/{args.epochs} | "
            f"Train: loss={avg_loss:.4f}, acc={avg_accuracy:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_accuracy:.4f}"
        )
        print(progress)
        
        # Increment epoch counter
        current_epoch += 1
    
    test_outputs, _ = forward_propagation(X_test, params)
    test_loss = compute_loss(test_outputs, Y_test, args.loss)
    
    test_pred_classes = np.argmax(test_outputs, axis=0)
    test_true_classes = np.argmax(Y_test, axis=0)
    
    matches = np.sum(test_pred_classes == test_true_classes)
    total_samples = test_true_classes.shape[0]
    test_accuracy = matches / total_samples
    
    print(f"\nEvaluation results:\n  Test loss: {test_loss:.4f}\n  Test accuracy: {test_accuracy:.4f}")
    
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_true_classes,
            preds=test_pred_classes,
            class_names=CLASS_NAMES
        )
    })
    
    return params

def main():
    args = parse_args()
    print("Starting training with the following configuration:")
    print(vars(args))
    
    # Ensure wandb is logged in
    try:
        wandb.login()
    except Exception as e:
        print(f"Error logging into Weights & Biases: {e}")
        print("Training will continue but results won't be logged to W&B.")
    
    # Train model
    trained_parameters = train_model(args)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
