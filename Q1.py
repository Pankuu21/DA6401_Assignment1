from keras.datasets import fashion_mnist
import numpy as np
import wandb
WANDB_ENTITY = "pankuu21-indian-institute-of-technology-madras"  
WANDB_PROJECT = "DA6401_Assignment1"
wandb.login()
run = wandb.init(project="DA6401_Assignment1", name="class-examples")

# Class labels for Fashion MNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def load_data(dataset="fashion_mnist", val_split=None):
    
    if dataset.lower() == "fashion_mnist":
        from keras.datasets import fashion_mnist as data_module
    elif dataset.lower() == "mnist":
        from keras.datasets import mnist as data_module
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'fashion_mnist' or 'mnist'")
    
    # Load data
    (X_train, y_train), (X_test, y_test) = data_module.load_data()
    
    # Keep the original image shape for visualization
    X_train_images = X_train.copy()
    X_test_images = X_test.copy()
    
    # Normalize images
    X_train_images = X_train_images / 255.0
    X_test_images = X_test_images / 255.0
    
    # Reshape and transpose for ML processing
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0

    # One-hot encode labels
    def one_hot_encode(y, num_classes=10):
        return np.eye(num_classes)[y].T

    Y_train = one_hot_encode(y_train)
    Y_test = one_hot_encode(y_test)
    
    # Create validation split if requested
    if val_split is not None:
        if not (0 < val_split < 1):
            raise ValueError("val_split must be between 0 and 1")
        
        m = X_train.shape[1]
        val_size = int(m * val_split)
        
        # Generate validation indices
        np.random.seed(42)  
        indices = np.random.permutation(m)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # Create validation set
        X_val = X_train[:, val_indices]
        Y_val = Y_train[:, val_indices]
        
        # Update training set
        X_train = X_train[:, train_indices]
        Y_train = Y_train[:, train_indices]
        
        print(f"Dataset: {dataset}")
        print(f"Train set: {X_train.shape[1]} examples")
        print(f"Validation set: {X_val.shape[1]} examples")
        print(f"Test set: {X_test.shape[1]} examples")
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    # Return original format if no validation split
    return X_train, Y_train, X_test, Y_test, y_test, X_train_images, y_train

# Correctly unpack the return values
X_train, Y_train, X_test, Y_test, y_test, X_train_images, y_train_orig = load_data()

# Create a table for wandb
columns = ["image", "label", "class_name"]



class_indices = {i: np.where(y_train_orig == i)[0][0] for i in range(10)}
data = [
    [wandb.Image(X_train_images[idx], caption=CLASS_NAMES[i]), int(i), CLASS_NAMES[i]]
    for i, idx in class_indices.items()
]


# Log to wandb
fashion_table = wandb.Table(data=data, columns=columns)
wandb.log({"fashion_mnist_samples": fashion_table})

# Create a collection of all images to display in one panel
image_collection = []
for i in range(10):
    idx = np.where(y_train_orig == i)[0][0]  
    image_collection.append(
        wandb.Image(
            X_train_images[idx],  
            caption=f"{i}: {CLASS_NAMES[i]}"
        )
    )

# Log the collection of images
wandb.log({"class_examples": image_collection})

print("successfully logged to Weights & Biases!")
wandb.finish()
