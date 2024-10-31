import os
import matplotlib.pyplot as plt
from collections import defaultdict

data_type = 'meander'
data_dir = f'resnet50/handwritten_dataset/{data_type}'


def count_images(data_dir, split):
    """Count the number of images in each class for a given split (train or val)."""
    split_dir = os.path.join(data_dir, split)
    class_counts = defaultdict(int)
    
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    
    return class_counts


def plot_class_distribution(train_counts, val_counts):
    classes = list(train_counts.keys())
    train_values = [train_counts[cla] for cla in classes]
    val_values = [val_counts.get(cla, 0) for cla in classes] 
    
    x = range(len(classes)) 

    plt.figure(figsize=(4, 6))
    
    plt.bar(x, train_values, width=0.1, label='Train', align='edge', color='blue')
    plt.bar(x, val_values, width=0.1, label='Validation', align='edge', color='yellow')

    plt.xlabel('Class')
    plt.ylabel('Number of Images') 
    plt.title(f'Distribution of Images in Train and Validation Sets ({data_type})')
    plt.xticks(ticks=x, labels=classes, rotation=45, ha='right')  # Rotate x labels for better readability
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{data_dir}/dataset.png')
    plt.show()


if __name__ == "__main__":
    # Count images in train and validation sets
    train_counts = count_images(data_dir, 'train')
    val_counts = count_images(data_dir, 'val')
    
    # Plot the distribution
    plot_class_distribution(train_counts, val_counts)

