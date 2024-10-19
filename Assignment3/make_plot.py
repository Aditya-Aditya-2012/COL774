import matplotlib.pyplot as plt

def make_plots(train_loss, val_loss, plt_name):
    epochs_range = range(1, len(train_loss) + 1)
    
    # Create figure and set size
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation loss
    plt.plot(epochs_range, train_loss, label='Training Accuracy', color='blue', linestyle='-', marker='o', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Accuracy', color='green', linestyle='--', marker='s', linewidth=2)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add title and labels with larger fonts
    plt.title('Training and Validation Accuracy Over Epochs', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    
    # Customize tick size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend with larger font size
    plt.legend(fontsize=12, loc='best')
    
    # Add tight layout for better spacing
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{plt_name}.png", dpi=300)
    plt.close()
