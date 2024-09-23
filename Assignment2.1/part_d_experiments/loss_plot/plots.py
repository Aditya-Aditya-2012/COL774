import pickle
import matplotlib.pyplot as plt
import os

# Define the directory where the loss files are stored
loss_dir = '/home/civil/btech/ce1210494/COL774/Assignment2.1/part_d_experiments/loss_plot'

# Define the loss files and the corresponding activation functions
loss_files = {
    'GELU': 'losses_5l_gelu.pkl',
    'ReLU': 'losses_5l_relu.pkl',
    'Sigmoid': 'losses_5l_sig.pkl',
    'Swish': 'losses_5l_swish.pkl'
}

# Initialize a dictionary to store the losses
all_losses = {}

# Load the losses from each file
for activation, loss_file in loss_files.items():
    file_path = os.path.join(loss_dir, loss_file)
    with open(file_path, 'rb') as f:
        all_losses[activation] = pickle.load(f)

# Plot the loss curves for each activation function
plt.figure(figsize=(10, 6))

for activation, losses in all_losses.items():
    plt.plot(losses, label=activation)

# Add labels, title, and legend
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Activation Functions')
plt.legend()  # Display activation function labels

# Save the plot as a PNG image
plot_file_path = 'loss_comparison_activations.png'
plt.savefig(plot_file_path)

# Show the plot (optional)
plt.show()

print(f"Plot saved at: {plot_file_path}")
