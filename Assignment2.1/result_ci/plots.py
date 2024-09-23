import pickle
import matplotlib.pyplot as plt

# Load the losses from the pickle file
with open('losses.pkl', 'rb') as f:
    all_losses = pickle.load(f)

# Plot the loss curves for each optimizer
plt.figure(figsize=(10, 6))

for optimizer, losses in all_losses.items():
    plt.plot(losses, label=optimizer)

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Optimizers')
plt.legend()  # Display optimizer labels

# Save the plot as a PNG image
plot_file_path = 'loss_plot.png'
plt.savefig(plot_file_path)

# Show the plot (optional)
plt.show()

print(f"Plot saved at: {plot_file_path}")
