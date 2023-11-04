import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Load the data
data_train = np.load("data/processed/AGCRN_dynamic/data_train.npy")
data_val = np.load("data/processed/AGCRN_dynamic/data_val.npy")
data_test = np.load("data/processed/AGCRN_dynamic/data_test.npy")

# Slice the first variable from your 3D data and transpose it
data_train_var1 = data_train[:, :, 0].T
data_val_var1 = data_val[:, :, 0].T
data_test_var1 = data_test[:, :, 0].T

# Read the CSV file
df = pd.read_csv("data/interim/node_names.csv", header=None)
node_names = df[0].tolist()

# Create a list of dates from 2019.1.1 to 2022.12.31
time_points = pd.date_range(start="2019-01-01", end="2022-12-31").tolist()
# Convert time_points to datetime
time_points_dt = pd.to_datetime(time_points)

# Combine the data
data_combined_var1 = np.concatenate(
    [data_train_var1, data_val_var1, data_test_var1], axis=1
)

# Combine the time points
time_points_combined_dt = time_points_dt.tolist()


# Get the indices for the first day of each month
xticks_indices = [i for i, date in enumerate(time_points_combined_dt) if date.day == 1]

# Create xtick labels for the selected indices
xtick_labels = [time_points_combined_dt[i].strftime("%Y-%m-%d") for i in xticks_indices]

# Plot the heatmap for combined data
fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
cax = sns.heatmap(
    data_combined_var1,
    cmap="Reds",
    yticklabels=node_names,
    cbar=False,  # Don't create a default vertical color bar
    ax=ax,
)
ax.set_title("Heatmap of combined data - Variable 1")
ax.set_xlabel("Time")
ax.set_ylabel("Nodes")

# Set xticks location and labels
plt.xticks(
    xticks_indices, xtick_labels, rotation=45
)  # Rotate the x labels for better visibility

# Create a custom color bar
cbar_ax = fig.add_axes([0.9, 0.93, 0.1, 0.02])  # [left, bottom, width, height]
fig.colorbar(cax.get_children()[0], cax=cbar_ax, orientation="horizontal")
cbar_ax.xaxis.set_ticks_position("top")

# Adjust plot margins
plt.subplots_adjust(
    bottom=0.25, left=0.15, right=0.85
)  # Adjust bottom, left and right margins

plt.savefig("data/processed/AGCRN_dynamic/0heatmap_combined_var1.png")
