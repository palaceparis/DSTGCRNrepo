import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("data/interim/emissions.csv")
df = df.rename(columns={"Unnamed: 0": "Date"})
df["Date"] = pd.to_datetime(df["Date"])

# Set 'Date' as the index of the DataFrame
df.set_index("Date", inplace=True)

# Plot the emissions trend for each province
plt.figure(figsize=(15, 10))

for column in df.columns:
    plt.plot(df.index, df[column], label=column)

plt.xlabel("Date")
plt.ylabel("Carbon Emissions")
plt.title("Carbon Emissions Trend for Each Province")
plt.legend()
plt.show()


# Calculate the correlation matrix
corr = df.corr()

# Create a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Heatmap of Carbon Emissions Correlation Between Provinces")
plt.show()
