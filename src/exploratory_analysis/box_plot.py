import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("data/interim/emissions.csv")
df = df.rename(columns={"Unnamed: 0": "Date"})
df["Date"] = pd.to_datetime(df["Date"])

# Set 'Date' as the index of the DataFrame
df.set_index("Date", inplace=True)

# Create a boxplot for each province
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.title("Boxplot of Carbon Emissions for Each Province")
plt.savefig(
    "src/exploratory_analysis/boxplot.png",
    dpi=300,
    transparent=True,
)
