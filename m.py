import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
true_df = pd.read_csv("dataset\True.csv")
fake_df = pd.read_csv("dataset\Fake.csv")

# Add labels: 1 for True, 0 for Fake
true_df['label'] = 1
fake_df['label'] = 0

# Merge the datasets
merged_df = pd.concat([true_df, fake_df], ignore_index=True)

# Plot a pie chart
label_counts = merged_df['label'].value_counts()
labels = ['True News', 'Fake News']
sizes = [label_counts[1], label_counts[0]]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of True vs Fake News')
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.show()
