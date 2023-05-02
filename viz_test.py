import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('test_outputs.csv')

# Group the data by race, gender, and age
grouped = df.groupby(['race', 'gender', 'age'])

# Create a scatter plot for each column
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(['race_scores_fair_4', 'gender_scores_fair', 'age_scores_fair']):
    axs[i].set_title(col.capitalize())
    axs[i].set_xlabel('Image')
    axs[i].set_ylabel('Score')
    for name, group in grouped:
        x = group.index
        y = group[col]
        axs[i].scatter(x, y, label=name)
    axs[i].legend()

plt.show()

plt.show()