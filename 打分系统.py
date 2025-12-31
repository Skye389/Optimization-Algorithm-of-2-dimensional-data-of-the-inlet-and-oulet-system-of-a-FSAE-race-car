import pandas as pd

# Read all non-negative Pareto optimal solutions
pareto_df = pd.read_csv('pareto_solutions_8.csv')

# Filter: Remove solutions where either intake manifold is shorter than 65mm,
# or the length difference between the two intake manifolds exceeds 30mm
pareto_df = pareto_df[
    (pareto_df['Length_man1'] >= 65) &
    (pareto_df['Length_man2'] >= 65) &
    (abs(pareto_df['Length_man1'] - pareto_df['Length_man2']) <= 30)
]

# Normalization function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

# Normalize mean torque and variance
pareto_df['norm_mean_torque'] = normalize(pareto_df['Mean_torque'])
pareto_df['norm_variance'] = normalize(pareto_df['Variance'])

# Scoring mechanism: score = 0.4 * normalized mean torque - 0.6 * normalized variance
pareto_df['score'] = 0.4 * pareto_df['norm_mean_torque'] - 0.6 * pareto_df['norm_variance']

# Sort by score and select the top 10 solutions
top10 = pareto_df.sort_values('score', ascending=False).head(10)

# Save the top 10 solutions to a new CSV file
top10.to_csv('top10_scored_pareto_8.csv', index=False)
print("Top 10 scored Pareto solutions have been saved to top10_scored_pareto_4.csv")
print(top10)