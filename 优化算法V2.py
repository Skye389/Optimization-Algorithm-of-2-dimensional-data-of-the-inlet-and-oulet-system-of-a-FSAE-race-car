import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import r2_score, mean_squared_error # type: ignore
import numpy as np # type: ignore
from deap import base, creator, tools, algorithms # type: ignore
import matplotlib.pyplot as plt
import joblib

# Read DOE data, skip the first row (unit row), use the second row as header
df = pd.read_excel('doe 6 variables ver1.xlsx', header=1)
df.columns = df.columns.str.strip()  # Remove whitespace from column names
print("Column names:", df.columns.tolist())

# Build features and targets
X = df.iloc[:, [0, 1, 2, 3, 4, 5]]    # Engine Speed, Length_man1, Length_man2, Volume, Length_ex1, Length_ex2
y_torque = df.iloc[:, 8]     # Brake Torque
y_cylinder1 = df.iloc[:, 6]  # Cylinder1
y_cylinder2 = df.iloc[:, 7]  # Cylinder2

# Train and evaluate engine torque model
X_train, X_test, y_train, y_test = train_test_split(X, y_torque, test_size=0.2, random_state=42)
rf_torque = RandomForestRegressor(random_state=42)
rf_torque.fit(X_train, y_train)
joblib.dump(rf_torque, 'rf_torque.pkl')  # Save torque model
y_pred = rf_torque.predict(X_test)
print("\nEngine torque model Test R²:", r2_score(y_test, y_pred))
print("Engine torque model Test MSE:", mean_squared_error(y_test, y_pred))
plt.figure(figsize=(6, 6))
plt.scatter(y_train, rf_torque.predict(X_train), color='red', s=20, label='Training')
plt.scatter(y_test, y_pred, color='blue', s=20, label='Testing')
min_torque = min(y_torque.min(), rf_torque.predict(X_train).min(), y_pred.min())
max_torque = max(y_torque.max(), rf_torque.predict(X_train).max(), y_pred.max())
plt.plot([min_torque, max_torque], [min_torque, max_torque], 'g-', lw=2, label='R²=1 (y=x)')
plt.xlabel('True Brake Torque')
plt.ylabel('Predicted Brake Torque')
plt.title(f'Engine Torque Regression\nTrain R²={r2_score(y_train, rf_torque.predict(X_train)):.3f}, Test R²={r2_score(y_test, y_pred):.3f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Train and evaluate Cylinder1 model
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_cylinder1, test_size=0.2, random_state=42)
rf_cylinder1 = RandomForestRegressor(random_state=42)
rf_cylinder1.fit(X_train1, y_train1)
joblib.dump(rf_cylinder1, 'rf_cylinder1.pkl')  # Save cylinder1 model
y_pred1 = rf_cylinder1.predict(X_test1)
print("\nCylinder1 model Test R²:", r2_score(y_test1, y_pred1))
print("Cylinder1 model Test MSE:", mean_squared_error(y_test1, y_pred1))
plt.figure(figsize=(6, 6))
plt.scatter(y_train1, rf_cylinder1.predict(X_train1), color='red', s=20, label='Training')
plt.scatter(y_test1, y_pred1, color='blue', s=20, label='Testing')
min_cyl1 = min(y_cylinder1.min(), rf_cylinder1.predict(X_train1).min(), y_pred1.min())
max_cyl1 = max(y_cylinder1.max(), rf_cylinder1.predict(X_train1).max(), y_pred1.max())
plt.plot([min_cyl1, max_cyl1], [min_cyl1, max_cyl1], 'g-', lw=2, label='R²=1 (y=x)')
plt.xlabel('True Cylinder1')
plt.ylabel('Predicted Cylinder1')
plt.title(f'Cylinder1 Regression\nTrain R²={r2_score(y_train1, rf_cylinder1.predict(X_train1)):.3f}, Test R²={r2_score(y_test1, y_pred1):.3f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Train and evaluate Cylinder2 model
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_cylinder2, test_size=0.2, random_state=42)
rf_cylinder2 = RandomForestRegressor(random_state=42)
rf_cylinder2.fit(X_train2, y_train2)
joblib.dump(rf_cylinder2, 'rf_cylinder2.pkl')  # Save cylinder2 model
y_pred2 = rf_cylinder2.predict(X_test2)
print("\nCylinder2 model Test R²:", r2_score(y_test2, y_pred2))
print("Cylinder2 model Test MSE:", mean_squared_error(y_test2, y_pred2))
plt.figure(figsize=(6, 6))
plt.scatter(y_train2, rf_cylinder2.predict(X_train2), color='red', s=20, label='Training')
plt.scatter(y_test2, y_pred2, color='blue', s=20, label='Testing')
min_cyl2 = min(y_cylinder2.min(), rf_cylinder2.predict(X_train2).min(), y_pred2.min())
max_cyl2 = max(y_cylinder2.max(), rf_cylinder2.predict(X_train2).max(), y_pred2.max())
plt.plot([min_cyl2, max_cyl2], [min_cyl2, max_cyl2], 'g-', lw=2, label='R²=1 (y=x)')
plt.xlabel('True Cylinder2')
plt.ylabel('Predicted Cylinder2')
plt.title(f'Cylinder2 Regression\nTrain R²={r2_score(y_train2, rf_cylinder2.predict(X_train2)):.3f}, Test R²={r2_score(y_test2, y_pred2):.3f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Generate random validation data and save as Test data.csv (with torque prediction)
num_samples = 10
rpm_range = np.arange(2000, 9001, 250)
test_data = {
    'Length_man1': np.random.uniform(40, 400, num_samples),
    'Length_man2': np.random.uniform(40, 400, num_samples),
    'Volume': np.random.uniform(3000000.0, 6500000.0, num_samples),
    'Length_ex1': np.random.uniform(10, 350, num_samples),
    'Length_ex2': np.random.uniform(10, 350, num_samples)
 }
test_df = pd.DataFrame(test_data)

# For each parameter set, predict torque at each RPM and save results
torque_results = []
for idx, row in test_df.iterrows():
     torque_row = {
         'Length_man1': row['Length_man1'],
         'Length_man2': row['Length_man2'],
         'Volume': row['Volume'],
         'Length_ex1': row['Length_ex1'],
         'Length_ex2': row['Length_ex2']
     }
     for rpm in rpm_range:
        features = np.array([[rpm, row['Length_man1'], row['Length_man2'], row['Volume'], row['Length_ex1'], row['Length_ex2']]])
        torque = rf_torque.predict(features)[0]
        torque_row[f'Torque_{rpm}'] = torque
     torque_results.append(torque_row)

result_df = pd.DataFrame(torque_results)
result_df.to_csv('Test data.csv', index=False)
print("\nRandom validation data generated and torque predictions saved to Test data.csv:")
print(result_df)

# Optimization variable range
engine_speeds = np.arange(3000, 6001, 250)  # 3000~6000rpm, step 250

# Multi-objective evaluation function for NSGA-II
def eval_multi(individual):
    length_man1, length_man2, volume, length_ex = individual
    # Clip variables to engineering range
    length_man1 = np.clip(length_man1, 75, 400)
    length_man2 = np.clip(length_man2, 75, 400)
    volume = np.clip(volume, 3000000.0, 6500000.0)
    length_ex1 = np.clip(length_ex, 100, 350)
    length_ex2 = length_ex1  # Always keep both exhaust manifold lengths equal

    # Penalty for engineering constraints
    penalty = 0
    # Intake manifold length difference penalty (if difference > 30mm)
    diff = abs(length_man1 - length_man2)
    if diff > 30:
        penalty += (diff - 30) * 10  # Penalty weight can be adjusted
    # Intake manifold length must not be less than 75mm
    if length_man1 < 75:
        penalty += (75 - length_man1) * 20
    if length_man2 < 75:
        penalty += (75 - length_man2) * 20

    X_pred = np.array([
        [rpm, length_man1, length_man2, volume, length_ex1, length_ex2]
        for rpm in engine_speeds
    ])
    torque_pred = rf_torque.predict(X_pred)
    mean_torque = np.mean(torque_pred)
    var_torque = np.var(torque_pred)
    cyl1_pred = rf_cylinder1.predict(X_pred)
    cyl2_pred = rf_cylinder2.predict(X_pred)
    intake_diff = np.mean(np.abs(cyl1_pred - cyl2_pred))

    # Return multi-objective: maximize mean torque, minimize variance (with penalty), minimize intake difference
    return -mean_torque, var_torque + penalty, intake_diff

# DEAP: define multi-objective fitness and individual
creator.create("FitnessMulti", base.Fitness, weights=(0.5, -1.0, -0.3))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
# Register random initialization for each optimization variable
toolbox.register("attr_length_man1", np.random.uniform, 40, 400)
toolbox.register("attr_length_man2", np.random.uniform, 40, 400)
toolbox.register("attr_volume", np.random.uniform, 3000000.0, 6500000.0)
toolbox.register("attr_length_ex", np.random.uniform, 100, 350)
toolbox.register(
    "individual", tools.initCycle, creator.Individual,
    (
        toolbox.attr_length_man1,
        toolbox.attr_length_man2,
        toolbox.attr_volume,
        toolbox.attr_length_ex
    ), n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_multi)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# For recording the best objective values of each generation
best_var_torque = []
best_mean_torque = []
best_intake_diff = []

NGEN = 200
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot([], [], label='Best Variance')
line2, = ax.plot([], [], label='Best Mean Torque')
line3, = ax.plot([], [], label='Best Intake Diff')
ax.set_xlabel('Generation')
ax.set_ylabel('Objective Value')
ax.set_title('Optimization Progress')
ax.legend()
ax.grid(True)

pop = toolbox.population(n=200)

# Evaluate initial population
for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

# Evolutionary optimization loop with real-time visualization
for gen in range(NGEN):
    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    pop = toolbox.select(pop + offspring, k=200)

    # Record the best individual's objective values of this generation
    best = tools.selBest(pop, 1)[0]
    best_var_torque.append(best.fitness.values[1])
    best_mean_torque.append(-best.fitness.values[0])
    best_intake_diff.append(best.fitness.values[2])

    # Dynamically update the plot
    line1.set_data(range(len(best_var_torque)), best_var_torque)
    line2.set_data(range(len(best_mean_torque)), best_mean_torque)
    line3.set_data(range(len(best_intake_diff)), best_intake_diff)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)  # Refresh interval

    print(f"Gen {gen+1}: Best variance={best.fitness.values[1]:.2f}, Mean torque={-best.fitness.values[0]:.2f}, Intake diff={best.fitness.values[2]:.4f}")

plt.tight_layout()
plt.show()

# Extract Pareto optimal solutions
pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

# Print Pareto optimal solutions
print("\nPareto optimal solutions:")
for ind in pareto_front:
    print(
        f"Length_man1: {ind[0]:.2f}, Length_man2: {ind[1]:.2f}, Volume: {ind[2]:.2f}, "
        f"Length_ex1/2: {ind[3]:.2f}, "
        f"Mean torque: {-ind.fitness.values[0]:.2f}, Variance: {ind.fitness.values[1]:.2f}, Intake diff: {ind.fitness.values[2]:.4f}"
    )

# Save Pareto optimal solutions to CSV
pareto_df = pd.DataFrame([
    {
        'Length_man1': ind[0],
        'Length_man2': ind[1],
        'Volume': ind[2],
        'Length_ex1': ind[3],
        'Length_ex2': ind[3],  # Always keep both exhaust manifold lengths equal
        'Mean_torque': -ind.fitness.values[0],
        'Variance': ind.fitness.values[1],
        'Intake_diff': ind.fitness.values[2]
    }
    for ind in pareto_front
])

# Filter out any solutions with negative values
pareto_df = pareto_df[(pareto_df >= 0).all(axis=1)]

pareto_df.to_csv('pareto_solutions_1.csv', index=False)
print("\nPareto optimal solutions saved to pareto_solutions_1.csv.")

















