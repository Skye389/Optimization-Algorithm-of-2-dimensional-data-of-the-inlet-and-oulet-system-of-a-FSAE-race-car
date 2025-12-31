import numpy as np
import pandas as pd
import joblib
from deap import base, creator, tools, algorithms

# Load trained models
rf_torque = joblib.load('rf_torque.pkl')
rf_cylinder1 = joblib.load('rf_cylinder1.pkl')
rf_cylinder2 = joblib.load('rf_cylinder2.pkl')

# Engine speed range for evaluation
engine_speeds = np.arange(3000, 6001, 250)

# Variable ranges for optimization
var_ranges = {
    'Length_man1': (75, 400),
    'Length_man2': (75, 400),
    'Volume': (3000000.0, 6500000.0),
    'Length_ex1': (100, 350),
    'Length_ex2': (100, 350)
}

# Specify known variables here; others will be optimized
fixed_vars = {
     'Length_man1': 107.3,
     'Length_man2': 94.7
     # If exhaust manifold lengths are known, add here
     # 'Length_ex1': 200,
     # 'Length_ex2': 200
}

# Automatically determine which variables to optimize
opt_vars = [v for v in var_ranges if v not in fixed_vars]

# If both exhaust manifold lengths need to be optimized, only keep one (Length_ex1), and always set Length_ex2 = Length_ex1
if ('Length_ex1' in opt_vars) and ('Length_ex2' in opt_vars):
    opt_vars.remove('Length_ex2')

def eval_multi(individual):
    # Build a complete variable dictionary
    var_dict = fixed_vars.copy()
    for i, v in enumerate(opt_vars):
        var_dict[v] = individual[i]
    # Only clip variables that actually exist in var_dict
    for v in var_dict:
        var_dict[v] = np.clip(var_dict[v], *var_ranges[v])

    # If exhaust manifold needs to be optimized, set Length_ex2 = Length_ex1
    if 'Length_ex1' in opt_vars:
        var_dict['Length_ex2'] = var_dict['Length_ex1']

    # Penalty for intake manifold length difference
    penalty = 0
    if 'Length_man1' in opt_vars or 'Length_man2' in opt_vars:
        diff = abs(var_dict['Length_man1'] - var_dict['Length_man2'])
        if diff > 30:
            penalty += (diff - 30) * 10  # Penalty weight can be adjusted

    # Build prediction input for all engine speeds
    X_pred = np.array([
        [rpm, var_dict['Length_man1'], var_dict['Length_man2'], var_dict['Volume'], var_dict['Length_ex1'], var_dict['Length_ex2']]
        for rpm in engine_speeds
    ])
    # Predict torque and intake for each speed
    torque_pred = rf_torque.predict(X_pred)
    mean_torque = np.mean(torque_pred)
    var_torque = np.var(torque_pred)
    cyl1_pred = rf_cylinder1.predict(X_pred)
    cyl2_pred = rf_cylinder2.predict(X_pred)
    intake_diff = np.mean(np.abs(cyl1_pred - cyl2_pred))
    # Return multi-objective: maximize mean torque, minimize variance (with penalty), minimize intake difference
    return -mean_torque, var_torque + penalty, intake_diff

# Define multi-objective fitness: (mean torque, variance, intake diff)
creator.create("FitnessMulti", base.Fitness, weights=(0.5, -1.0, -0.3))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
# Register random initialization for each optimization variable
for v in opt_vars:
    toolbox.register(f"attr_{v}", np.random.uniform, *var_ranges[v])
toolbox.register(
    "individual", tools.initCycle, creator.Individual,
    tuple(getattr(toolbox, f"attr_{v}") for v in opt_vars), n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_multi)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Create initial population
pop = toolbox.population(n=100)
# Run the evolutionary algorithm
algorithms.eaMuPlusLambda(
    pop, toolbox,
    mu=100, lambda_=200,
    cxpb=0.7, mutpb=0.3,
    ngen=100, verbose=True
)

# Extract Pareto front
pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

# Output and save results
results = []
for ind in pareto_front:
    var_dict = fixed_vars.copy()
    for i, v in enumerate(opt_vars):
        var_dict[v] = ind[i]
    # Only clip variables that actually exist in var_dict
    for v in var_dict:
        var_dict[v] = np.clip(var_dict[v], *var_ranges[v])
    # If exhaust manifold needs to be optimized, set Length_ex2 = Length_ex1
    if 'Length_ex1' in opt_vars:
        var_dict['Length_ex2'] = var_dict['Length_ex1']
    results.append({
        **var_dict,
        'Mean_torque': -ind.fitness.values[0],
        'Variance': ind.fitness.values[1],
        'Intake_diff': ind.fitness.values[2]
    })

pareto_df = pd.DataFrame(results)
# Filter out any solutions with negative values
pareto_df = pareto_df[(pareto_df >= 0).all(axis=1)]
pareto_df.to_csv('pareto_solutions_custom_1.csv', index=False)
print("\nPareto optimal solutions saved to pareto_solutions_custom.csv.")