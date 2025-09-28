import numpy as np

def elitist_selection(population, num_selected):
    sorted_pop = sorted(population.individuals, key=lambda ind: ind.fitness, reverse=True)
    return sorted_pop[:num_selected]

def roulette_selection(population, num_selected):
    fitnesses = np.array([ind.fitness for ind in population.individuals])
    probs = fitnesses / fitnesses.sum()
    selected = np.random.choice(population.individuals, num_selected, p=probs)
    return list(selected)

def ranking_selection_v1(population, num_selected):
    sorted_pop = sorted(population.individuals, key=lambda ind: ind.fitness)
    ranks = np.arange(1, len(sorted_pop)+1)
    probs = ranks / ranks.sum()
    selected = np.random.choice(sorted_pop, num_selected, p=probs)
    return list(selected)

def ranking_selection_v2(population, num_selected):
    sorted_pop = sorted(population.individuals, key=lambda ind: ind.fitness)
    ranks = np.linspace(0, 1, len(sorted_pop))
    probs = ranks / ranks.sum()
    selected = np.random.choice(sorted_pop, num_selected, p=probs)
    return list(selected)

def tournament_selection(population, num_selected, k=3):
    selected = []
    for _ in range(num_selected):
        aspirants = np.random.choice(population.individuals, k)
        winner = max(aspirants, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected

def steady_state_selection(population, num_selected):
    return elitist_selection(population, num_selected)

def crowding_selection(population, num_selected):
    # Placeholder: implement crowding factor selection
    return np.random.choice(population.individuals, num_selected)
