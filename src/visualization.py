import matplotlib.pyplot as plt
import numpy as np

def plot_evolution(best_fitness, avg_fitness):
    generations = np.arange(len(best_fitness))
    plt.figure(figsize=(10,5))
    plt.plot(generations, best_fitness, label='Best Fitness')
    plt.plot(generations, avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_generation_table(population):
    print(f"{'Individuo':>10} | {'Mapeo':>10} | {'ADN':>20} | {'Fitness':>10}")
    print('-'*60)
    for i, ind in enumerate(population.individuals):
        print(f"{i:>10} | {str(ind.decode()):>10} | {str(ind.dna):>20} | {ind.fitness:>10.4f}")
