from individual import Individual
import numpy as np

class Population:
    def __init__(self, size, dna_length, encoding='binary', bounds=(0, 1)):
        self.individuals = [Individual.random(dna_length, encoding, bounds) for _ in range(size)]

    def evaluate(self, fitness_func):
        for ind in self.individuals:
            ind.fitness = fitness_func(ind.decode())

    def get_best(self):
        return max(self.individuals, key=lambda ind: ind.fitness)

    def get_average_fitness(self):
        return np.mean([ind.fitness for ind in self.individuals])

    def __repr__(self):
        return f"Population(size={len(self.individuals)})"
