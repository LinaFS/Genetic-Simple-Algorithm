import numpy as np

def uniform_crossover(parent1, parent2, pc=0.8):
    if np.random.rand() > pc:
        return parent1.dna.copy(), parent2.dna.copy()
    mask = np.random.randint(0, 2, len(parent1.dna)).astype(bool)
    child1 = np.where(mask, parent1.dna, parent2.dna)
    child2 = np.where(mask, parent2.dna, parent1.dna)
    return child1, child2

def arithmetic_crossover(parent1, parent2, pc=0.8):
    if np.random.rand() > pc:
        return parent1.dna.copy(), parent2.dna.copy()
    alpha = np.random.rand()
    child1 = alpha * parent1.dna + (1-alpha) * parent2.dna
    child2 = alpha * parent2.dna + (1-alpha) * parent1.dna
    return child1, child2

def flat_crossover(parent1, parent2, pc=0.8):
    if np.random.rand() > pc:
        return parent1.dna.copy(), parent2.dna.copy()
    min_dna = np.minimum(parent1.dna, parent2.dna)
    max_dna = np.maximum(parent1.dna, parent2.dna)
    child1 = np.random.uniform(min_dna, max_dna)
    child2 = np.random.uniform(min_dna, max_dna)
    return child1, child2

def blx_alpha_crossover(parent1, parent2, alpha=0.5, pc=0.8):
    if np.random.rand() > pc:
        return parent1.dna.copy(), parent2.dna.copy()
    min_dna = np.minimum(parent1.dna, parent2.dna)
    max_dna = np.maximum(parent1.dna, parent2.dna)
    diff = max_dna - min_dna
    child1 = np.random.uniform(min_dna - alpha*diff, max_dna + alpha*diff)
    child2 = np.random.uniform(min_dna - alpha*diff, max_dna + alpha*diff)
    return child1, child2

def pmx_crossover(parent1, parent2, pc=0.8):
    # Placeholder for PMX (permutation crossover)
    return parent1.dna.copy(), parent2.dna.copy()
