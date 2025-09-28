import numpy as np

def random_mutation(dna, pm=0.01, encoding='binary', bounds=(0, 1)):
    dna = dna.copy()
    for i in range(len(dna)):
        if np.random.rand() < pm:
            if encoding == 'binary':
                dna[i] = 1 - dna[i]
            elif encoding == 'real':
                dna[i] = np.random.uniform(bounds[0], bounds[1])
    return dna

def uniform_mutation(dna, pm=0.01, encoding='binary', bounds=(0, 1)):
    dna = dna.copy()
    if encoding == 'binary':
        mask = np.random.rand(len(dna)) < pm
        dna[mask] = 1 - dna[mask]
    elif encoding == 'real':
        mask = np.random.rand(len(dna)) < pm
        dna[mask] = np.random.uniform(bounds[0], bounds[1], np.sum(mask))
    return dna

def non_uniform_mutation(dna, pm=0.01, generation=1, max_generations=100, encoding='real', bounds=(0, 1)):
    dna = dna.copy()
    for i in range(len(dna)):
        if np.random.rand() < pm:
            delta = (bounds[1] - bounds[0]) * (1 - generation / max_generations)
            dna[i] += np.random.uniform(-delta, delta)
            dna[i] = np.clip(dna[i], bounds[0], bounds[1])
    return dna

def swap_mutation(dna, pm=0.01):
    dna = dna.copy()
    if np.random.rand() < pm:
        idx1, idx2 = np.random.choice(len(dna), 2, replace=False)
        dna[idx1], dna[idx2] = dna[idx2], dna[idx1]
    return dna
