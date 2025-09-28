import numpy as np

class Individual:
    def __init__(self, dna, encoding='binary'):
        self.encoding = encoding
        self.dna = dna
        self.fitness = None

    @staticmethod
    def random(length, encoding='binary', bounds=(0, 1)):
        if encoding == 'binary':
            dna = np.random.randint(0, 2, length)
        elif encoding == 'real':
            dna = np.random.uniform(bounds[0], bounds[1], length)
        else:
            raise ValueError('Unknown encoding')
        return Individual(dna, encoding)

    def decode(self):
        if self.encoding == 'binary':
            return self.dna
        elif self.encoding == 'real':
            return self.dna
        else:
            raise ValueError('Unknown encoding')

    def __repr__(self):
        return f"Individual(dna={self.dna}, fitness={self.fitness})"
