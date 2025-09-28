# Genetic Algorithm Project

This project implements a simple genetic algorithm with the following features:

- Binary and real encoding for individuals
- Configurable hyperparameters (population size, generations, etc.)
- Multiple selection methods: elitist, roulette, ranking (v1/v2), tournament, steady-state, crowding factor
- Multiple crossover methods: uniform, arithmetic, flat, BLX-α, PMX (for permutations)
- Multiple mutation methods: random, uniform, non-uniform, swap
- Fitness function for n alleles
- Generation results table (Individual, Mapping, DNA, Fitness)
- Visualization: best and average fitness per generation

## Usage

1. Configure parameters in `main.py`.
2. Run the project: `python main.py`
3. View results and evolution graphs.

## Structure
- `src/individual.py`: Individual representation
- `src/population.py`: Population management
- `src/selection.py`: Selection methods
- `src/crossover.py`: Crossover methods
- `src/mutation.py`: Mutation methods
- `src/fitness.py`: Fitness function
- `src/visualization.py`: Plotting results
- `src/main.py`: Main script

## Requirements
- Python 3.8+
- matplotlib
- numpy

Install dependencies:
```
pip install matplotlib numpy
```
