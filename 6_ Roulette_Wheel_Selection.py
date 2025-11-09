import random

atoms = {
    "C": 12,
    "O": 16,
    "N": 14,
    "H": 1,
    "S": 32
}

pop_size = 6             # population size
target_weight = 150      # target molecular weight
chrom_len = 10           # number of atoms per molecule
generations = 50         # number of generations
mutation_rate = 0.2      # mutation probability

# Genetic Algorithm Functions
def create_random_molecule():
    """Make a random molecule = list of atoms."""
    return [random.choice(list(atoms.keys())) for _ in range(chrom_len)]


def fitness(molecule):
    """Fitness = negative distance from target weight (closer = better)."""
    weight = sum(atoms[a] for a in molecule)
    return -abs(target_weight - weight)


def population_fitness(pop):
    """Return fitness scores for all molecules."""
    return [fitness(mol) for mol in pop]


def roulette_selection(pop, fit):
    """Roulette selection: fitter molecules have higher chance."""
    min_fit = min(fit)
    shifted = [f - min_fit + 1 for f in fit]
    return random.choices(pop, weights=shifted, k=pop_size)


def crossover(p1, p2):
    """Single-point crossover between two parents."""
    point = random.randint(1, chrom_len - 1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2


def mutate(molecule):
    """Randomly change one atom with probability mutation_rate."""
    if random.random() < mutation_rate:
        idx = random.randint(0, chrom_len - 1)
        molecule[idx] = random.choice(list(atoms.keys()))
    return molecule

# Main Genetic Algorithm
def run():
    population = [create_random_molecule() for _ in range(pop_size)]

    for g in range(generations):
        fits = population_fitness(population)
        best_idx = fits.index(max(fits))
        best_mol = population[best_idx]
        best_weight = sum(atoms[a] for a in best_mol)

        print(f"\nGeneration {g}")
        print(" Best Molecule:", "".join(best_mol))
        print(" Weight:", best_weight)
        print(" Fitness:", fits[best_idx])

        if abs(best_weight - target_weight) == 0:
            print("\nTarget weight reached!")
            break

        parents = roulette_selection(population, fits)
        new_pop = []

        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % pop_size]
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            new_pop.append(mutate(c2))

        population = new_pop

    print("\nFinal Best Molecule:", "".join(best_mol), "Weight:", best_weight)

if __name__ == "__main__":
    run()
