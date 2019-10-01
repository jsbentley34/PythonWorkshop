from typing import Callable, List, Tuple
import random

Chromosome = List[float]
FitFunction = Callable[[Chromosome, ], float]
Boundaries = List[Tuple[float, float]]


def draw_chromosome(bounds: Boundaries) -> Chromosome:
    return [random.uniform(lower_bound, upper_bound)
            for lower_bound, upper_bound in bounds]


def selection(fit_function: FitFunction, population: List[Chromosome],
              candidates: int = 10):
    group = random.sample(population, min(candidates, len(population)))
    return min(group, key=fit_function)


def crossover(chromosome_a: Chromosome, chromosome_b: Chromosome) -> Chromosome:
    new_chromosome = []
    for gene_a, gene_b in zip(chromosome_a, chromosome_b):
        percent_a = random.random()
        percent_b = 1 - percent_a
        new_chromosome.append(percent_a * gene_a + percent_b * gene_b)
    return new_chromosome


def mutation(chromosome: Chromosome, boundaries: Boundaries,
             mutation_probability: float):
    new_chromosome = []
    for gene, (lower_bound, upper_bound) in zip(chromosome, boundaries):
        if random.random() < mutation_probability:
            new_chromosome.append(random.uniform(lower_bound, upper_bound))
        else:
            new_chromosome.append(gene)
    return new_chromosome


class Solver:
    def minimize(self, fit_function: FitFunction, bounds: Boundaries) -> Chromosome:
        population = [self.draw_method(bounds)
                      for _ in range(self.pupulation_size)]
        survivors_number = int(self.pupulation_size * self.elitarism)

        for _ in range(self.generations):
            new_population = []
            for _ in range(survivors_number):
                parent_a = self.selection_method(fit_function, population)
                parent_b = self.selection_method(fit_function, population)

                if random.random() < self.crossover_probability:
                    child = self.crossover_method(parent_a, parent_b)
                else:
                    child = min(parent_a, parent_b, key=fit_function)

                child = self.mutation_method(
                    child, bounds, self.mutation_probability)
                new_population.append(child)

            random_chromosomes = [self.draw_method(bounds)
                                  for _ in range(self.pupulation_size - survivors_number)]
            population = new_population + random_chromosomes
        return min(population, key=fit_function)

    def __init__(self, pupulation_size: int = 200,
                 generations: int = 10,
                 mutation_probability: float = 0.05,
                 crossover_probability: float = 0.8,
                 elitarism: float = 0.8,
                 selection_method=selection,
                 crossover_method=crossover,
                 mutation_method=mutation,
                 draw_method=draw_chromosome):
        self.pupulation_size = pupulation_size
        self.generations = generations

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.elitarism = elitarism

        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.draw_method = draw_method
