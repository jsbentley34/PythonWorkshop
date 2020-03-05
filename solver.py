from typing import Callable, List, Tuple
import random

Chromosome = List[float]
FitFunction = Callable[[Chromosome,], float]
Boundaries = List[Tuple[float, float]]


def draw_chromosome(bounds: Boundaries) -> Chromosome:
    """
    :param bounds: Pairs of (lower, upper) bounds
    :return: Chromosome where each gene is draw at random complying
             with given boundaries.
    """

    return [
        random.uniform(lower_bound, upper_bound) for lower_bound, upper_bound in bounds
    ]


def selection(
    fit_function: FitFunction, population: List[Chromosome], candidates: int = 10
) -> Chromosome:
    """
    Select candidate from given population using tournament method

    :param fit_function: Fir function
    :param population: Population to search
    :param candidates: Number of candidates participating in "tournament"
    :return: Best candidate from random group sampled from given population
    """

    group = random.sample(population, min(candidates, len(population)))
    return min(group, key=fit_function)


def crossover(chromosome_a: Chromosome, chromosome_b: Chromosome) -> Chromosome:
    """
    Each gene is going to be random value between (gene_a, gene_b)
    where gene_a and gene_b are same genes from same position of chromosome_a and chromosome_b

    :param chromosome_a:
    :param chromosome_b:
    :return: New chromosome with features inherited from 2 parents.
    """

    new_chromosome = []
    for gene_a, gene_b in zip(chromosome_a, chromosome_b):
        percent_a = random.random()
        percent_b = 1 - percent_a
        new_chromosome.append(percent_a * gene_a + percent_b * gene_b)
    return new_chromosome


def mutation(
    chromosome: Chromosome, boundaries: Boundaries, mutation_probability: float
) -> Chromosome:
    """
    Each mutated gene will be replaced with new one, draw randomly.

    :param chromosome: Chromosome to mutate
    :param boundaries: Boundaries to preserve
    :param mutation_probability: How often single gene is going to mutate
    :return: New, possibly changed Chromosome
    """

    new_chromosome = []
    for gene, (lower_bound, upper_bound) in zip(chromosome, boundaries):
        if random.random() < mutation_probability:
            new_chromosome.append(random.uniform(lower_bound, upper_bound))
        else:
            new_chromosome.append(gene)
    return new_chromosome


class Solver:
    def minimize(self, fit_function: FitFunction, bounds: Boundaries) -> Chromosome:
        """
        Search for coefficients that will minimize given `fit_function`

        :param fit_function: Function to minimize
        :param bounds: Boundaries to preserve
        :return: Best found solution
        """

        population = [self.draw_method(bounds) for _ in range(self.population_size)]
        survivors_number = int(self.population_size * self.elitism)

        for _ in range(self.generations):
            new_population = []
            for _ in range(survivors_number):
                parent_a = self.selection_method(fit_function, population)
                parent_b = self.selection_method(fit_function, population)

                if random.random() < self.crossover_probability:
                    child = self.crossover_method(parent_a, parent_b)
                else:
                    child = min(parent_a, parent_b, key=fit_function)

                child = self.mutation_method(child, bounds, self.mutation_probability)
                new_population.append(child)

            random_chromosomes = [
                self.draw_method(bounds)
                for _ in range(self.population_size - survivors_number)
            ]
            population = new_population + random_chromosomes
        return min(population, key=fit_function)

    def __init__(
        self,
        population_size: int = 200,
        generations: int = 10,
        mutation_probability: float = 0.05,
        crossover_probability: float = 0.8,
        elitism: float = 0.8,
        selection_method=selection,
        crossover_method=crossover,
        mutation_method=mutation,
        draw_method=draw_chromosome,
    ):
        """
        
        :param population_size: Number of chromosomes in population
        :param generations: How many iterations' system should perform
        :param mutation_probability: How often chromosome genes should mutate
        :param crossover_probability: How often child is going to inherit features of both parents
        :param elitism: What percent of population is going to survive
        :param selection_method: How crossover candidates are going to be searched
        :param crossover_method: In what way children's are going to inherit parents features
        :param mutation_method: How gene mutation will be performed
        :param draw_method: How initial chromosomes are going to be created
        """

        self.population_size = population_size
        self.generations = generations

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.elitism = elitism

        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.draw_method = draw_method
