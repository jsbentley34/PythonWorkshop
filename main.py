from typing import List, Tuple

import matplotlib.pyplot as plt

from solver import Solver, Chromosome
from timeseries import holt, holt_winters, mape
from data import load_wb_xml


FORECAST_HORIZON = 10


def f(x: float) -> float:
    return (14 * x ** 2) + (6 * x) - 8


def find_root():
    """
    Find root of 14x^2 + 6x - 8
    """

    def fit(chromosome: Chromosome) -> float:
        return abs(f(chromosome[0]))

    solver = Solver(generations=10)

    model = solver.minimize(fit, [(-1000, 1000)])

    print("Example 1".center(80, "*"))
    print("f(%f)=%f" % (model[0], f(model[0])))
    print("*" * 80)


def find_roots():
    """
    Find all roots of 14x^2 + 6x - 8
    We know that this function does have exactly 2 roots in a real domains
    """

    def fit(chromosome: Chromosome) -> float:
        root1 = abs(f(chromosome[0]))
        root2 = abs(f(chromosome[1]))

        return root1 + root2 - abs(chromosome[0] - chromosome[1])

    solver = Solver(generations=150)

    model = solver.minimize(fit, [(-1000, 1000), (-1000, 1000)])

    print("Example 2".center(80, "*"))
    print("x1=%f, x2=%f" % (model[0], model[1]))
    print("f(x1)=%f, f(x2)=%f" % (f(model[0]), f(model[1])))
    print("*" * 80)


def optimize_holt(data: List[Tuple[int, float]], horizon: int):
    """
    Optimize Holt parameters

    :param data: Time series to forecast
    :param horizon: How many step forward we want to forecast
    """

    years, measurements = zip(*data)
    estimated_years = years + tuple(range(years[-1], years[-1] + horizon))

    def fit(chromosome: Chromosome) -> float:
        smoothing, _ = holt(
            measurements,
            chromosome[0],
            chromosome[1],
            chromosome[2],
            chromosome[3],
            horizon=horizon,
        )

        return mape(measurements, smoothing)

    solver = Solver()
    model = solver.minimize(fit, [(0, 1), (0, 1), (0, 100), (0, 100)])
    print("Example 3".center(80, "*"))
    print("MAPE = %f" % fit(model))
    smoothing, forecast = holt(
        measurements, model[0], model[1], model[2], model[3], horizon=horizon
    )
    plt.plot(years, measurements, label="Population, total")
    plt.plot(estimated_years, smoothing + forecast, label="Smoothing")
    plt.title(f"Population, total - Benin. Holt smoothing")
    plt.legend()

    plt.savefig("total_population.png")
    plt.clf()
    print("*" * 80)


def optimize_holt_winters(data: List[Tuple[int, float]], horizon: int):
    """
    Optimize Holt-Winters parameters

    :param data: Time series to forecast
    :param horizon: How many step forward we want to forecast
    """

    years, measurements = zip(*data)
    estimated_years = years + tuple(range(years[-1], years[-1] + horizon))

    def fit(chromosome: Chromosome) -> float:
        smoothing, _ = holt_winters(
            measurements, chromosome[0], chromosome[1], chromosome[2], 12, horizon
        )

        return mape(measurements, smoothing)

    solver = Solver()
    model = solver.minimize(fit, [(0, 1), (0, 1), (0, 1)])

    print("Example 4".center(80, "*"))
    print("MAPE: %f" % fit(model))
    smoothing, forecast = holt_winters(
        measurements, model[0], model[1], model[2], 12, horizon
    )
    plt.plot(years, measurements, label="Population, total")
    plt.plot(estimated_years, smoothing + forecast, label="Smoothing")
    plt.title("Population, total - Benin. Holt-Winters smoothing")
    plt.legend()

    plt.savefig("total_population2.png")
    plt.clf()
    print("*" * 80)


if __name__ == "__main__":
    find_root()
    find_roots()

    persistence_data = load_wb_xml(r"Persistence_to_last_grade_of_primary.xml", "WLD")

    optimize_holt(persistence_data, FORECAST_HORIZON)
    optimize_holt_winters(persistence_data, FORECAST_HORIZON)
