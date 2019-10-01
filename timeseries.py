from typing import List


def holt(timeseries, a, b, initial_level, initial_trend, horizon=10):
    """Holt's trend method.

    Args:
        a: level smoothing parameter.
        b: trend smoothing parameter.
        initial_level: initial value for level component.
        initial_trend: initial value for trend component.

    Returns:
        Pair of (smoothing, forecast)
    """
    smoothing = []

    prev_level = initial_level
    prev_trend = initial_trend
    for yt in timeseries:
        yp = prev_level + prev_trend
        level = a * yt + (1 - a) * (prev_level + prev_trend)
        trend = b * (level - prev_level) + (1 - b) * prev_trend

        prev_trend = trend
        prev_level = level

        smoothing.append(yp)

    forecast = []
    for h in range(1, horizon + 1):
        forecast.append(prev_level + h * prev_trend)

    return smoothing, forecast


def mape(observations: List[float], predictions: List[float]) -> float:
    return 100 * sum(abs((y - yp) / y)
                     for y, yp in zip(observations, predictions)) / len(observations)


def holt_winters(series, a, b, g, season_length, horizon):
    prev_level = sum(series[:season_length]) / season_length
    prev_trend = (sum(series[season_length:season_length*2]) -
                  sum(series[:season_length])) / (season_length ** 2)
    seasons = [observation -
               prev_level for observation in series[:season_length]]

    index = 0
    smoothing = []
    for index, observation in enumerate(series):
        level = a * (observation - seasons[index]) + \
            (1 - a) * (prev_level + prev_trend)
        trend = b * (level - prev_level) + (1 - b) * prev_trend
        season = g * (observation - prev_level - prev_trend) + \
            (1 - g) * seasons[index]
        seasons.append(season)

        yp = level + trend + seasons[index]
        prev_level = level
        prev_trend = trend

        smoothing.append(yp)

    forecast = []
    for h in range(horizon):
        prev_season_i = - season_length + (h % season_length)
        yp = prev_level + h * prev_trend + seasons[prev_season_i]
        forecast.append(yp)

    return smoothing, forecast
