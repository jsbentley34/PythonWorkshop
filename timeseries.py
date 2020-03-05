"""
Given algorithms are translated from "Forecasting: Principles and Practice"
by Rob J Hyndman and George Athanasopoulos.
"""

from typing import List, Tuple


def holt(
    time_series: List[float],
    a: float,
    b: float,
    initial_level: float,
    initial_trend: float,
    horizon=10,
) -> Tuple[List[float], List[float]]:
    """
    Holt's trend method.

    :param time_series: Time series to forecast
    :param a: Level smoothing parameter.
    :param b: Trend smoothing parameter.
    :param initial_level: Initial value for level component.
    :param initial_trend: Initial value for trend component.
    :param horizon: How many step forward we want to forecast
    :return: Smoothed time series paired with forecast
    """

    smoothing = []

    prev_level = initial_level
    prev_trend = initial_trend
    for yt in time_series:
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
    """
    Mean Absolute Percentage Error

    :param observations: Observed values
    :param predictions: Predicted values
    :return: Relative error of forecasting/regression
    """

    return (
        100
        * sum(abs((y - yp) / y) for y, yp in zip(observations, predictions))
        / len(observations)
    )


def holt_winters(
    series: List[float],
    a: float,
    b: float,
    g: float,
    season_frequency: int,
    horizon: int,
) -> Tuple[List[float], List[float]]:
    """

    :param series:
    :param a: Level smoothing parameter
    :param b: Trend smoothing parameter
    :param g: Season smoothing parameter
    :param season_frequency: Frequency of season in each year.
                             season_frequency = 1 means yearly seasons when
                             season_frequency = 12 means monthly seasons
    :param horizon: How many step forward we want to forecast
    :return: Smoothed time series paired with forecast
    """

    prev_level = sum(series[:season_frequency]) / season_frequency
    prev_trend = (
        sum(series[season_frequency: season_frequency * 2])
        - sum(series[:season_frequency])
    ) / (season_frequency ** 2)
    seasons = [observation - prev_level for observation in series[:season_frequency]]

    smoothing = []
    for index, observation in enumerate(series):
        level = a * (observation - seasons[index]) + (1 - a) * (prev_level + prev_trend)
        trend = b * (level - prev_level) + (1 - b) * prev_trend
        season = g * (observation - prev_level - prev_trend) + (1 - g) * seasons[index]
        seasons.append(season)

        yp = level + trend + seasons[index]
        prev_level = level
        prev_trend = trend

        smoothing.append(yp)

    forecast = []
    for h in range(horizon):
        prev_season_i = -season_frequency + (h % season_frequency)
        yp = prev_level + h * prev_trend + seasons[prev_season_i]
        forecast.append(yp)

    return smoothing, forecast
