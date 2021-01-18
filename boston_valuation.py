from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data,
                    columns=boston_dataset.feature_names)
# data.head()
features = data.drop(['INDUS', 'AGE'], axis=1)
# features.head()

log_prices = np.log(boston_dataset.target)
# log_prices.shape
target = pd.DataFrame(log_prices, columns=['PRICE'])
print(target.shape)

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_rooms,
                     students_per_classroom,
                     next_to_river=False,
                     high_confidence=True):
    # Confidence property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0

    # Make prediction
    log_estimate = regr.predict(property_stats)

    # Clac Range
    if high_confidence:
        upper_bound = log_estimate + 2 * RMSE
        lower_bound = log_estimate - 2 * RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68

    return log_estimate, upper_bound, lower_bound, interval


def get_dollar_estimate(rm, pt_ratio, next_to_river=False, large_range=True):
    """Estimate the price of a property in Boston
    
    Keyword arguments:
    rm -- number of rooms in the property
    pt_ratio -- number of studentd per teacher in the classroom for the school in the area
    next_to_river -- True if the property is next to river, False otherwise
    large_range -- True for a 95% prediction interval, False for a 68% interval
    """
    if rm < 1 or pt_ratio < 1:
        print('That is unrealistic, try again')
        return

    log_est, upper, lower, conf = get_log_estimate(nr_rooms=3,
                                                   students_per_classroom=pt_ratio,
                                                   next_to_river=next_to_river,
                                                   high_confidence=large_range)
    # Convert to today's dollars
    dollar_est = np.e ** log_est * 1000 * SCALE_FACTOR
    dollar_high = np.e ** upper * 1000 * SCALE_FACTOR
    dollar_low = np.e ** lower * 1000 * SCALE_FACTOR

    # Rounded the dollar values to nearest thousand
    rounded_est = np.round(dollar_est, 2)
    rounded_high = np.round(dollar_high, 2)
    rounded_low = np.round(dollar_low, 2)

    print(f'The estimated property value is USD {rounded_est[0][0]}')
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {rounded_low[0][0]} at the lower end to USD {rounded_high[0][0]} at the high end.')
