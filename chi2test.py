from scipy import stats
from statistics import mean, stdev, StatisticsError
import math

def chisquare(obs, exp, std, ddof=1):
    length = len(obs)
    chis = sum([((obs[i] - exp[i])/(std[i]))**2 for i in range(0,length)])
    pVal = 1 - stats.chi2.cdf(chis, length-ddof if length-ddof > 1 else 1)
    return (chis, pVal)

def meanDiffInterval(xa, ya, alpha=0.05, equalVar=True, x_std=None, y_std=None):
    if x_std is None:
        x_std = stdev(xa)
    if y_std is None:
        y_std = stdev(ya)

    n_x = len(xa)
    n_y = len(ya)

    if equalVar is True:
        dof = n_x + n_y - 2
        s_p = math.sqrt(((n_x - 1) * (x_std ** 2) + (n_y - 1) * (y_std ** 2)) / dof)
        s_i = s_p * math.sqrt((1 / n_x) + (1 / n_y))
    else:
        dof = ((((x_std ** 2)/n_x) + ((y_std ** 2)/n_y)) ** 2) / (((((x_std ** 2) / n_x) ** 2) / (n_x - 1)) + ((((y_std ** 2) / n_y) ** 2) / (n_y - 1)))
        s_i = math.sqrt(((x_std ** 2) / n_x) + ((y_std ** 2) / n_y))

    x_bar = mean(xa)
    y_bar = mean(ya)

    t_value = stats.t.ppf(1-(alpha/2), dof)

    d = t_value * s_i
    m = x_bar - y_bar

    return [m, m - d, m + d]
