import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt



def fit_distribution(data, dist_name):

    if dist_name == 'fixed':
        params = np.mean(data)
        generated_values = np.array([params] * len(data))
        wass_distance = stats.wasserstein_distance(data, generated_values)

        return params, wass_distance, 'fixed'

    dist = getattr(stats, dist_name)
    
    params= dist.fit(data)
    
    generated_values = sampling_from_dist(dist, params, min(data), max(data), np.median(data), n_sample=len(data))
    wass_distance = stats.wasserstein_distance(data, generated_values)
    
    return params, wass_distance, dist


def return_best_distribution(data):

    dict_fitting_distributions = dict()
    dict_fitting_dist_params = dict()
    dict_wass = dict()

    if len(set(data)) == 1:
        return 'fixed', data.iloc[0]

    for dist_name in ['fixed', 'norm', 'expon', 'lognorm', 'uniform']:

        params, goodness_of_fit, dist = fit_distribution(data, dist_name)
        dict_fitting_distributions[dist_name] = dist
        dict_wass[dist_name] = goodness_of_fit
        dict_fitting_dist_params[dist_name] = params

    best_fit_dist_name = min(dict_wass, key=dict_wass.get)
    best_fit_dist = dict_fitting_distributions[best_fit_dist_name]
    best_fit_dist_params =  dict_fitting_dist_params[best_fit_dist_name]

    return best_fit_dist, best_fit_dist_params


def sampling_from_dist(dist, params, min_value, max_value, mean_value, n_sample=1000):

    if dist == 'fixed':
        return np.array([max_value] * n_sample)

    l = dist.rvs(*params, n_sample)
    l[l < min_value] = mean_value
    l[l > max_value] = mean_value

    return l



def plot_distribution(data, params, dist):

    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label="Data Histogram")
    
    x = np.linspace(min(data), max(data), 1000)
    y = dist.pdf(x, *params)
    
    plt.plot(x, y, 'r-', lw=2, label=f'Fitted {dist.name} Distribution')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Fitted {dist.name.capitalize()} Distribution")
    plt.show()