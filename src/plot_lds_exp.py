import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lds_sys_id import DEFAULT_SAMPLES_PER_ITERATION

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def plot_data(seeds):
    mle_costs = []
    moment_based_costs = []

    for seed in seeds:
        mle_costs.append(np.load(f"data/mle_{seed}.npy"))
        moment_based_costs.append(np.load(f"data/moment_based_{seed}.npy"))

    mle_mean_costs, mle_std_costs = np.mean(mle_costs, axis=0), np.std(mle_costs, axis=0)
    moment_based_mean_costs, moment_based_std_costs = np.mean(moment_based_costs, axis=0), np.std(moment_based_costs, axis=0)

    best_cost = 6.60466212752298e-15

    plt.clf()
    xrange = np.arange(mle_mean_costs.shape[0]) * DEFAULT_SAMPLES_PER_ITERATION

    plt.plot(xrange, mle_mean_costs, color=CB_color_cycle[0], label="MLE")
    # plt.fill_between(xrange, mle_mean_costs - mle_std_costs, mle_mean_costs + mle_std_costs, color=CB_color_cycle[0], alpha=0.2)

    plt.plot(xrange, moment_based_mean_costs, color=CB_color_cycle[1], label="Moment based")
    # plt.fill_between(xrange, moment_based_mean_costs - moment_based_std_costs, moment_based_mean_costs + moment_based_std_costs, color=CB_color_cycle[1], alpha=0.2)

    plt.plot(xrange, [best_cost for _ in range(xrange.shape[0])], color=CB_color_cycle[2], label="Expert")

    plt.xlabel("Number of real world interactions")
    plt.ylabel("Cost of policy in real world")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig("lds_exp.png")

if __name__ == "__main__":
    seeds = np.arange(10)
    plot_data(seeds)