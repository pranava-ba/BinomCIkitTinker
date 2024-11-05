import numpy as np
import pandas as pd
from scipy.stats import binom, beta
from plotnine import ggplot, aes, geom_line, geom_point, geom_hline, labs, theme_minimal,geom_text


def covpsim(n, LL, UL, alp, s, a, b, t1, t2):
    k = n + 1
    cp = np.zeros((k, s))
    cpp = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    hp = beta.rvs(a, b, size=s)  # Generate hypothetical "p" values

    for j in range(s):
        for i in range(k):
            if LL[i] < hp[j] < UL[i]:  # Check if hp[j] is within limits
                cp[i, j] = binom.pmf(i, n, hp[j])  # Use i for binomial PMF
        cpp[j] = cp[:, j].sum()  # Calculate coverage probability
        RMSE_N1[j] = (cpp[j] - (1 - alp)) ** 2  # RMSE from nominal size
        if t1 < cpp[j] < t2:
            ctr += 1  # Count how many times cpp is within tolerance

    mcp = np.mean(cpp)  # Mean coverage probability
    micp = np.min(cpp[cpp > 0]) if np.any(cpp > 0) else 0  # Minimum coverage probability
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # RMSE calculations for mean and min coverage probabilities
    RMSE_M1 = (cpp - mcp) ** 2
    RMSE_Mi1 = (cpp - micp) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s  # Calculate tolerance percentage

    return pd.DataFrame({
        'mcp': [mcp],
        'micp': [micp],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })


def plotcovpsim(n, LL, UL, alp, s, a, b, t1, t2):
    cp = np.zeros((n + 1, s))
    cpp = np.zeros(s)

    hp = beta.rvs(a, b, size=s)  # Generate hypothetical "p"
    for j in range(s):
        for i in range(n + 1):
            if LL[i] < hp[j] < UL[i]:
                cp[i, j] = binom.pmf(i, n, hp[j])  # Use i for binomial PMF
        cpp[j] = cp[:, j].sum()

    CP = pd.DataFrame({'hp': hp, 'cpp': cpp})
    CP['mcp'] = np.mean(cpp)
    CP['micp'] = np.min(cpp[cpp > 0]) if np.any(cpp) else 0  # Minimum coverage probability

    # Plotting with plotnine
    plot = (ggplot(CP, aes(x='hp', y='cpp')) +
            geom_line(color='red') +
            geom_point(color='red') +
            geom_hline(yintercept=t1, linetype='dashed', color='red') +
            geom_hline(yintercept=t2, linetype='dashed', color='blue') +
            geom_hline(yintercept=1 - alp, linetype='dashed', color='brown') +
            geom_hline(yintercept=CP['micp'].iloc[0], linetype='dashed', color='black') +
            geom_hline(yintercept=CP['mcp'].iloc[0], linetype='dashed', color='blue') +
            labs(title='Coverage Probability using Simulation',
                 x='Hypothetical p',
                 y='Coverage Probability') +
            theme_minimal())

    plot.show()


# Define parameters
LL = np.array([0, 0.01, 0.0734, 0.18237, 0.3344, 0.5492])
UL = np.array([0.4507, 0.6655, 0.8176, 0.9265, 0.9899, 1])
n = 5
alp = 0.05
s = 5000
a = 1
b = 1
t1 = 0.93
t2 = 0.97

# Call the functions
result = covpsim(n, LL, UL, alp, s, a, b, t1, t2)
print(result)

plotcovpsim(n, LL, UL, alp, s, a, b, t1, t2)
