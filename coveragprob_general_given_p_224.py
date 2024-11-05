
from scipy.stats import binom, beta
from plotnine import ggplot, aes, geom_line, geom_point, geom_hline, labs, theme
import numpy as np
import pandas as pd
from scipy.stats import binom

def covpgen(n, LL, UL, alp, hp, t1, t2):
    # Check for missing arguments
    if n is None:
        raise ValueError("'n' is missing")
    if LL is None:
        raise ValueError("'Lower limit' is missing")
    if UL is None:
        raise ValueError("'Upper Limit' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if hp is None:
        raise ValueError("'hp' is missing")
    if t1 is None:
        raise ValueError("'t1' is missing")
    if t2 is None:
        raise ValueError("'t2' is missing")

    # Validate inputs
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if any(x < 0 for x in LL):
        raise ValueError("'LL' has to be a set of positive numeric vectors")
    if any(x < 0 for x in UL):
        raise ValueError("'UL' has to be a set of positive numeric vectors")
    if len(LL) <= n:
        raise ValueError("Length of vector LL has to be greater than n")
    if len(UL) <= n:
        raise ValueError("Length of vector UL has to be greater than n")
    if any(LL[i] > UL[i] for i in range(n + 1)):
        raise ValueError("LL value must be lower than the corresponding UL value")
    if alp > 1 or alp < 0:
        raise ValueError("'alpha' has to be between 0 and 1")
    if any(x > 1 or x < 0 for x in hp):
        raise ValueError("'hp' has to be between 0 and 1")
    if t1 > t2:
        raise ValueError("t1 has to be less than t2")
    if t1 < 0 or t1 > 1:
        raise ValueError("'t1' has to be between 0 and 1")
    if t2 < 0 or t2 > 1:
        raise ValueError("'t2' has to be between 0 and 1")

    k = n + 1
    s = len(hp)
    cp = np.zeros((k, s))
    cpp = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # Coverage probabilities
    for j in range(s):
        for i in range(k):
            if LL[i] < hp[j] < UL[i]:
                cp[i, j] = binom.pmf(i, n, hp[j])

        cpp[j] = cp[:, j].sum()
        RMSE_N1[j] = (cpp[j] - (1 - alp)) ** 2  # Root mean square from nominal size
        if t1 < cpp[j] < t2:
            ctr += 1

    mcp = cpp.mean()
    micp = cpp.min()  # Mean Coverage Probability
    RMSE_N = np.sqrt(RMSE_N1.mean())

    # Root mean square from min and mean CoPr
    RMSE_M1 = (cpp - mcp) ** 2
    RMSE_Mi1 = (cpp - micp) ** 2
    RMSE_M = np.sqrt(RMSE_M1.mean())
    RMSE_MI = np.sqrt(RMSE_Mi1.mean())
    tol = 100 * ctr / s

    return pd.DataFrame({
        'mcp': [mcp],
        'micp': [micp],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })

def plotcovpgen(n, LL, UL, alp, hp, t1, t2):
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if any(ll < 0 for ll in LL):
        raise ValueError("'LL' has to be a set of positive numeric vectors")
    if any(ul < 0 for ul in UL):
        raise ValueError("'UL' has to be a set of positive numeric vectors")
    if len(LL) <= n:
        raise ValueError("Length of vector LL has to be greater than n")
    if len(UL) <= n:
        raise ValueError("Length of vector UL has to be greater than n")
    if any(LL[i] > UL[i] for i in range(len(LL))):
        raise ValueError("LL values have to be lower than the corresponding UL values")
    if alp < 0 or alp > 1:
        raise ValueError("'alpha' has to be between 0 and 1")
    if any(h < 0 or h > 1 for h in hp):
        raise ValueError("'hp' has to be between 0 and 1")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than 't2'")
    if t1 < 0 or t1 > 1:
        raise ValueError("'t1' has to be between 0 and 1")
    if t2 < 0 or t2 > 1:
        raise ValueError("'t2' has to be between 0 and 1")

    k = n + 1
    s = len(hp)
    cp = np.zeros((k, s))
    cpp = np.zeros(s)
    ctr = 0

    # Calculate coverage probabilities
    for j in range(s):
        for i in range(k):
            if LL[i] < hp[j] < UL[i]:
                cp[i, j] = binom.pmf(i, n, hp[j])
        cpp[j] = np.sum(cp[:, j])
        if t1 < cpp[j] < t2:
            ctr += 1  # tolerance for coverage probability

    mcp = np.mean(cpp)  # Mean Coverage Probability
    micp = np.min(cpp[cpp > 0]) if np.any(cpp > 0) else 0  # Minimum Coverage Probability

    # Create a DataFrame for plotting
    CP = pd.DataFrame({'hp': hp, 'cpp': cpp})

    # Plotting with plotnine
    plot = (ggplot(CP, aes(x='hp', y='cpp')) +
            labs(x='p', y='Coverage Probability', title='Coverage Probability - General Method') +
            geom_line(color='red') +
            geom_point(color='red') +
            geom_hline(yintercept=t1, color='red', linetype='dashed') +
            geom_hline(yintercept=t2, color='blue', linetype='dashed') +
            geom_hline(yintercept=1 - alp, color='brown', linetype='dashed') +
            geom_hline(yintercept=micp, color='black', linetype='dashed') +
            geom_hline(yintercept=mcp, color='blue', linetype='dashed') +

            theme(legend_position='none'))


    plot.show()

# Parameters
LL = np.array([0, 0.01, 0.0734, 0.18237, 0.3344, 0.5492])
UL = np.array([0.4507, 0.6655, 0.8176, 0.9265, 0.9899, 1])
n = 5
alp = 0.05
s = 5000
a = 1
b = 1
t1 = 0.93
t2 = 0.97

# Call the function and print results


# Generate hypothetical values for plotting
hp = np.sort(beta.rvs(a, b, size=s))

# Plot the coverage probabilities
plotcovpgen(n, LL, UL, alp, hp, t1, t2)
