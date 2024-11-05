import numpy as np
import pandas as pd
from scipy import stats
from plotnine import (ggplot, aes, labs, geom_line, geom_hline,
                      geom_text, guides, guide_legend, theme_gray,
                      scale_color_manual, theme, element_text)

from coverageprob_cc_all_221 import *
def plotcovpcwd(n, alp, c, a, b, t1, t2):
    # Input validation
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not 0 <= alp <= 1:
        raise ValueError("'alpha' has to be between 0 and 1")
    if not isinstance(c, (int, float)) or c < 0:
        raise ValueError("'c' has to be positive")
    if not isinstance(a, (int, float)) or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 >= t2:
        raise ValueError("t1 has to be lesser than t2")
    if not 0 <= t1 <= 1:
        raise ValueError("'t1' has to be between 0 and 1")
    if not 0 <= t2 <= 1:
        raise ValueError("'t2' has to be between 0 and 1")

    # Generate hypothetical p values
    s = 5000
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    # Calculate coverage probabilities
    x = np.arange(n + 1)
    k = n + 1
    cv = stats.norm.ppf(1 - (alp / 2))

    # Initialize arrays
    cpp_values = np.zeros(s)

    # Calculate Wald intervals and coverage probabilities
    for j in range(s):
        cp_sum = 0
        for i in range(k):
            p_hat = i / n
            se = np.sqrt(p_hat * (1 - p_hat) / n)
            l = max(0, p_hat - (cv * se + c))
            u = min(1, p_hat + (cv * se + c))

            if l <= hp[j] <= u:
                cp_sum += stats.binom.pmf(i, n, hp[j])

        cpp_values[j] = cp_sum

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'hp': hp,
        'cp': cpp_values,
        'method': 'Continuity corrected Wald'
    })

    # Calculate statistics
    mcp = np.mean(cpp_values)
    micp = np.min(cpp_values)

    # Create the plot
    plot = (ggplot(plot_data, aes(x='hp', y='cp'))
            + labs(title='Coverage Probability of Continuity corrected Wald method',
                   y='Coverage Probability',
                   x='p')
            + geom_line(aes(color='method'), size=1)
            + geom_hline(yintercept=mcp, color='#00FF00', size=1)
            + geom_hline(yintercept=micp, color='#0000FF', size=1)
            + geom_hline(yintercept=t1, color='red', linetype='dashed', size=1)
            + geom_hline(yintercept=t2, color='blue', linetype='dashed', size=1)
            + geom_text(aes(x=0, y=t1 + 0.02),
                        label='Lower tolerance(t1)',
                        color='red',
                        ha='left',
                        size=8)
            + geom_text(aes(x=0, y=t2 + 0.02),
                        label='Higher tolerance(t2)',
                        color='blue',
                        ha='left',
                        size=8)
            + scale_color_manual(values={'Continuity corrected Wald': '#FF0000'},
                                 name='Heading')
            + theme_gray()
            + theme(
                plot_title=element_text(size=12),
                axis_title=element_text(size=10),
                legend_title=element_text(size=10),
                legend_text=element_text(size=8)
            )
            )

    plot.show()


def plotcovpcas(n, alp, c, a, b, t1, t2):
    # Input validation
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not 0 <= alp <= 1:
        raise ValueError("'alpha' has to be between 0 and 1")
    if not isinstance(c, (int, float)) or c < 0:
        raise ValueError("'c' has to be positive")
    if not isinstance(a, (int, float)) or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 >= t2:
        raise ValueError("t1 has to be lesser than t2")
    if not 0 <= t1 <= 1:
        raise ValueError("'t1' has to be between 0 and 1")
    if not 0 <= t2 <= 1:
        raise ValueError("'t2' has to be between 0 and 1")

    # Generate hypothetical p values
    s = 5000
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    # Calculate coverage probabilities
    x = np.arange(n + 1)
    k = n + 1
    cv = stats.norm.ppf(1 - (alp / 2))

    # Initialize arrays
    cpp_values = np.zeros(s)

    # Calculate ArcSine intervals and coverage probabilities
    for j in range(s):
        cp_sum = 0
        for i in range(k):
            p_hat = i / n
            se = cv / np.sqrt(4 * n)
            l = (np.sin(np.arcsin(np.sqrt(p_hat)) - se - c)) ** 2
            u = (np.sin(np.arcsin(np.sqrt(p_hat)) + se + c)) ** 2
            l = max(0, l)
            u = min(1, u)

            if l <= hp[j] <= u:
                cp_sum += stats.binom.pmf(i, n, hp[j])

        cpp_values[j] = cp_sum

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'hp': hp,
        'cp': cpp_values,
        'method': 'Continuity corrected ArcSine'
    })

    # Calculate statistics
    mcp = np.mean(cpp_values)
    micp = np.min(cpp_values)

    # Calculate midpoint between t1 and t2
    midpoint = (t1 + t2) / 2

    # Create the plot
    plot = (ggplot(plot_data, aes(x='hp', y='cp'))
            + labs(title='Coverage Probability of Continuity corrected ArcSine method',
                   y='Coverage Probability',
                   x='p')
            + geom_line(aes(color='method'), size=1)
            + geom_hline(yintercept=mcp, color='#00FF00', size=1)
            + geom_hline(yintercept=micp, color='#0000FF', size=1)
            + geom_hline(yintercept=t1, color='red', linetype='dashed', size=1)
            + geom_hline(yintercept=t2, color='blue', linetype='dashed', size=1)
            + geom_hline(yintercept=midpoint, color='black', linetype='dashed', size=1)  # Added middle dashed line
            + geom_text(aes(x=0, y=t1 + 0.02),
                        label='Lower tolerance(t1)',
                        color='red',
                        ha='left',
                        size=8)
            + geom_text(aes(x=0, y=t2 + 0.02),
                        label='Higher tolerance(t2)',
                        color='blue',
                        ha='left',
                        size=8)
            + scale_color_manual(values={'Continuity corrected ArcSine': '#FF0000'},
                                 name='Heading')
            + theme_gray()
            + theme(
                plot_title=element_text(size=12),
                axis_title=element_text(size=10),
                legend_title=element_text(size=10),
                legend_text=element_text(size=8)
            )
            )

    plot.show()


import numpy as np
from scipy import stats
import pandas as pd
from plotnine import *


def plotcovpcsc(n, alp, c, a, b, t1, t2):
    # Input validation
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not 0 <= alp <= 1:
        raise ValueError("'alpha' has to be between 0 and 1")
    if c <= 0 or c > 1 / (2 * n):
        raise ValueError("'c' has to be positive and less than or equal to 1/(2*n)")
    if not isinstance(a, (int, float)) or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("t1 has to be lesser than t2")
    if not 0 <= t1 <= 1:
        raise ValueError("'t1' has to be between 0 and 1")
    if not 0 <= t2 <= 1:
        raise ValueError("'t2' has to be between 0 and 1")

    # Initializations
    x = np.arange(n + 1)
    k = n + 1
    s = 5000  # Simulation run to generate hypothetical p

    # Critical values
    cv = stats.norm.ppf(1 - (alp / 2))
    cv1 = (cv ** 2) / (2 * n)
    cv2 = cv / (2 * n)

    # SCORE (WILSON) METHOD calculations
    pCS = x / n
    qCS = 1 - pCS
    seCS_L = np.sqrt((cv ** 2) - (4 * n * (c + c ** 2)) + (4 * n * pCS * (1 - pCS + (2 * c))))
    seCS_U = np.sqrt((cv ** 2) + (4 * n * (c - c ** 2)) + (4 * n * pCS * (1 - pCS - (2 * c))))
    LCS = (n / (n + cv ** 2)) * ((pCS - c + cv1) - (cv2 * seCS_L))
    UCS = (n / (n + cv ** 2)) * ((pCS + c + cv1) + (cv2 * seCS_U))
    LCS = np.maximum(0, LCS)
    UCS = np.minimum(1, UCS)

    # Generate hypothetical p values and calculate coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))
    cpp_values = np.zeros(s)

    for j in range(s):
        cp_sum = 0
        for i in range(k):
            if LCS[i] <= hp[j] <= UCS[i]:
                cp_sum += stats.binom.pmf(i, n, hp[j])
        cpp_values[j] = cp_sum

    # Calculate statistics
    mcp = np.mean(cpp_values)
    micp = np.min(cpp_values)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'hp': hp,
        'cp': cpp_values,
        'method': 'Score (Wilson)'
    })

    # Calculate midpoint between t1 and t2
    midpoint = (t1 + t2) / 2

    # Create the plot
    plot = (ggplot(plot_data, aes(x='hp', y='cp'))
            + labs(title='Coverage Probability of Score (Wilson) method',
                   y='Coverage Probability',
                   x='p')
            + geom_line(aes(color='method'), size=1)
            + geom_hline(yintercept=mcp, color='#00FF00', size=1)
            + geom_hline(yintercept=micp, color='#0000FF', size=1)
            + geom_hline(yintercept=t1, color='red', linetype='dashed', size=1)
            + geom_hline(yintercept=t2, color='blue', linetype='dashed', size=1)
            + geom_hline(yintercept=midpoint, color='black', linetype='dashed', size=1)
            + geom_text(aes(x=0, y=t1 + 0.02),
                        label='Lower tolerance(t1)',
                        color='red',
                        ha='left',
                        size=8)
            + geom_text(aes(x=0, y=t2 + 0.02),
                        label='Higher tolerance(t2)',
                        color='blue',
                        ha='left',
                        size=8)
            + scale_color_manual(values={'Score (Wilson)': '#FF0000'},
                                 name='Method')
            + theme_gray()
            + theme(
                plot_title=element_text(size=12),
                axis_title=element_text(size=10),
                legend_title=element_text(size=10),
                legend_text=element_text(size=8)
            ))

    plot.show()

import numpy as np
import pandas as pd
from scipy import stats
from plotnine import ggplot, aes, geom_line, geom_hline, geom_text, labs, theme_gray, theme, element_text, scale_color_manual

def plotcovpclt(n, alp, c, a, b, t1, t2):
    # Input validation
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not 0 <= alp <= 1:
        raise ValueError("'alpha' has to be between 0 and 1")
    if not isinstance(c, (int, float)) or c < 0:
        raise ValueError("'c' has to be positive")
    if not isinstance(a, (int, float)) or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 >= t2:
        raise ValueError("t1 has to be lesser than t2")
    if not 0 <= t1 <= 1:
        raise ValueError("'t1' has to be between 0 and 1")
    if not 0 <= t2 <= 1:
        raise ValueError("'t2' has to be between 0 and 1")

    # Input n
    x = np.arange(n + 1)
    k = n + 1

    # Initializations
    p_clt = np.zeros(k)
    q_clt = np.zeros(k)
    se_clt = np.zeros(k)
    lgit = np.zeros(k)
    l_clt = np.zeros(k)
    u_clt = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p

    # Critical values
    cv = stats.norm.ppf(1 - (alp / 2))

    # Logit inverse function
    def lgiti(t):
        return np.exp(t) / (1 + np.exp(t))

    # LOGIT-WALD METHOD
    # For x = 0
    l_clt[0] = 0
    u_clt[0] = 1 - ((alp / 2) ** (1 / n))

    # For x = n
    l_clt[-1] = (alp / 2) ** (1 / n)
    u_clt[-1] = 1

    # For x = 1 to n-1
    for j in range(1, k - 1):
        p_clt[j] = x[j] / n
        q_clt[j] = 1 - p_clt[j]
        lgit[j] = np.log(p_clt[j] / q_clt[j])
        se_clt[j] = np.sqrt(p_clt[j] * q_clt[j] * n)
        l_clt[j] = lgiti(lgit[j] - (cv / se_clt[j]) - c)
        u_clt[j] = lgiti(lgit[j] + (cv / se_clt[j]) + c)

    # Bound adjustment
    l_clt = np.clip(l_clt, 0, 1)
    u_clt = np.clip(u_clt, 0, 1)

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # Hypothetical "p"
    cpp_values = np.zeros(s)

    for j in range(s):
        cp_sum = 0
        for i in range(k):
            if l_clt[i] < hp[j] < u_clt[i]:
                cp_sum += stats.binom.pmf(i, n, hp[j])
        cpp_values[j] = cp_sum

    # Calculate statistics
    mcp = np.mean(cpp_values)
    micp = np.min(cpp_values)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'hp': hp,
        'cp': cpp_values,
        'method': 'Logit-Wald'
    })

    # Calculate midpoint between t1 and t2
    midpoint = (t1 + t2) / 2

    # Create the plot
    plot = (ggplot(plot_data, aes(x='hp', y='cp'))
            + labs(title='Coverage Probability of Logit-Wald method',
                   y='Coverage Probability',
                   x='p')
            + geom_line(aes(color='method'), size=1)
            + geom_hline(yintercept=mcp, color='#00FF00', size=1)
            + geom_hline(yintercept=micp, color='#0000FF', size=1)
            + geom_hline(yintercept=t1, color='red', linetype='dashed', size=1)
            + geom_hline(yintercept=t2, color='blue', linetype='dashed', size=1)
            + geom_hline(yintercept=midpoint, color='black', linetype='dashed', size=1)
            + geom_text(aes(x=0, y=t1 + 0.02),
                        label='Lower tolerance(t1)',
                        color='red',
                        ha='left',
                        size=8)
            + geom_text(aes(x=0, y=t2 + 0.02),
                        label='Higher tolerance(t2)',
                        color='blue',
                        ha='left',
                        size=8)
            + scale_color_manual(values={'Logit-Wald': '#FF0000'},
                                 name='Method')
            + theme_gray()
            + theme(
                plot_title=element_text(size=12),
                axis_title=element_text(size=10),
                legend_title=element_text(size=10),
                legend_text=element_text(size=8)
            ))

    plot.show()
import numpy as np
import pandas as pd
from scipy import stats
from plotnine import ggplot, aes, geom_line, geom_hline, geom_text, labs, theme_gray, theme, element_text, scale_color_manual

def plotcovpctw(n, alp, c, a, b, t1, t2):
    # Input validation
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not 0 <= alp <= 1:
        raise ValueError("'alpha' has to be between 0 and 1")
    if not isinstance(c, (int, float)) or c < 0:
        raise ValueError("'c' has to be positive")
    if not isinstance(a, (int, float)) or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 >= t2:
        raise ValueError("t1 has to be lesser than t2")
    if not 0 <= t1 <= 1:
        raise ValueError("'t1' has to be between 0 and 1")
    if not 0 <= t2 <= 1:
        raise ValueError("'t2' has to be between 0 and 1")

    # Input n
    x = np.arange(n + 1)
    k = n + 1

    # Initializations
    p_ctw = np.zeros(k)
    q_ctw = np.zeros(k)
    se_ctw = np.zeros(k)
    l_ctw = np.zeros(k)
    u_ctw = np.zeros(k)
    dof = np.zeros(k)
    cv = np.zeros(k)
    s = 5000

    # Define helper functions
    def f1(p, n):
        return p * (1 - p) / n

    def f2(p, n):
        term1 = (p * (1 - p)) / (n ** 3)
        term2 = (p + (6 * n - 7) * (p ** 2) + (4 * (n - 1) * (n - 3) * (p ** 3)) -
                 (2 * (n - 1) * ((2 * n) - 3) * (p ** 4))) / (n ** 5)
        term3 = -2 * (p + ((2 * n) - 3) * (p ** 2) - 2 * (n - 1) * (p ** 3)) / (n ** 4)
        return term1 + term2 + term3

    # MODIFIED t-WALD METHOD
    for i in range(k):
        # Modified probability calculation for edge cases
        if x[i] == 0 or x[i] == n:
            p_ctw[i] = (x[i] + 2) / (n + 4)
        else:
            p_ctw[i] = x[i] / n

        q_ctw[i] = 1 - p_ctw[i]

        # Calculate degrees of freedom and critical value
        dof[i] = 2 * (f1(p_ctw[i], n) ** 2) / f2(p_ctw[i], n)
        cv[i] = stats.t.ppf(1 - (alp / 2), df=dof[i])

        # Calculate confidence intervals
        se_ctw[i] = cv[i] * np.sqrt(f1(p_ctw[i], n))
        l_ctw[i] = p_ctw[i] - (se_ctw[i] + c)
        u_ctw[i] = p_ctw[i] + (se_ctw[i] + c)

        # Bound adjustments
        l_ctw[i] = max(0, l_ctw[i])
        u_ctw[i] = min(1, u_ctw[i])

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # Hypothetical "p"
    cpp_values = np.zeros(s)

    for j in range(s):
        cp_sum = 0
        for i in range(k):
            if l_ctw[i] < hp[j] < u_ctw[i]:
                cp_sum += stats.binom.pmf(i, n, hp[j])
        cpp_values[j] = cp_sum

    # Calculate statistics
    mcp = np.mean(cpp_values)
    micp = np.min(cpp_values)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'hp': hp,
        'cp': cpp_values,
        'method': 'Modified t-Wald'
    })

    # Calculate midpoint between t1 and t2
    midpoint = (t1 + t2) / 2

    # Create the plot
    plot = (ggplot(plot_data, aes(x='hp', y='cp'))
            + labs(title='Coverage Probability of Modified t-Wald method',
                   y='Coverage Probability',
                   x='p')
            + geom_line(aes(color='method'), size=1)
            + geom_hline(yintercept=mcp, color='#00FF00', size=1)
            + geom_hline(yintercept=micp, color='#0000FF', size=1)
            + geom_hline(yintercept=t1, color='red', linetype='dashed', size=1)
            + geom_hline(yintercept=t2, color='blue', linetype='dashed', size=1)
            + geom_hline(yintercept=midpoint, color='black', linetype='dashed', size=1)
            + geom_text(aes(x=0, y=t1 + 0.02),
                        label='Lower tolerance(t1)',
                        color='red',
                        ha='left',
                        size=8)
            + geom_text(aes(x=0, y=t2 + 0.02),
                        label='Higher tolerance(t2)',
                        color='blue',
                        ha='left',
                        size=8)
            + scale_color_manual(values={'Modified t-Wald': '#FF0000'},
                                 name='Method')
            + theme_gray()
            + theme(
                plot_title=element_text(size=12),
                axis_title=element_text(size=10),
                legend_title=element_text(size=10),
                legend_text=element_text(size=8)
            ))

    plot.show()


#hi siddesh

n= 10; alp=0.05; c=1/(2*n);a=1;b=1; t1=0.93;t2=0.97
plotcovpclt(n, alp, c, a, b, t1, t2)