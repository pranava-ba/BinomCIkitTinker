import pandas as pd
from scipy.stats import norm, beta, binom
import numpy as np
from scipy import stats


def covpcwd(n, alp, c, a, b, t1, t2):
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
    p_cw = np.zeros(k)
    q_cw = np.zeros(k)
    se_cw = np.zeros(k)
    l_cw = np.zeros(k)
    u_cw = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cp_cw = np.zeros((k, s))
    ct_cw = np.zeros((k, s))
    cpp_cw = np.zeros(s)
    rmse_n1 = np.zeros(s)
    ctr = 0

    # Critical values
    cv = stats.norm.ppf(1 - (alp / 2))

    # Wald method
    for i in range(k):
        p_cw[i] = x[i] / n
        q_cw[i] = 1 - p_cw[i]
        se_cw[i] = np.sqrt(p_cw[i] * q_cw[i] / n)
        l_cw[i] = p_cw[i] - ((cv * se_cw[i]) + c)
        u_cw[i] = p_cw[i] + ((cv * se_cw[i]) + c)
        l_cw[i] = max(0, l_cw[i])
        u_cw[i] = min(1, u_cw[i])

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # Hypothetical "p"

    for j in range(s):
        for i in range(k):
            if l_cw[i] < hp[j] < u_cw[i]:
                cp_cw[i, j] = stats.binom.pmf(i, n, hp[j])
                ct_cw[i, j] = 1

        cpp_cw[j] = np.sum(cp_cw[:, j])
        rmse_n1[j] = (cpp_cw[j] - (1 - alp)) ** 2  # Root mean Square from nominal size
        if t1 < cpp_cw[j] < t2:
            ctr += 1  # tolerance for cov prob - user defined

    mcp_cw = np.mean(cpp_cw)
    micp_cw = np.min(cpp_cw)  # Mean Cov Prob
    rmse_n = np.sqrt(np.mean(rmse_n1))

    # Root mean Square from min and mean CoPr
    rmse_m1 = (cpp_cw - mcp_cw) ** 2
    rmse_mi1 = (cpp_cw - micp_cw) ** 2

    rmse_m = np.sqrt(np.mean(rmse_m1))
    rmse_mi = np.sqrt(np.mean(rmse_mi1))
    tol = 100 * ctr / s

    return pd.DataFrame({
        'mcp_cw': [mcp_cw],
        'micp_cw': [micp_cw],
        'rmse_n': [rmse_n],
        'rmse_m': [rmse_m],
        'rmse_mi': [rmse_mi],
        'tol': [tol]
    })


# Example usage:
# result = covp_cwd(n=100, alp=0.05, c=0.1, a=2, b=2, t1=0.9, t2=0.95)

def covpcas(n, alp, c, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' must be greater than 0")
    if not (0 < alp < 1):
        raise ValueError("'alpha' must be between 0 and 1")
    if c < 0:
        raise ValueError("'c' must be positive")
    if a < 0:
        raise ValueError("'a' must be non-negative")
    if b < 0:
        raise ValueError("'b' must be non-negative")
    if t1 >= t2:
        raise ValueError("t1 must be less than t2")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' must be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' must be between 0 and 1")

    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    s = 5000  # Number of simulations

    # Initialize arrays
    pCA = np.zeros(k)
    qCA = np.zeros(k)
    seCA = np.zeros(k)
    LCA = np.zeros(k)
    UCA = np.zeros(k)
    cpCA = np.zeros((k, s))
    cppCA = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # Critical values
    cv = norm.ppf(1 - (alp / 2))

    # ARC-SINE METHOD
    for i in range(k):
        pCA[i] = x[i] / n
        qCA[i] = 1 - pCA[i]
        seCA[i] = cv / np.sqrt(4 * n)
        LCA[i] = (np.sin(np.arcsin(np.sqrt(pCA[i])) - seCA[i] - c)) ** 2
        UCA[i] = (np.sin(np.arcsin(np.sqrt(pCA[i])) + seCA[i] + c)) ** 2
        LCA[i] = max(LCA[i], 0)  # Ensure lower bound is not less than 0
        UCA[i] = min(UCA[i], 1)  # Ensure upper bound is not greater than 1

    # Simulate hypothetical p values
    hp = np.sort(beta.rvs(a, b, size=s))

    # Calculate coverage probabilities
    for j in range(s):
        for i in range(k):
            if LCA[i] < hp[j] < UCA[i]:
                cpCA[i, j] = binom.pmf(i - 1, n, hp[j])
        cppCA[j] = np.sum(cpCA[:, j])
        RMSE_N1[j] = (cppCA[j] - (1 - alp)) ** 2
        if t1 < cppCA[j] < t2:
            ctr += 1

    # Calculate mean and min coverage probabilities
    mcpCA = np.mean(cppCA)  # Mean Coverage Probability
    micpCA = np.min(cppCA)

    # RMSE calculations
    RMSE_N = np.sqrt(np.mean(RMSE_N1))
    RMSE_M = np.sqrt(np.mean((cppCA - mcpCA) ** 2))
    RMSE_MI = np.sqrt(np.mean((cppCA - micpCA) ** 2))

    # Tolerance calculation
    tol = 100 * ctr / s

    # Return results as a DataFrame
    return pd.DataFrame({'mcpCA': mcpCA, 'micpCA': micpCA, 'RMSE_N': RMSE_N,
                         'RMSE_M': RMSE_M, 'RMSE_MI': RMSE_MI, 'tol': tol}, index=[0])

########################################
'''
def covpcall(n, alp, c, a, b, t1, t2):
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 < alp < 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if c <= 0 or c > (1 / (2 * n)):
        raise ValueError("'c' has to be positive and less than or equal to 1/(2*n)")
    if a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("t1 has to be lesser than t2")

    # Calling functions to get coverage probability data frames
    WaldcovpA_df = covpcwd(n, alp, c, a, b, t1, t2)
    ArcSinecovpA_df = covpcas(n, alp, c, a, b, t1, t2)
    ScorecovpA_df = covpcsc(n, alp, c, a, b, t1, t2)
    WaldLcovpA_df = covpclt(n, alp, c, a, b, t1, t2)
    AdWaldcovpA_df = covpctw(n, alp, c, a, b, t1, t2)

    # Adding method columns
    WaldcovpA_df['method'] = 'CC-Wald'
    ArcSinecovpA_df['method'] = 'CC-ArcSine'
    WaldLcovpA_df['method'] = 'CC-Logit-Wald'
    ScorecovpA_df['method'] = 'CC-Score'
    AdWaldcovpA_df['method'] = 'CC-Wald-T'

    # Creating generic data frames
    Generic_1 = pd.DataFrame({
        'method': WaldcovpA_df['method'],
        'MeanCP': WaldcovpA_df['mcpCW'],
        'MinCP': WaldcovpA_df['micpCW'],
        'RMSE_N': WaldcovpA_df['RMSE_N'],
        'RMSE_M': WaldcovpA_df['RMSE_M'],
        'RMSE_MI': WaldcovpA_df['RMSE_MI'],
        'tol': WaldcovpA_df['tol']
    })

    Generic_2 = pd.DataFrame({
        'method': ArcSinecovpA_df['method'],
        'MeanCP': ArcSinecovpA_df['mcpCA'],
        'MinCP': ArcSinecovpA_df['micpCA'],
        'RMSE_N': ArcSinecovpA_df['RMSE_N'],
        'RMSE_M': ArcSinecovpA_df['RMSE_M'],
        'RMSE_MI': ArcSinecovpA_df['RMSE_MI'],
        'tol': ArcSinecovpA_df['tol']
    })

    Generic_4 = pd.DataFrame({
        'method': ScorecovpA_df['method'],
        'MeanCP': ScorecovpA_df['mcpCS'],
        'MinCP': ScorecovpA_df['micpCS'],
        'RMSE_N': ScorecovpA_df['RMSE_N'],
        'RMSE_M': ScorecovpA_df['RMSE_M'],
        'RMSE_MI': ScorecovpA_df['RMSE_MI'],
        'tol': ScorecovpA_df['tol']
    })

    Generic_5 = pd.DataFrame({
        'method': WaldLcovpA_df['method'],
        'MeanCP': WaldLcovpA_df['mcpCLT'],
        'MinCP': WaldLcovpA_df['micpCLT'],
        'RMSE_N': WaldLcovpA_df['RMSE_N'],
        'RMSE_M': WaldLcovpA_df['RMSE_M'],
        'RMSE_MI': WaldLcovpA_df['RMSE_MI'],
        'tol': WaldLcovpA_df['tol']
    })

    Generic_6 = pd.DataFrame({
        'method': AdWaldcovpA_df['method'],
        'MeanCP': AdWaldcovpA_df['mcpCTW'],
        'MinCP': AdWaldcovpA_df['micpCTW'],
        'RMSE_N': AdWaldcovpA_df['RMSE_N'],
        'RMSE_M': AdWaldcovpA_df['RMSE_M'],
        'RMSE_MI': AdWaldcovpA_df['RMSE_MI'],
        'tol': AdWaldcovpA_df['tol']
    })

    # Combining all data frames into one final data frame
    Final_df = pd.concat([Generic_1, Generic_2, Generic_4, Generic_5, Generic_6], ignore_index=True)

    return Final_df
'''
########################################################################################

def covpcsc(n, alp, c, a, b, t1, t2):
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

    pCS = np.zeros(k)
    qCS = np.zeros(k)
    seCS_L = np.zeros(k)
    seCS_U = np.zeros(k)
    LCS = np.zeros(k)
    UCS = np.zeros(k)
    cpCS = np.zeros((k, s))
    ctCS = np.zeros((k, s))
    cppCS = np.zeros(s)

    # Critical values
    cv = stats.norm.ppf(1 - (alp / 2))
    cv1 = (cv ** 2) / (2 * n)
    cv2 = cv / (2 * n)

    # SCORE (WILSON) METHOD
    for i in range(k):
        pCS[i] = x[i] / n
        qCS[i] = 1 - pCS[i]
        seCS_L[i] = np.sqrt((cv ** 2) - (4 * n * (c + c ** 2)) + (4 * n * pCS[i] * (1 - pCS[i] + (2 * c))))
        seCS_U[i] = np.sqrt((cv ** 2) + (4 * n * (c - c ** 2)) + (4 * n * pCS[i] * (1 - pCS[i] - (2 * c))))
        LCS[i] = (n / (n + (cv) ** 2)) * ((pCS[i] - c + cv1) - (cv2 * seCS_L[i]))
        UCS[i] = (n / (n + (cv) ** 2)) * ((pCS[i] + c + cv1) + (cv2 * seCS_U[i]))
        LCS[i] = max(0, LCS[i])
        UCS[i] = min(1, UCS[i])

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))
    ctr = 0
    RMSE_N1 = np.zeros(s)

    for j in range(s):
        for i in range(k):
            if LCS[i] < hp[j] < UCS[i]:
                cpCS[i, j] = stats.binom.pmf(i, n, hp[j])
                ctCS[i, j] = 1

        cppCS[j] = np.sum(cpCS[:, j])
        RMSE_N1[j] = (cppCS[j] - (1 - alp)) ** 2
        if t1 < cppCS[j] < t2:
            ctr += 1

    mcpCS = np.mean(cppCS)
    micpCS = np.min(cppCS)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    RMSE_M1 = (cppCS - mcpCS) ** 2
    RMSE_Mi1 = (cppCS - micpCS) ** 2

    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s
    output = f"      mcpCS micpCS    RMSE_N    RMSE_M   RMSE_MI  tol\n"
    output += f"1 {mcpCS:.7f} {micpCS:.7f} {RMSE_N:.7f} {RMSE_M:.7f} {RMSE_MI:.7f} {tol:.2f}"
    return output

def covpclt(n, alp, c, a, b, t1, t2):
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
    cp_clt = np.zeros((k, s))
    ct_clt = np.zeros((k, s))
    cpp_clt = np.zeros(s)
    rmse_n1 = np.zeros(s)
    ctr = 0

    # Critical values
    cv = stats.norm.ppf(1 - (alp / 2))

    # Logit inverse function
    def lgiti(t):
        return np.exp(t) / (1 + np.exp(t))

    # LOGIT-WALD METHOD
    # For x = 0
    p_clt[0] = 0
    q_clt[0] = 1
    l_clt[0] = 0
    u_clt[0] = 1 - ((alp / 2) ** (1 / n))

    # For x = n
    p_clt[-1] = 1
    q_clt[-1] = 0
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

    for j in range(s):
        for i in range(k):
            if l_clt[i] < hp[j] < u_clt[i]:
                cp_clt[i, j] = stats.binom.pmf(i, n, hp[j])
                ct_clt[i, j] = 1

        cpp_clt[j] = np.sum(cp_clt[:, j])
        rmse_n1[j] = (cpp_clt[j] - (1 - alp)) ** 2
        if t1 < cpp_clt[j] < t2:
            ctr += 1

    # Calculate statistics
    mcp_clt = np.mean(cpp_clt)
    micp_clt = np.min(cpp_clt)
    rmse_n = np.sqrt(np.mean(rmse_n1))

    rmse_m1 = (cpp_clt - mcp_clt) ** 2
    rmse_mi1 = (cpp_clt - micp_clt) ** 2

    rmse_m = np.sqrt(np.mean(rmse_m1))
    rmse_mi = np.sqrt(np.mean(rmse_mi1))
    tol = 100 * ctr / s

    return pd.DataFrame({
        'mcp_clt': [mcp_clt],
        'micp_clt': [micp_clt],
        'rmse_n': [rmse_n],
        'rmse_m': [rmse_m],
        'rmse_mi': [rmse_mi],
        'tol': [tol]
    })


# Example usage:
# result = covp_clt(n=100, alp=0.05, c=0.1, a=2, b=2, t1=0.9, t2=0.95)


def covpctw(n, alp, c, a, b, t1, t2):
    """
    Calculate Modified t-Wald coverage probabilities for confidence intervals.

    Parameters:
    n (int): Sample size
    alp (float): Alpha level (between 0 and 1)
    c (float): Positive constant
    a (float): Beta distribution parameter a
    b (float): Beta distribution parameter b
    t1 (float): Lower tolerance bound (between 0 and 1)
    t2 (float): Upper tolerance bound (between 0 and 1)

    Returns:
    DataFrame: Coverage probability statistics
    """
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
    cp_ctw = np.zeros((k, s))
    ct_ctw = np.zeros((k, s))
    cpp_ctw = np.zeros(s)
    rmse_n1 = np.zeros(s)
    ctr = 0

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

    for j in range(s):
        for i in range(k):
            if l_ctw[i] < hp[j] < u_ctw[i]:
                cp_ctw[i, j] = stats.binom.pmf(i, n, hp[j])
                ct_ctw[i, j] = 1

        cpp_ctw[j] = np.sum(cp_ctw[:, j])
        rmse_n1[j] = (cpp_ctw[j] - (1 - alp)) ** 2
        if t1 < cpp_ctw[j] < t2:
            ctr += 1

    # Calculate statistics
    mcp_ctw = np.mean(cpp_ctw)
    micp_ctw = np.min(cpp_ctw)
    rmse_n = np.sqrt(np.mean(rmse_n1))

    rmse_m1 = (cpp_ctw - mcp_ctw) ** 2
    rmse_mi1 = (cpp_ctw - micp_ctw) ** 2

    rmse_m = np.sqrt(np.mean(rmse_m1))
    rmse_mi = np.sqrt(np.mean(rmse_mi1))
    tol = 100 * ctr / s

    return pd.DataFrame({
        'mcp_ctw': [mcp_ctw],
        'micp_ctw': [micp_ctw],
        'rmse_n': [rmse_n],
        'rmse_m': [rmse_m],
        'rmse_mi': [rmse_mi],
        'tol': [tol]
    })
n= 10; alp=0.05; c=1/(2*n); a=1;b=1; t1=0.93;t2=0.97
lemon=covpcsc(n,alp,c,a,b,t1,t2)
