import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

def covpAWD(n, alp, h, a, b, t1, t2):
    # Error checks
    if n <= 0 or not isinstance(n, (int, float)):
        raise ValueError("'n' must be a positive number")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' must be between 0 and 1")
    if h < 0 or not isinstance(h, (int, float)):
        raise ValueError("'h' must be non-negative")
    if a < 0 or not isinstance(a, (int, float)):
        raise ValueError("'a' must be non-negative")
    if b < 0 or not isinstance(b, (int, float)):
        raise ValueError("'b' must be non-negative")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' must be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' must be between 0 and 1")
    if t1 > t2:
        raise ValueError("'t1' must be less than 't2'")

    # Initial parameters
    x = np.arange(0, n+1)
    k = n + 1
    y = x + h
    n1 = n + (2 * h)

    # Initialize arrays
    pAW = np.zeros(k)
    qAW = np.zeros(k)
    seAW = np.zeros(k)
    LAW = np.zeros(k)
    UAW = np.zeros(k)
    
    s = 5000  # Number of simulation runs
    cpAW = np.zeros((k, s))
    cppAW = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0
    
    # Critical value for Wald confidence interval
    cv = stats.norm.ppf(1 - (alp / 2))

    # Wald method for each value of x
    for i in range(k):
        pAW[i] = y[i] / n1
        qAW[i] = 1 - pAW[i]
        seAW[i] = np.sqrt(pAW[i] * qAW[i] / n1)
        LAW[i] = pAW[i] - (cv * seAW[i])
        UAW[i] = pAW[i] + (cv * seAW[i])
        
        # Ensure bounds are within [0, 1]
        LAW[i] = max(LAW[i], 0)
        UAW[i] = min(UAW[i], 1)

    # Hypothetical values of p from Beta distribution
    hp = np.sort(np.random.beta(a, b, s))

    # Coverage probabilities calculation
    for j in range(s):
        for i in range(k):
            if LAW[i] < hp[j] < UAW[i]:
                cpAW[i, j] = stats.binom.pmf(i, n, hp[j])
        
        cppAW[j] = np.sum(cpAW[:, j])
        RMSE_N1[j] = (cppAW[j] - (1 - alp)) ** 2  # Root mean square from nominal size
        if t1 < cppAW[j] < t2:
            ctr += 1  # Coverage tolerance

    # Mean and minimum coverage probability
    mcpAW = np.mean(cppAW)
    micpAW = np.min(cppAW)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from mean and minimum CoPr
    RMSE_M1 = (cppAW - mcpAW) ** 2
    RMSE_Mi1 = (cppAW - micpAW) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))

    # Tolerance percentage
    tol = 100 * ctr / s

    return pd.DataFrame({
        'mcpAW': [mcpAW],
        'micpAW': [micpAW],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })




def covpASC(n, alp, h, a, b, t1, t2):
    # Error checks
    if n <= 0 or not isinstance(n, (int, float)):
        raise ValueError("'n' must be a positive number")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' must be between 0 and 1")
    if h < 0 or not isinstance(h, (int, float)):
        raise ValueError("'h' must be non-negative")
    if a < 0 or not isinstance(a, (int, float)):
        raise ValueError("'a' must be non-negative")
    if b < 0 or not isinstance(b, (int, float)):
        raise ValueError("'b' must be non-negative")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' must be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' must be between 0 and 1")
    if t1 > t2:
        raise ValueError("'t1' must be less than 't2'")

    # Initial parameters
    x = np.arange(0, n+1)
    k = n + 1
    y = x + h
    n1 = n + (2 * h)

    # Initialize arrays
    pAS = np.zeros(k)
    qAS = np.zeros(k)
    seAS = np.zeros(k)
    LAS = np.zeros(k)
    UAS = np.zeros(k)

    s = 1000  # Number of simulation runs
    cpAS = np.zeros((k, s))
    cppAS = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # Critical values for the Score (Wilson) method
    cv = stats.norm.ppf(1 - (alp / 2))
    cv1 = (cv**2) / (2 * n1)
    cv2 = (cv / (2 * n1))**2

    # Score (Wilson) method for each value of x
    for i in range(k):
        pAS[i] = y[i] / n1
        qAS[i] = 1 - pAS[i]
        seAS[i] = np.sqrt((pAS[i] * qAS[i] / n1) + cv2)
        LAS[i] = (n1 / (n1 + (cv**2))) * ((pAS[i] + cv1) - (cv * seAS[i]))
        UAS[i] = (n1 / (n1 + (cv**2))) * ((pAS[i] + cv1) + (cv * seAS[i]))
        
        # Ensure bounds are within [0, 1]
        LAS[i] = max(LAS[i], 0)
        UAS[i] = min(UAS[i], 1)

    # Hypothetical values of p from Beta distribution
    hp = np.sort(np.random.beta(a, b, s))

    # Coverage probabilities calculation
    for j in range(s):
        for i in range(k):
            if LAS[i] < hp[j] < UAS[i]:
                cpAS[i, j] = stats.binom.pmf(i, n, hp[j])
        
        cppAS[j] = np.sum(cpAS[:, j])
        RMSE_N1[j] = (cppAS[j] - (1 - alp)) ** 2  # Root mean square from nominal size
        if t1 < cppAS[j] < t2:
            ctr += 1  # Coverage tolerance

    # Mean and minimum coverage probability
    mcpAS = np.mean(cppAS)
    micpAS = np.min(cppAS)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from mean and minimum CoPr
    RMSE_M1 = (cppAS - mcpAS) ** 2
    RMSE_Mi1 = (cppAS - micpAS) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))

    # Tolerance percentage
    tol = 100 * ctr / s

    return pd.DataFrame({
        'mcpAS': [mcpAS],
        'micpAS': [micpAS],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI':[RMSE_MI],
        'tol': [tol]
    })






def covpAAS(n, alp, h, a, b, t1, t2):
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if not isinstance(h, (int, float)) or h < 0:
        raise ValueError("'h' has to be greater than or equal to 0")
    if not isinstance(a, (int, float)) or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' and 't2' have to be between 0 and 1")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than 't2'")

    # Input n
    x = np.arange(n+1)
    k = n + 1
    y = x + h
    n1 = n + 2 * h

    # Initializations
    pAA = np.zeros(k)
    qAA = np.zeros(k)
    seAA = np.zeros(k)
    LAA = np.zeros(k)
    UAA = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpAA = np.zeros((k, s))
    cppAA = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    RMSE_M1 = np.zeros(s)
    RMSE_Mi1 = np.zeros(s)
    ctr = 0

    # Critical values
    cv = stats.norm.ppf(1 - alp / 2)
    
    # Adjusted Arc-Sine Method
    for i in range(k):
        pAA[i] = y[i] / n1
        qAA[i] = 1 - pAA[i]
        seAA[i] = cv / np.sqrt(4 * n1)
        LAA[i] = np.sin(np.arcsin(np.sqrt(pAA[i])) - seAA[i])**2
        UAA[i] = np.sin(np.arcsin(np.sqrt(pAA[i])) + seAA[i])**2
        LAA[i] = max(LAA[i], 0)
        UAA[i] = min(UAA[i], 1)

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # Hypothetical "p"
    for j in range(s):
        for i in range(k):
            if LAA[i] < hp[j] < UAA[i]:
                cpAA[i, j] = stats.binom.pmf(i, n, hp[j])
        cppAA[j] = np.sum(cpAA[:, j])
        RMSE_N1[j] = (cppAA[j] - (1 - alp))**2
        if t1 < cppAA[j] < t2:
            ctr += 1

    mcpAA = np.mean(cppAA)  # Mean Cov Prob
    micpAA = np.min(cppAA)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean Square from min and mean CoPr
    for j in range(s):
        RMSE_M1[j] = (cppAA[j] - mcpAA)**2
        RMSE_Mi1[j] = (cppAA[j] - micpAA)**2

    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s

    return pd.DataFrame({
        "mcpAA": [mcpAA],
        "micpAA": [micpAA],
        "RMSE_N": [RMSE_N],
        "RMSE_M": [RMSE_M],
        "RMSE_MI": [RMSE_MI],
        "tol": [tol]
    })
#sjtp

def covpALT(n, alp, h, a, b, t1, t2):
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if not isinstance(h, (int, float)) or h < 0:
        raise ValueError("'h' has to be greater than or equal to 0")
    if not isinstance(a, (int, float)) or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' and 't2' have to be between 0 and 1")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than 't2'")

    # Input n
    x = np.arange(n + 1)
    k = n + 1
    y = x + h
    n1 = n + 2 * h

    # Initializations
    pALT = np.zeros(k)
    qALT = np.zeros(k)
    lgit = np.zeros(k)
    LALT = np.zeros(k)
    UALT = np.zeros(k)
    seALT = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpALT = np.zeros((k, s))
    cppALT = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    RMSE_M1 = np.zeros(s)
    RMSE_Mi1 = np.zeros(s)
    ctr = 0

    # Critical values
    cv = stats.norm.ppf(1 - alp / 2)
    
    # Adjusted Logit-Wald Method
    for i in range(k):
        pALT[i] = y[i] / n1
        qALT[i] = 1 - pALT[i]
        lgit[i] = np.log(pALT[i] / qALT[i])
        seALT[i] = np.sqrt(pALT[i] * qALT[i] * n1)
        LALT[i] = 1 / (1 + np.exp(-lgit[i] + (cv / seALT[i])))
        UALT[i] = 1 / (1 + np.exp(-lgit[i] - (cv / seALT[i])))
        LALT[i] = max(LALT[i], 0)
        UALT[i] = min(UALT[i], 1)

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # Hypothetical "p"
    for j in range(s):
        for i in range(k):
            if LALT[i] < hp[j] < UALT[i]:
                cpALT[i, j] = stats.binom.pmf(i, n, hp[j])
        cppALT[j] = np.sum(cpALT[:, j])
        RMSE_N1[j] = (cppALT[j] - (1 - alp))**2
        if t1 < cppALT[j] < t2:
            ctr += 1

    mcpALT = np.mean(cppALT)  # Mean Cov Prob
    micpALT = np.min(cppALT)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean Square from min and mean CoPr
    for j in range(s):
        RMSE_M1[j] = (cppALT[j] - mcpALT)**2
        RMSE_Mi1[j] = (cppALT[j] - micpALT)**2

    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s

    return pd.DataFrame({
        "mcpALT": [mcpALT],
        "micpALT": [micpALT],
        "RMSE_N": [RMSE_N],
        "RMSE_M": [RMSE_M],
        "RMSE_MI": [RMSE_MI],
        "tol": [tol]
    })



def covpATW(n, alpha, h, a, b, t1, t2, s=5000):
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alpha <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if h < 0:
        raise ValueError("'h' has to be greater than or equal to 0")
    if a < 0 or b < 0:
        raise ValueError("'a' and 'b' have to be greater than or equal to 0")
    if not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' and 't2' have to be between 0 and 1")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than 't2'")

    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    y = x + h
    n1 = n + 2 * h

    # Arrays for storing values
    pATW = np.zeros(k)
    qATW = np.zeros(k)
    seATW = np.zeros(k)
    LATW = np.zeros(k)
    UATW = np.zeros(k)
    DOF = np.zeros(k)
    cv = np.zeros(k)

    # Modified t-Wald method
    for i in range(k):
        pATW[i] = y[i] / n1
        qATW[i] = 1 - pATW[i]

        # Degrees of freedom
        f1 = lambda p, n: p * (1 - p) / n
        f2 = lambda p, n: (p * (1 - p) / n ** 3 + (p + (6 * n - 7) * p ** 2 +
                                                   4 * (n - 1) * (n - 3) * p ** 3 - 2 * (n - 1) * (
                                                               2 * n - 3) * p ** 4) / n ** 5 -
                           2 * (p + (2 * n - 3) * p ** 2 - 2 * (n - 1) * p ** 3) / n ** 4)

        DOF[i] = 2 * (f1(pATW[i], n1)) ** 2 / f2(pATW[i], n1)
        cv[i] = stats.t.ppf(1 - alpha / 2, df=DOF[i])
        seATW[i] = cv[i] * np.sqrt(f1(pATW[i], n1))
        LATW[i] = max(0, pATW[i] - seATW[i])
        UATW[i] = min(1, pATW[i] + seATW[i])

    # Generate hypothetical p values and calculate coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))
    cpATW = np.zeros((k, s))
    cppATW = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    for j in range(s):
        for i in range(k):
            if LATW[i] < hp[j] < UATW[i]:
                cpATW[i, j] = stats.binom.pmf(i, n, hp[j])
        cppATW[j] = np.sum(cpATW[:, j])
        RMSE_N1[j] = (cppATW[j] - (1 - alpha)) ** 2
        if t1 < cppATW[j] < t2:
            ctr += 1

    # Compute results
    mcpATW = np.mean(cppATW)
    micpATW = np.min(cppATW)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from min and mean coverage probability
    RMSE_M = np.sqrt(np.mean((cppATW - mcpATW) ** 2))
    RMSE_MI = np.sqrt(np.mean((cppATW - micpATW) ** 2))
    tol = 100 * ctr / s

    return pd.DataFrame({
        'mcpATW': [mcpATW],
        'micpATW': [micpATW],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })


def covpALR(n, alp, h, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if h < 0 or not isinstance(h, int):
        raise ValueError("'h' has to be an integer greater than or equal to 0")
    if a < 0 or b < 0:
        raise ValueError("'a' and 'b' have to be greater than or equal to 0")
    if t1 > t2 or not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' has to be lesser than 't2' and both must be between 0 and 1")

    # INPUT n
    y = np.arange(n+1)
    k = n + 1
    y1 = y + h
    n1 = n + (2 * h)

    # INITIALIZATIONS
    mle = np.zeros(k)
    cutoff = np.zeros(k)
    LAL = np.zeros(k)
    UAL = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpAL = np.zeros((k, s))
    cppAL = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # CRITICAL VALUES
    cv = stats.norm.ppf(1 - (alp / 2), loc=0, scale=1)

    # ADJUSTED LIKELIHOOD-RATIO METHOD
    for i in range(k):
        def likelhd(p):
            return stats.binom.pmf(y1[i], n1, p)

        def loglik(p):
            return stats.binom.logpmf(y1[i], n1, p)

        mle[i] = optimize.minimize_scalar(lambda p: -likelhd(p), bounds=(0, 1), method='bounded').x
        cutoff[i] = loglik(mle[i]) - (cv ** 2 / 2)

        def loglik_optim(p):
            return abs(cutoff[i] - loglik(p))

        LAL[i] = optimize.minimize_scalar(loglik_optim, bounds=(0, mle[i]), method='bounded').x
        UAL[i] = optimize.minimize_scalar(loglik_optim, bounds=(mle[i], 1), method='bounded').x

    # COVERAGE PROBABILITIES
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # HYPOTHETICAL "p"

    for j in range(s):
        for i in range(k):
            if LAL[i] < hp[j] < UAL[i]:
                cpAL[i, j] = stats.binom.pmf(i, n, hp[j])

        cppAL[j] = np.sum(cpAL[:, j])  # Coverage Probability
        RMSE_N1[j] = (cppAL[j] - (1 - alp)) ** 2
        if t1 < cppAL[j] < t2:
            ctr += 1

    mcpAL = np.mean(cppAL)
    micpAL = np.min(cppAL)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean Square from min and mean CoPr
    RMSE_M1 = (cppAL - mcpAL) ** 2
    RMSE_Mi1 = (cppAL - micpAL) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s


    return pd.DataFrame({
        "mcpAL":[mcpAL],
        "micpAL":[micpAL],
        "RMSE_N":[RMSE_N],
        "RMSE_M":[RMSE_M],
        "RMSE_MI":[RMSE_MI],
        "tol":[tol]
        })







def covpAAll(n, alp, h, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if h < 0 or not isinstance(h, int):
        raise ValueError("'h' has to be an integer greater than or equal to 0")
    if a < 0 or b < 0:
        raise ValueError("'a' and 'b' have to be greater than or equal to 0")
    if t1 > t2 or not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' has to be lesser than 't2' and both must be between 0 and 1")

    # Call the individual functions
    Waldcovp_df = covpAWD(n, alp, h, a, b, t1, t2)
    ArcSinecovp_df = covpAAS(n, alp, h, a, b, t1, t2)
    LRcovp_df = covpALR(n, alp, h, a, b, t1, t2)
    Scorecovp_df = covpASC(n, alp, h, a, b, t1, t2)
    WaldLcovp_df = covpALT(n, alp, h, a, b, t1, t2)
    AdWaldcovp_df = covpATW(n, alp, h, a, b, t1, t2)

    # Add method labels
    Waldcovp_df['method'] = 'Adj-Wald'
    ArcSinecovp_df['method'] = 'Adj-ArcSine'
    LRcovp_df['method'] = 'Adj-Likelihood'
    Scorecovp_df['method'] = 'Adj-Score'
    WaldLcovp_df['method'] = 'Adj-Logit-Wald'
    AdWaldcovp_df['method'] = 'Adj-Wald-T'

    # Create the Generic dataframes
    Generic_1 = pd.DataFrame({
        'method': Waldcovp_df['method'],
        'MeanCP': Waldcovp_df['mcpAW'],
        'MinCP': Waldcovp_df['micpAW'],
        'RMSE_N': Waldcovp_df['RMSE_N'],
        'RMSE_M': Waldcovp_df['RMSE_M'],
        'RMSE_MI': Waldcovp_df['RMSE_MI'],
        'tol': Waldcovp_df['tol']
    })

    Generic_2 = pd.DataFrame({
        'method': ArcSinecovp_df['method'],
        'MeanCP': ArcSinecovp_df['mcpAA'],
        'MinCP': ArcSinecovp_df['micpAA'],
        'RMSE_N': ArcSinecovp_df['RMSE_N'],
        'RMSE_M': ArcSinecovp_df['RMSE_M'],
        'RMSE_MI': ArcSinecovp_df['RMSE_MI'],
        'tol': ArcSinecovp_df['tol']
    })

    Generic_3 = pd.DataFrame({
        'method': LRcovp_df['method'],
        'MeanCP': LRcovp_df['mcpAL'],
        'MinCP': LRcovp_df['micpAL'],
        'RMSE_N': LRcovp_df['RMSE_N'],
        'RMSE_M': LRcovp_df['RMSE_M'],
        'RMSE_MI': LRcovp_df['RMSE_MI'],
        'tol': LRcovp_df['tol']
    })

    Generic_4 = pd.DataFrame({
        'method': Scorecovp_df['method'],
        'MeanCP': Scorecovp_df['mcpAS'],
        'MinCP': Scorecovp_df['micpAS'],
        'RMSE_N': Scorecovp_df['RMSE_N'],
        'RMSE_M': Scorecovp_df['RMSE_M'],
        'RMSE_MI': Scorecovp_df['RMSE_MI'],
        'tol': Scorecovp_df['tol']
    })

    Generic_5 = pd.DataFrame({
        'method': WaldLcovp_df['method'],
        'MeanCP': WaldLcovp_df['mcpALT'],
        'MinCP': WaldLcovp_df['micpALT'],
        'RMSE_N': WaldLcovp_df['RMSE_N'],
        'RMSE_M': WaldLcovp_df['RMSE_M'],
        'RMSE_MI': WaldLcovp_df['RMSE_MI'],
        'tol': WaldLcovp_df['tol']
    })

    Generic_6 = pd.DataFrame({
        'method': AdWaldcovp_df['method'],
        'MeanCP': AdWaldcovp_df['mcpATW'],
        'MinCP': AdWaldcovp_df['micpATW'],
        'RMSE_N': AdWaldcovp_df['RMSE_N'],
        'RMSE_M': AdWaldcovp_df['RMSE_M'],
        'RMSE_MI': AdWaldcovp_df['RMSE_MI'],
        'tol': AdWaldcovp_df['tol']
    })

    # Combine the dataframes
    Final_df = pd.concat([Generic_1, Generic_2, Generic_3, Generic_4, Generic_5, Generic_6],ignore_index=True)

    return Final_df
#SJTP