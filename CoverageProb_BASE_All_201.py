import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

def covpWD(n, alp, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("t1 has to be lesser than t2")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")

    x = np.arange(0, n + 1)
    k = n + 1
    s = 5000  # Simulation run to generate hypothetical p
    cpW = np.zeros((k, s))
    cppW = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # Critical value for normal distribution
    cv = stats.norm.ppf(1 - (alp / 2))

    # WALD method
    pW = x / n
    qW = 1 - pW
    seW = np.sqrt(pW * qW / n)
    LW = pW - (cv * seW)
    UW = pW + (cv * seW)

    LW[LW < 0] = 0
    UW[UW > 1] = 1

    # Coverage probabilities using hypothetical "p"
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    for j in range(s):
        for i in range(k):
            if LW[i] < hp[j] < UW[i]:
                cpW[i, j] = stats.binom.pmf(i, n, hp[j])
        cppW[j] = np.sum(cpW[:, j])
        RMSE_N1[j] = (cppW[j] - (1 - alp)) ** 2  # Root Mean Square Error from nominal size
        if t1 < cppW[j] < t2:
            ctr += 1

    mcpW = np.mean(cppW)
    micpW = np.min(cppW)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from min and mean CoPr
    RMSE_M1 = (cppW - mcpW) ** 2
    RMSE_Mi1 = (cppW - micpW) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))

    tol = 100 * ctr / s

    # Create a DataFrame for the results
    return pd.DataFrame({
        'mcpW': [mcpW],
        'micpW': [micpW],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })



def covpSC(n, alp, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than t2")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")

    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    s = 5000  # Simulation run to generate hypothetical p

    # Arrays for storing values
    pS = np.zeros(k)
    qS = np.zeros(k)
    seS = np.zeros(k)
    LS = np.zeros(k)
    US = np.zeros(k)
    cpS = np.zeros((k, s))
    cppS = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # Critical value for normal distribution
    cv = stats.norm.ppf(1 - (alp / 2))
    cv1 = (cv ** 2) / (2 * n)
    cv2 = (cv / (2 * n)) ** 2

    # SCORE (WILSON) METHOD
    for i in range(k):
        pS[i] = x[i] / n
        qS[i] = 1 - pS[i]
        seS[i] = np.sqrt((pS[i] * qS[i] / n) + cv2)
        LS[i] = (n / (n + cv ** 2)) * ((pS[i] + cv1) - (cv * seS[i]))
        US[i] = (n / (n + cv ** 2)) * ((pS[i] + cv1) + (cv * seS[i]))
        LS[i] = max(0, LS[i])
        US[i] = min(1, US[i])

    # Hypothetical "p" values
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    # Coverage probabilities
    for j in range(s):
        for i in range(k):
            if LS[i] < hp[j] < US[i]:
                cpS[i, j] = stats.binom.pmf(i, n, hp[j])
        cppS[j] = np.sum(cpS[:, j])  # Coverage Probability
        RMSE_N1[j] = (cppS[j] - (1 - alp)) ** 2  # Root Mean Square from nominal size
        if t1 < cppS[j] < t2:
            ctr += 1

    # Calculate metrics
    mcpS = np.mean(cppS)
    micpS = np.min(cppS)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from min and mean CoPr
    RMSE_M1 = (cppS - mcpS) ** 2
    RMSE_Mi1 = (cppS - micpS) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s

    # Create a DataFrame for the results
    return pd.DataFrame({
        'mcpS': [mcpS],
        'micpS': [micpS],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })








def covpAS(n, alp, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than t2")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")

    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    s = 5000  # Simulation run to generate hypothetical p

    # Arrays for storing values
    pA = np.zeros(k)
    qA = np.zeros(k)
    seA = np.zeros(k)
    LA = np.zeros(k)
    UA = np.zeros(k)
    cpA = np.zeros((k, s))
    cppA = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # Critical value for normal distribution
    cv = stats.norm.ppf(1 - (alp / 2))

    # ARC-SINE METHOD
    for i in range(k):
        pA[i] = x[i] / n
        qA[i] = 1 - pA[i]
        seA[i] = cv / np.sqrt(4 * n)
        LA[i] = (np.sin(np.arcsin(np.sqrt(pA[i])) - seA[i])) ** 2
        UA[i] = (np.sin(np.arcsin(np.sqrt(pA[i])) + seA[i])) ** 2
        LA[i] = max(0, LA[i])
        UA[i] = min(1, UA[i])

    # Hypothetical "p" values
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    # Coverage probabilities
    for j in range(s):
        for i in range(k):
            if LA[i] < hp[j] < UA[i]:
                cpA[i, j] = stats.binom.pmf(i, n, hp[j])
        cppA[j] = np.sum(cpA[:, j])  # Coverage Probability
        RMSE_N1[j] = (cppA[j] - (1 - alp)) ** 2  # Root Mean Square from nominal size
        if t1 < cppA[j] < t2:
            ctr += 1

    # Calculate metrics
    mcpA = np.mean(cppA)
    micpA = np.min(cppA)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from min and mean CoPr
    RMSE_M1 = (cppA - mcpA) ** 2
    RMSE_Mi1 = (cppA - micpA) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s

    # Create a DataFrame for the results
    return pd.DataFrame({
        'mcpA': [mcpA],
        'micpA': [micpA],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })



def covpLT(n, alp, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than t2")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")
    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    s = 5000  # Simulation run to generate hypothetical p

    # Arrays for storing values
    pLT = np.zeros(k)
    qLT = np.zeros(k)
    seLT = np.zeros(k)
    lgit = np.zeros(k)
    LLT = np.zeros(k)
    ULT = np.zeros(k)
    cpLT = np.zeros((k, s))
    cppLT = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # Critical value for normal distribution
    cv = stats.norm.ppf(1 - (alp / 2))

    # LOGIT-WALD METHOD
    pLT[0], qLT[0] = 0, 1
    LLT[0], ULT[0] = 0, 1 - (alp / 2) ** (1 / n)

    pLT[-1], qLT[-1] = 1, 0
    LLT[-1], ULT[-1] = (alp / 2) ** (1 / n), 1

    for j in range(1, k - 1):
        pLT[j] = x[j] / n
        qLT[j] = 1 - pLT[j]
        lgit[j] = np.log(pLT[j] / qLT[j])
        seLT[j] = np.sqrt(pLT[j] * qLT[j] * n)
        LLT[j] = 1 / (1 + np.exp(-lgit[j] + (cv / seLT[j])))
        ULT[j] = 1 / (1 + np.exp(-lgit[j] - (cv / seLT[j])))
        LLT[j] = max(0, LLT[j])
        ULT[j] = min(1, ULT[j])

    # Hypothetical "p" values
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    # Coverage probabilities
    for j in range(s):
        for i in range(k):
            if LLT[i] < hp[j] < ULT[i]:
                cpLT[i, j] = stats.binom.pmf(i, n, hp[j])
        cppLT[j] = np.sum(cpLT[:, j])  # Coverage Probability
        RMSE_N1[j] = (cppLT[j] - (1 - alp)) ** 2  # Root mean square from nominal size
        if t1 < cppLT[j] < t2:
            ctr += 1

    # Calculate metrics
    mcpLT = np.mean(cppLT)
    micpLT = np.min(cppLT)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from min and mean CoPr
    RMSE_M1 = (cppLT - mcpLT) ** 2
    RMSE_Mi1 = (cppLT - micpLT) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s

    # Create a DataFrame for the results
    return pd.DataFrame({
        'mcpLT': [mcpLT],
        'micpLT': [micpLT],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })




def covpTW(n, alp, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than t2")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")

    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    s = 5000  # Simulation run to generate hypothetical p

    # Arrays for storing values
    pTW = np.zeros(k)
    qTW = np.zeros(k)
    seTW = np.zeros(k)
    LTW = np.zeros(k)
    UTW = np.zeros(k)
    DOF = np.zeros(k)
    cv = np.zeros(k)
    cpTW = np.zeros((k, s))
    cppTW = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # MODIFIED t-WALD METHOD
    for i in range(k):
        if x[i] == 0 or x[i] == n:
            pTW[i] = (x[i] + 2) / (n + 4)
        else:
            pTW[i] = x[i] / n
        
        qTW[i] = 1 - pTW[i]
        
        # Define functions for DOF calculation
        f1 = lambda p: p * (1 - p) / n
        f2 = lambda p: (p * (1 - p) / (n ** 3) +
                        (p + (6 * n - 7) * p ** 2 + 4 * (n - 1) * (n - 3) * p ** 3 -
                         2 * (n - 1) * (2 * n - 3) * p ** 4) / (n ** 5) -
                        2 * (p + (2 * n - 3) * p ** 2 - 2 * (n - 1) * p ** 3) / (n ** 4))

        DOF[i] = 2 * (f1(pTW[i])) ** 2 / f2(pTW[i])
        cv[i] = stats.t.ppf(1 - (alp / 2), df=DOF[i])
        seTW[i] = cv[i] * np.sqrt(f1(pTW[i]))
        LTW[i] = pTW[i] - seTW[i]
        UTW[i] = pTW[i] + seTW[i]
        
        # Ensure bounds are within [0, 1]
        LTW[i] = max(0, LTW[i])
        UTW[i] = min(1, UTW[i])

    # Hypothetical "p" values
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    # Coverage probabilities
    for j in range(s):
        for i in range(k):
            if LTW[i] < hp[j] < UTW[i]:
                cpTW[i, j] = stats.binom.pmf(i, n, hp[j])
        cppTW[j] = np.sum(cpTW[:, j])  # Coverage Probability
        RMSE_N1[j] = (cppTW[j] - (1 - alp)) ** 2  # Root mean square from nominal size
        if t1 < cppTW[j] < t2:
            ctr += 1

    # Calculate metrics
    mcpTW = np.mean(cppTW)
    micpTW = np.min(cppTW)  # Mean Coverage Probability
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from min and mean CoPr
    RMSE_M1 = (cppTW - mcpTW) ** 2
    RMSE_Mi1 = (cppTW - micpTW) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s

    # Create a DataFrame for the results
    return pd.DataFrame({
        'mcpTW': [mcpTW],
        'micpTW': [micpTW],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })




def covpLR(n, alp, a, b, t1, t2):
    # Input validation
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be lesser than t2")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")

    # Initialize variables
    y = np.arange(0, n + 1)
    k = n + 1
    s = 5000  # Simulation run to generate hypothetical p

    # Arrays for storing values
    mle = np.zeros(k)
    cutoff = np.zeros(k)
    LL = np.zeros(k)
    UL = np.zeros(k)
    cpL = np.zeros((k, s))
    cppL = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # CRITICAL VALUES
    cv = stats.norm.ppf(1 - (alp / 2))

    # LIKELIHOOD-RATIO METHOD
    for i in range(k):
        likelhd = lambda p: stats.binom.pmf(y[i], n, p)
        loglik = lambda p: stats.binom.logpmf(y[i], n, p)
        
        # Maximum likelihood estimate
        mle[i] = optimize.minimize_scalar(lambda p: -likelhd(p), bounds=(0, 1), method='bounded').x

        # Cutoff value for log-likelihood
        cutoff[i] = loglik(mle[i]) - (cv ** 2 / 2)

        # Finding bounds for the log-likelihood ratio
        loglik_opt = lambda p: abs(cutoff[i] - loglik(p))
        LL[i] = optimize.minimize_scalar(loglik_opt, bounds=(0, mle[i]), method='bounded').x
        UL[i] = optimize.minimize_scalar(loglik_opt, bounds=(mle[i], 1), method='bounded').x

    # COVERAGE PROBABILITIES
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # HYPOTHETICAL "p"
    for j in range(s):
        for i in range(k):
            if LL[i] < hp[j] < UL[i]:
                cpL[i, j] = stats.binom.pmf(i - 1, n, hp[j])
        
        cppL[j] = np.sum(cpL[:, j])  # Coverage Probability
        RMSE_N1[j] = (cppL[j] - (1 - alp)) ** 2  # Root mean square from nominal size
        if t1 < cppL[j] < t2:
            ctr += 1  # Tolerance for coverage probability - user defined

    # Calculate metrics
    mcpL = np.mean(cppL)
    micpL = np.min(cppL)  # Mean Coverage Probability
    RMSE_N = np.sqrt(np.mean(RMSE_N1))

    # Root mean square from min and mean CoPr
    RMSE_M1 = (cppL - mcpL) ** 2
    RMSE_Mi1 = (cppL - micpL) ** 2
    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s

    # Create a DataFrame for the results
    return pd.DataFrame({
        'mcpL': [mcpL],
        'micpL': [micpL],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol]
    })







def covpEX(n, alp, e, a, b, t1, t2):
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if isinstance(e, (float, int)):
        e = [e]
    elif not isinstance(e, (list, np.ndarray)):
        raise ValueError("'e' must be a list, array, or float with values between 0 and 1")
    if any(val < 0 or val > 1 for val in e) or len(e) > 10:
        raise ValueError("'e' has to be between 0 and 1 and can have only 10 intervals")
    if a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("t1 has to be lesser than t2")
    if not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")

    res = pd.DataFrame()

    for e_val in e:
        lu = oldEX201(n, alp, e_val, a, b, t1, t2)
        res = pd.concat([res, lu], ignore_index=True)

    return res
def oldEX201(n, alp, e, a, b, t1, t2):
    x = np.arange(n + 1)
    k = n + 1
    LEX = np.zeros(k)
    UEX = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpEX = np.zeros((k, s))
    cppEX = np.zeros(s)
    RMSE_N1 = np.zeros(s)
    ctr = 0

    # EXACT METHOD
    LEX[0] = 0
    UEX[0] = 1 - ((alp / (2 * e)) ** (1/n))
    LEX[-1] = (alp / (2 * e)) ** (1/n)
    UEX[-1] = 1

    for i in range(1, k - 1):
        LEX[i] = exlim201l(x[i], n, alp, e)
        UEX[i] = exlim201u(x[i], n, alp, e)

    # COVERAGE PROBABILITIES
    hp = np.sort(stats.beta.rvs(a, b, size=s))
    for j in range(s):
        for i in range(k):
            if LEX[i] < hp[j] < UEX[i]:
                cpEX[i, j] = stats.binom.pmf(i - 1, n, hp[j])
                ctr += 1

        cppEX[j] = np.sum(cpEX[:, j])
        RMSE_N1[j] = (cppEX[j] - (1 - alp)) ** 2

    mcpEX = np.mean(cppEX)
    micpEX = np.min(cppEX)
    RMSE_N = np.sqrt(np.mean(RMSE_N1))
    RMSE_M1 = (cppEX - mcpEX) ** 2
    RMSE_Mi1 = (cppEX - micpEX) ** 2

    RMSE_M = np.sqrt(np.mean(RMSE_M1))
    RMSE_MI = np.sqrt(np.mean(RMSE_Mi1))
    tol = 100 * ctr / s

    return pd.DataFrame({
        'mcpEX': [mcpEX],
        'micpEX': [micpEX],
        'RMSE_N': [RMSE_N],
        'RMSE_M': [RMSE_M],
        'RMSE_MI': [RMSE_MI],
        'tol': [tol],
        'e': [e]
    })
def exlim201l(x, n, alp, e):
    z = int(x) - 1
    y = np.arange(z + 1)
    f1 = lambda p: (1 - e) * stats.binom.pmf(x, n, p) + np.sum(stats.binom.pmf(y, n, p)) - (1 - (alp / 2))
    LEX = optimize.root_scalar(f1, bracket=[0, 1]).root
    return LEX

# Function to find upper limits
def exlim201u(x, n, alp, e):
    z = int(x) - 1
    y = np.arange(z + 1)
    f2 = lambda p: e * stats.binom.pmf(x, n, p) + np.sum(stats.binom.pmf(y, n, p)) - (alp / 2)
    UEX = optimize.root_scalar(f2, bracket=[0, 1]).root
    return UEX

def covpAll(n, alp, a, b, t1, t2):
    """
    Calculate coverage probabilities using different methods.

    Parameters:
    n (int): Sample size
    alp (float): Alpha value (between 0 and 1)
    a (float): Lower bound parameter
    b (float): Upper bound parameter
    t1 (float): Lower threshold (between 0 and 1)
    t2 (float): Upper threshold (between 0 and 1)

    Returns:
    pandas.DataFrame: Combined results from all methods
    """
    # Input validation
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if a is None:
        raise ValueError("'a' is missing")
    if b is None:
        raise ValueError("'b' is missing")
    if t1 is None:
        raise ValueError("'t1' is missing")
    if t2 is None:
        raise ValueError("'t2' is missing")

    # Type and value checking
    if not isinstance(n, (int, float)) or isinstance(n, bool) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not isinstance(alp, (int, float)) or alp > 1 or alp < 0:
        raise ValueError("'alpha' has to be between 0 and 1")
    if not isinstance(a, (int, float)) or isinstance(a, bool) or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or isinstance(b, bool) or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 >= t2:
        raise ValueError("t1 has to be lesser than t2")
    if not isinstance(t1, (int, float)) or t1 < 0 or t1 > 1:
        raise ValueError("'t1' has to be between 0 and 1")
    if not isinstance(t2, (int, float)) or t2 < 0 or t2 > 1:
        raise ValueError("'t2' has to be between 0 and 1")

    # Assuming these functions exist and return pandas DataFrames
    Waldcovp_df = covpWD(n, alp, a, b, t1, t2)
    ArcSinecovp_df = covpAS(n, alp, a, b, t1, t2)
    LRcovp_df = covpLR(n, alp, a, b, t1, t2)
    Scorecovp_df = covpSC(n, alp, a, b, t1, t2)
    WaldLcovp_df = covpLT(n, alp, a, b, t1, t2)
    AdWaldcovp_df = covpTW(n, alp, a, b, t1, t2)

    # Create individual method dataframes
    generic_dfs = [
        pd.DataFrame({
            'method': 'Wald',
            'MeanCP': Waldcovp_df['mcpW'],
            'MinCP': Waldcovp_df['micpW'],
            'RMSE_N': Waldcovp_df['RMSE_N'],
            'RMSE_M': Waldcovp_df['RMSE_M'],
            'RMSE_MI': Waldcovp_df['RMSE_MI'],
            'tol': Waldcovp_df['tol']
        }),
        pd.DataFrame({
            'method': 'ArcSine',
            'MeanCP': ArcSinecovp_df['mcpA'],
            'MinCP': ArcSinecovp_df['micpA'],
            'RMSE_N': ArcSinecovp_df['RMSE_N'],
            'RMSE_M': ArcSinecovp_df['RMSE_M'],
            'RMSE_MI': ArcSinecovp_df['RMSE_MI'],
            'tol': ArcSinecovp_df['tol']
        }),
        pd.DataFrame({
            'method': 'Likelihood',
            'MeanCP': LRcovp_df['mcpL'],
            'MinCP': LRcovp_df['micpL'],
            'RMSE_N': LRcovp_df['RMSE_N'],
            'RMSE_M': LRcovp_df['RMSE_M'],
            'RMSE_MI': LRcovp_df['RMSE_MI'],
            'tol': LRcovp_df['tol']
        }),
        pd.DataFrame({
            'method': 'Score',
            'MeanCP': Scorecovp_df['mcpS'],
            'MinCP': Scorecovp_df['micpS'],
            'RMSE_N': Scorecovp_df['RMSE_N'],
            'RMSE_M': Scorecovp_df['RMSE_M'],
            'RMSE_MI': Scorecovp_df['RMSE_MI'],
            'tol': Scorecovp_df['tol']
        }),
        pd.DataFrame({
            'method': 'WaldLogit',
            'MeanCP': WaldLcovp_df['mcpLT'],
            'MinCP': WaldLcovp_df['micpLT'],
            'RMSE_N': WaldLcovp_df['RMSE_N'],
            'RMSE_M': WaldLcovp_df['RMSE_M'],
            'RMSE_MI': WaldLcovp_df['RMSE_MI'],
            'tol': WaldLcovp_df['tol']
        }),
        pd.DataFrame({
            'method': 'Wald-T',
            'MeanCP': AdWaldcovp_df['mcpTW'],
            'MinCP': AdWaldcovp_df['micpTW'],
            'RMSE_N': AdWaldcovp_df['RMSE_N'],
            'RMSE_M': AdWaldcovp_df['RMSE_M'],
            'RMSE_MI': AdWaldcovp_df['RMSE_MI'],
            'tol': AdWaldcovp_df['tol']
        })
    ]

    # Combine all dataframes
    final_df = pd.concat(generic_dfs, ignore_index=True)

    # Convert method to categorical type (equivalent to as.factor in R)
    final_df['method'] = pd.Categorical(final_df['method'])

    return final_df
#SJTP