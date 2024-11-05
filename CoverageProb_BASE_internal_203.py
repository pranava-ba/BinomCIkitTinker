import numpy as np
import scipy.stats as stats
import pandas as pd
import scipy.optimize as optimize

def gcovpW(n, alpha, a, b, t1, t2, s=5000):
    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    
    # Initialize lists and arrays
    pW = np.zeros(k)
    qW = np.zeros(k)
    seW = np.zeros(k)
    LW = np.zeros(k)
    UW = np.zeros(k)
    cpW = np.zeros((k, s))
    cppW = np.zeros(s)
    ctr = 0
    
    # Critical values
    cv = stats.norm.ppf(1 - (alpha / 2), loc=0, scale=1)
    
    # Wald Method
    for i in range(k):
        pW[i] = x[i] / n
        qW[i] = 1 - (x[i] / n)
        seW[i] = np.sqrt(pW[i] * qW[i] / n)
        LW[i] = pW[i] - (cv * seW[i])
        UW[i] = pW[i] + (cv * seW[i])
        LW[i] = max(LW[i], 0)  # Ensure lower bound is not less than 0
        UW[i] = min(UW[i], 1)  # Ensure upper bound is not greater than 1
    
    # Generate hypothetical "p" values from a beta distribution
    hp = np.sort(np.random.beta(a, b, s))
    
    # Calculate coverage probabilities
    for j in range(s):
        for i in range(k):
            if LW[i] < hp[j] < UW[i]:
                cpW[i, j] = stats.binom.pmf(i, n, hp[j])
        cppW[j] = np.sum(cpW[:, j])
        if t1 < cppW[j] < t2:
            ctr += 1
    return pd.DataFrame({'hp': hp, 'cp': cppW, 'method': 'Wald'})

def gcovpS(n, alpha, a, b, t1, t2, s=5000):
    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    
    # Initialize lists and arrays
    pS = np.zeros(k)
    qS = np.zeros(k)
    seS = np.zeros(k)
    LS = np.zeros(k)
    US = np.zeros(k)
    cpS = np.zeros((k, s))
    cppS = np.zeros(s)
    ctr = 0
    
    # Critical values
    cv = stats.norm.ppf(1 - (alpha / 2), loc=0, scale=1)
    cv1 = (cv**2) / (2 * n)
    cv2 = (cv / (2 * n))**2
    
    # Score (Wilson) Method
    for i in range(k):
        pS[i] = x[i] / n
        qS[i] = 1 - (x[i] / n)
        seS[i] = np.sqrt((pS[i] * qS[i] / n) + cv2)
        LS[i] = (n / (n + cv**2)) * ((pS[i] + cv1) - (cv * seS[i]))
        US[i] = (n / (n + cv**2)) * ((pS[i] + cv1) + (cv * seS[i]))
        LS[i] = max(LS[i], 0)  # Ensure lower bound is not less than 0
        US[i] = min(US[i], 1)  # Ensure upper bound is not greater than 1
    
    # Generate hypothetical "p" values from a beta distribution
    hp = np.sort(np.random.beta(a, b, s))
    
    # Calculate coverage probabilities
    for j in range(s):
        for i in range(k):
            if LS[i] < hp[j] < US[i]:
                cpS[i, j] = stats.binom.pmf(i, n, hp[j])
        cppS[j] = np.sum(cpS[:, j])
        if t1 < cppS[j] < t2:
            ctr += 1
    
    # Compile results into a dictionary or dataframe
    return  pd.DataFrame({'hp': hp, 'cp': cppS, 'method': 'Score'})

def gcovpA(n, alpha, a, b, t1, t2, s=5000):
    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    
    # Initialize lists and arrays
    pA = np.zeros(k)
    qA = np.zeros(k)
    seA = np.zeros(k)
    LA = np.zeros(k)
    UA = np.zeros(k)
    cpA = np.zeros((k, s))
    cppA = np.zeros(s)
    ctr = 0
    
    # Critical value
    cv = stats.norm.ppf(1 - (alpha / 2), loc=0, scale=1)
    
    # Arc-Sine Method
    for i in range(k):
        pA[i] = x[i] / n
        qA[i] = 1 - pA[i]
        seA[i] = cv / np.sqrt(4 * n)
        LA[i] = np.sin(np.arcsin(np.sqrt(pA[i])) - seA[i])**2
        UA[i] = np.sin(np.arcsin(np.sqrt(pA[i])) + seA[i])**2
        LA[i] = max(LA[i], 0)  # Ensure lower bound is not less than 0
        UA[i] = min(UA[i], 1)  # Ensure upper bound is not greater than 1
    
    # Generate hypothetical "p" values from a beta distribution
    hp = np.sort(np.random.beta(a, b, s))
    
    # Calculate coverage probabilities
    for j in range(s):
        for i in range(k):
            if LA[i] < hp[j] < UA[i]:
                cpA[i, j] = stats.binom.pmf(i, n, hp[j])
        cppA[j] = np.sum(cpA[:, j])
        if t1 < cppA[j] < t2:
            ctr += 1
    
    # Compile results into a dictionary or dataframe
    return pd.DataFrame({'hp': hp, 'cp': cppA, 'method': 'ArcSine'})



def gcovpLT(n, alpha, a, b, t1, t2, s=5000):
    # Initialize variables
    x = np.arange(0, n + 1)
    k = n + 1
    
    # Initialize arrays
    pLT = np.zeros(k)
    qLT = np.zeros(k)
    seLT = np.zeros(k)
    lgit = np.zeros(k)
    LLT = np.zeros(k)
    ULT = np.zeros(k)
    cpLT = np.zeros((k, s))
    cppLT = np.zeros(s)
    ctr = 0
    
    # Critical value
    cv = stats.norm.ppf(1 - (alpha / 2), loc=0, scale=1)
    
    # Logit-Wald Method: Boundary cases
    pLT[0] = 0
    qLT[0] = 1
    LLT[0] = 0
    ULT[0] = 1 - (alpha / 2)**(1 / n)

    pLT[k - 1] = 1
    qLT[k - 1] = 0
    LLT[k - 1] = (alpha / 2)**(1 / n)
    ULT[k - 1] = 1

    # Logit-Wald for other cases
    for j in range(1, k - 1):
        pLT[j] = x[j] / n
        qLT[j] = 1 - pLT[j]
        lgit[j] = np.log(pLT[j] / qLT[j])
        seLT[j] = np.sqrt(pLT[j] * qLT[j] * n)
        LLT[j] = 1 / (1 + np.exp(-lgit[j] + (cv / seLT[j])))
        ULT[j] = 1 / (1 + np.exp(-lgit[j] - (cv / seLT[j])))
        LLT[j] = max(LLT[j], 0)  # Ensure lower bound is not less than 0
        ULT[j] = min(ULT[j], 1)  # Ensure upper bound is not greater than 1
    hp = np.sort(np.random.beta(a, b, s))
    for j in range(s):
        for i in range(k):
            if LLT[i] < hp[j] < ULT[i]:
                cpLT[i, j] = stats.binom.pmf(i, n, hp[j])
        cppLT[j] = np.sum(cpLT[:, j])
        if t1 < cppLT[j] < t2:
            ctr += 1
    return pd.DataFrame({'hp': hp, 'cp': cppLT, 'method': 'Logit-Wald'})


def gcovpTW(n, alp, a, b, t1, t2):
    # Initialization
    x = np.arange(0, n + 1)
    k = n + 1

    pTW = np.zeros(k)
    qTW = np.zeros(k)
    seTW = np.zeros(k)
    LTW = np.zeros(k)
    UTW = np.zeros(k)
    DOF = np.zeros(k)
    cv = np.zeros(k)
    s = 5000  # Simulation runs for hypothetical p
    cpTW = np.zeros((k, s))
    cppTW = np.zeros(s)
    ctr = 0

    # Modified t-Wald Method
    for i in range(k):
        if x[i] == 0 or x[i] == n:
            pTW[i] = (x[i] + 2) / (n + 4)
            qTW[i] = 1 - pTW[i]
        else:
            pTW[i] = x[i] / n
            qTW[i] = 1 - pTW[i]

        # Defining helper functions
        f1 = lambda p, n: p * (1 - p) / n
        f2 = lambda p, n: ((p * (1 - p) / (n ** 3)) +
                           (p + ((6 * n) - 7) * (p ** 2) + (4 * (n - 1) * (n - 3) * (p ** 3)) -
                            (2 * (n - 1) * ((2 * n) - 3) * (p ** 4))) / (n ** 5) -
                           (2 * (p + ((2 * n) - 3) * (p ** 2) - 2 * (n - 1) * (p ** 3))) / (n ** 4))

        DOF[i] = 2 * (f1(pTW[i], n) ** 2) / f2(pTW[i], n)
        cv[i] = stats.t.ppf(1 - (alp / 2), df=DOF[i])
        seTW[i] = cv[i] * np.sqrt(f1(pTW[i], n))
        LTW[i] = max(pTW[i] - seTW[i], 0)
        UTW[i] = min(pTW[i] + seTW[i], 1)

    # Coverage Probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # Hypothetical "p" values
    for j in range(s):
        for i in range(k):
            if LTW[i] < hp[j] < UTW[i]:
                cpTW[i, j] = stats.binom.pmf(i, n, hp[j])
        cppTW[j] = np.sum(cpTW[:, j])  # Coverage Probability
        if t1 < cppTW[j] < t2:
            ctr += 1

    # Create DataFrame
    CPTW = pd.DataFrame({
        'hp': hp,
        'cp': cppTW,
        'method': ['Wald-T'] * s
    })

    return CPTW


def gcovpL(n, alp, a, b, t1, t2):
    # Initialization
    y = np.arange(0, n + 1)
    k = n + 1

    mle = np.zeros(k)
    cutoff = np.zeros(k)
    LL = np.zeros(k)
    UL = np.zeros(k)
    s = 5000  # Simulation runs for hypothetical p
    cpL = np.zeros((k, s))
    cppL = np.zeros(s)
    ctr = 0

    # Critical value
    cv = stats.norm.ppf(1 - (alp / 2), loc=0, scale=1)

    # Likelihood-Ratio Method
    for i in range(k):
        # Define the likelihood and log-likelihood functions
        likelhd = lambda p: stats.binom.pmf(y[i], n, p)
        loglik = lambda p: stats.binom.logpmf(y[i], n, p)

        # Calculate MLE
        result = optimize.minimize_scalar(lambda p: -likelhd(p), bounds=(0, 1), method='bounded')
        mle[i] = result.x

        # Calculate cutoff
        cutoff[i] = loglik(mle[i]) - (cv ** 2 / 2)

        # Define the optimization for lower and upper limits
        loglik_optim = lambda p: abs(cutoff[i] - loglik(p))

        # Calculate LL (Lower Limit)
        LL_result = optimize.minimize_scalar(loglik_optim, bounds=(0, mle[i]), method='bounded')
        LL[i] = LL_result.x

        # Calculate UL (Upper Limit)
        UL_result = optimize.minimize_scalar(loglik_optim, bounds=(mle[i], 1), method='bounded')
        UL[i] = UL_result.x

    # Coverage Probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))  # Hypothetical "p" values
    for j in range(s):
        for i in range(k):
            if LL[i] < hp[j] < UL[i]:
                cpL[i, j] = stats.binom.pmf(i, n, hp[j])
        cppL[j] = np.sum(cpL[:, j])  # Coverage Probability
        if t1 < cppL[j] < t2:
            ctr += 1

    # Create DataFrame
    CPL = pd.DataFrame({
        'hp': hp,
        'cp': cppL,
        'method': ['Likelihood'] * s
    })

    return CPL


def gcovpEX(n, alpha, e_vals, a, b, t1, t2, s=5000):
    if isinstance(e_vals, (int, float)):
        e_vals = np.array([e_vals])
    elif isinstance(e_vals, list):
        e_vals = np.array(e_vals)
    elif not isinstance(e_vals, np.ndarray):
        raise ValueError(f"Invalid type for 'e_vals': {type(e_vals)}")

    nvar = len(e_vals)
    result = pd.DataFrame()

    for e in e_vals:
        lu = gintcovpEX202(n, alpha, e, a, b, t1, t2, s)
        result = pd.concat([result, lu], ignore_index=True)

    return result


def gintcovpEX202(n, alpha, e, a, b, t1, t2, s=5000):
    x = np.arange(0, n + 1)
    k = n + 1
    LEX = np.zeros(k)
    UEX = np.zeros(k)
    cpEX = np.zeros((k, s))
    cppEX = np.zeros(s)

    # Exact method boundaries
    LEX[0] = 0
    UEX[0] = 1 - (alpha / (2 * e)) ** (1 / n)
    LEX[-1] = (alpha / (2 * e)) ** (1 / n)
    UEX[-1] = 1

    for i in range(1, k - 1):
        LEX[i] = exlim202l(x[i], n, alpha, e)
        UEX[i] = exlim202u(x[i], n, alpha, e)

    # Generate hypothetical "p" values from beta distribution
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    # Calculate coverage probabilities
    for j in range(s):
        for i in range(k):
            if LEX[i] < hp[j] < UEX[i]:
                cpEX[i, j] = stats.binom.pmf(i, n, hp[j])
        cppEX[j] = np.sum(cpEX[:, j])

    # Mean and minimum coverage probabilities
    mcpEX = np.mean(cppEX)
    micpEX = np.min(cppEX)

    # Compile results
    CPEX = pd.DataFrame({'hp': hp, 'cpp': cppEX})

    # Create a DataFrame with repeated values for micpEX and e
    micp_values = pd.Series([micpEX] * s)
    e_values = pd.Series([e] * s)

    result_df = pd.concat([CPEX,micp_values.rename('mcpEX'), micp_values.rename('micpEX'), e_values.rename('e')], axis=1)
    return result_df


def exlim202l(x, n, alpha, e):
    z = x - 1
    y = np.arange(0, z + 1)

    # Function to find the lower limit
    def f1(p):
        return (1 - e) * stats.binom.pmf(x, n, p) + np.sum(stats.binom.pmf(y, n, p)) - (1 - (alpha / 2))

    result = optimize.root_scalar(f1, bracket=[0, 1], method='brentq')
    return result.root


def exlim202u(x, n, alpha, e):
    z = x - 1
    y = np.arange(0, z + 1)

    # Function to find the upper limit
    def f2(p):
        return e * stats.binom.pmf(x, n, p) + np.sum(stats.binom.pmf(y, n, p)) - (alpha / 2)

    result = optimize.root_scalar(f2, bracket=[0, 1], method='brentq')
    return result.root


#SJTP