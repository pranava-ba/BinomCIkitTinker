import scipy.stats as stats
import numpy as np
import pandas as pd
import scipy.optimize as optimize

def gcovpAWD(n, alp, h, a, b, t1, t2):
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

    # Input setup
    x = np.arange(0, n + 1)
    k = n + 1
    y = x + h
    n1 = n + (2 * h)

    # Initializations
    pAW = np.zeros(k)
    qAW = np.zeros(k)
    seAW = np.zeros(k)
    LAW = np.zeros(k)
    UAW = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpAW = np.zeros((k, s))
    cppAW = np.zeros(s)
    ctr = 0

    # Critical value
    cv = stats.norm.ppf(1 - (alp / 2))

    # WALD METHOD
    for i in range(k):
        pAW[i] = y[i] / n1
        qAW[i] = 1 - pAW[i]
        seAW[i] = np.sqrt(pAW[i] * qAW[i] / n1)
        LAW[i] = pAW[i] - (cv * seAW[i])
        UAW[i] = pAW[i] + (cv * seAW[i])
        LAW[i] = max(LAW[i], 0)
        UAW[i] = min(UAW[i], 1)

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    for j in range(s):
        for i in range(k):
            if LAW[i] < hp[j] < UAW[i]:
                cpAW[i, j] = stats.binom.pmf(i, n, hp[j])
        cppAW[j] = np.sum(cpAW[:, j])
        if t1 < cppAW[j] < t2:
            ctr += 1

    # Create dataframe
    CPAW = pd.DataFrame({
        'hp': hp,
        'cp': cppAW,
        'method': ['Adj-Wald'] * s
    })

    return CPAW






def gcovpASC(n, alp, h, a, b, t1, t2):
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

    # Input setup
    x = np.arange(0, n + 1)
    k = n + 1
    y = x + h
    n1 = n + (2 * h)

    # Initializations
    pAS = np.zeros(k)
    qAS = np.zeros(k)
    seAS = np.zeros(k)
    LAS = np.zeros(k)
    UAS = np.zeros(k)
    s = 1000  # Simulation run to generate hypothetical p
    cpAS = np.zeros((k, s))
    cppAS = np.zeros(s)
    ctr = 0

    # Critical values
    cv = stats.norm.ppf(1 - (alp / 2))
    cv1 = (cv ** 2) / (2 * n1)
    cv2 = (cv / (2 * n1)) ** 2

    # SCORE (WILSON) METHOD
    for i in range(k):
        pAS[i] = y[i] / n1
        qAS[i] = 1 - pAS[i]
        seAS[i] = np.sqrt((pAS[i] * qAS[i] / n1) + cv2)
        LAS[i] = (n1 / (n1 + cv ** 2)) * ((pAS[i] + cv1) - (cv * seAS[i]))
        UAS[i] = (n1 / (n1 + cv ** 2)) * ((pAS[i] + cv1) + (cv * seAS[i]))
        LAS[i] = max(LAS[i], 0)
        UAS[i] = min(UAS[i], 1)

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    for j in range(s):
        for i in range(k):
            if LAS[i] < hp[j] < UAS[i]:
                cpAS[i, j] = stats.binom.pmf(i, n, hp[j])
        cppAS[j] = np.sum(cpAS[:, j])
        if t1 < cppAS[j] < t2:
            ctr += 1

    # Create dataframe
    CPAS = pd.DataFrame({
        'hp': hp,
        'cp': cppAS,
        'method': ['Adj-Wilson'] * s
    })

    return CPAS




def gcovpAAS(n, alp, h, a, b, t1, t2):
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

    # Input setup
    x = np.arange(0, n + 1)
    k = n + 1
    y = x + h
    n1 = n + (2 * h)

    # Initializations
    pAA = np.zeros(k)
    qAA = np.zeros(k)
    seAA = np.zeros(k)
    LAA = np.zeros(k)
    UAA = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpAA = np.zeros((k, s))
    cppAA = np.zeros(s)
    ctr = 0

    # Critical values
    cv = stats.norm.ppf(1 - (alp / 2))

    # ADJUSTED ARC-SINE METHOD
    for i in range(k):
        pAA[i] = y[i] / n1
        qAA[i] = 1 - pAA[i]
        seAA[i] = cv / np.sqrt(4 * n1)
        LAA[i] = np.sin(np.arcsin(np.sqrt(pAA[i])) - seAA[i]) ** 2
        UAA[i] = np.sin(np.arcsin(np.sqrt(pAA[i])) + seAA[i]) ** 2
        LAA[i] = max(LAA[i], 0)
        UAA[i] = min(UAA[i], 1)

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    for j in range(s):
        for i in range(k):
            if LAA[i] < hp[j] < UAA[i]:
                cpAA[i, j] = stats.binom.pmf(i, n, hp[j])
        cppAA[j] = np.sum(cpAA[:, j])
        if t1 < cppAA[j] < t2:
            ctr += 1

    # Create dataframe
    CPAA = pd.DataFrame({
        'hp': hp,
        'cp': cppAA,
        'method': ['Adj-ArcSine'] * s
    })

    return CPAA

def gcovpALT(n, alp, h, a, b, t1, t2):
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

    # Input setup
    x = np.arange(0, n + 1)
    k = n + 1
    y = x + h
    n1 = n + (2 * h)

    # Initializations
    pALT = np.zeros(k)
    qALT = np.zeros(k)
    lgit = np.zeros(k)
    seALT = np.zeros(k)
    LALT = np.zeros(k)
    UALT = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpALT = np.zeros((k, s))
    cppALT = np.zeros(s)
    ctr = 0

    # Critical values
    cv = stats.norm.ppf(1 - (alp / 2))

    # ADJUSTED LOGIT-WALD METHOD
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
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    for j in range(s):
        for i in range(k):
            if LALT[i] < hp[j] < UALT[i]:
                cpALT[i, j] = stats.binom.pmf(i, n, hp[j])
        cppALT[j] = np.sum(cpALT[:, j])
        if t1 < cppALT[j] < t2:
            ctr += 1

    # Create dataframe
    CPALT = pd.DataFrame({
        'hp': hp,
        'cp': cppALT,
        'method': ['Adj-Logit-Wald'] * s
    })

    return CPALT

def gcovpATW(n, alp, h, a, b, t1, t2):
    # Input setup
    x = np.arange(0, n + 1)
    k = n + 1
    y = x + h
    n1 = n + (2 * h)

    # Initializations
    pATW = np.zeros(k)
    qATW = np.zeros(k)
    seATW = np.zeros(k)
    LATW = np.zeros(k)
    UATW = np.zeros(k)
    DOF = np.zeros(k)
    cv = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpATW = np.zeros((k, s))
    cppATW = np.zeros(s)
    ctr = 0

    # Function definitions for f1 and f2
    def f1(p, n):
        return p * (1 - p) / n

    def f2(p, n):
        return (p * (1 - p) / (n**3)) + (
            (p + (6 * n - 7) * (p**2) + 4 * (n - 1) * (n - 3) * (p**3) -
             2 * (n - 1) * (2 * n - 3) * (p**4)) / (n**5)
            - 2 * (p + (2 * n - 3) * (p**2) - 2 * (n - 1) * (p**3)) / (n**4)
        )

    # MODIFIED t-WALD METHOD
    for i in range(k):
        pATW[i] = y[i] / n1
        qATW[i] = 1 - pATW[i]
        DOF[i] = 2 * (f1(pATW[i], n1)**2) / f2(pATW[i], n1)
        cv[i] = stats.t.ppf(1 - (alp / 2), df=DOF[i])
        seATW[i] = cv[i] * np.sqrt(f1(pATW[i], n1))
        LATW[i] = pATW[i] - seATW[i]
        UATW[i] = pATW[i] + seATW[i]
        LATW[i] = max(LATW[i], 0)
        UATW[i] = min(UATW[i], 1)

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    for j in range(s):
        for i in range(k):
            if LATW[i] < hp[j] < UATW[i]:
                cpATW[i, j] =stats.binom.pmf(i, n, hp[j])
        cppATW[j] = np.sum(cpATW[:, j])
        if t1 < cppATW[j] < t2:
            ctr += 1

    # Create dataframe
    CPATW = pd.DataFrame({
        'hp': hp,
        'cp': cppATW,
        'method': ['Adj-Wald-T'] * s
    })

    return CPATW


def gcovpALR(n, alp, h, a, b, t1, t2):
    # Input setup
    y = np.arange(0, n + 1)
    k = n + 1
    y1 = y + h
    n1 = n + (2 * h)

    # Initializations
    mle = np.zeros(k)
    cutoff = np.zeros(k)
    LAL = np.zeros(k)
    UAL = np.zeros(k)
    s = 5000  # Simulation run to generate hypothetical p
    cpAL = np.zeros((k, s))
    cppAL = np.zeros(s)
    ctr = 0

    # Critical value
    cv = stats.norm.ppf(1 - (alp / 2))

    # Adjusted Likelihood-Ratio Method
    for i in range(k):
        # Likelihood function
        def likelihood(p):
            return stats.binom.pmf(y1[i], n1, p)

        # Log-likelihood function
        def log_likelihood(p):
            return stats.binom.logpmf(y1[i], n1, p)

        # Maximum likelihood estimate (MLE)
        mle_res = optimize.minimize_scalar(lambda p: -likelihood(p), bounds=(0, 1), method='bounded')
        mle[i] = mle_res.x

        # Cutoff for likelihood
        cutoff[i] = log_likelihood(mle[i]) - (cv ** 2 / 2)

        # Optimization to find LAL (lower bound)
        def loglik_optim(p):
            return abs(cutoff[i] - log_likelihood(p))

        LAL_res = optimize.minimize_scalar(loglik_optim, bounds=(0, mle[i]), method='bounded')
        LAL[i] = LAL_res.x

        # Optimization to find UAL (upper bound)
        UAL_res = optimize.minimize_scalar(loglik_optim, bounds=(mle[i], 1), method='bounded')
        UAL[i] = UAL_res.x

    # Coverage probabilities
    hp = np.sort(stats.beta.rvs(a, b, size=s))

    for j in range(s):
        for i in range(k):
            if LAL[i] < hp[j] < UAL[i]:
                cpAL[i, j] = stats.binom.pmf(i, n, hp[j])
        cppAL[j] = np.sum(cpAL[:, j])
        if t1 < cppAL[j] < t2:
            ctr += 1

    # Create dataframe
    CPAL = pd.DataFrame({
        'hp': hp,
        'cp': cppAL,
        'method': ['Adj-Likelihood'] * s
    })
    return CPAL
n= 10; alp=0.05; h=2;a=1;b=1; t1=0.93;t2=0.97
print(gcovpALR(n,alp,h,a,b,t1,t2))