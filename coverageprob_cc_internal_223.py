import numpy as np
import pandas as pd
from scipy.stats import norm, beta, binom, t

def gcovpcwd(n, alp, c, a, b, t1, t2):
    x = np.arange(0, n + 1)
    k = n + 1

    pCW = np.zeros(k)
    seCW = np.zeros(k)
    LCW = np.zeros(k)
    UCW = np.zeros(k)
    s = 5000
    cpCW = np.zeros((k, s))
    cppCW = np.zeros(s)
    ctr = 0

    cv = norm.ppf(1 - (alp / 2))

    for i in range(k):
        pCW[i] = x[i] / n
        qCW = 1 - pCW[i]
        seCW[i] = np.sqrt(pCW[i] * qCW / n)
        margin = cv * seCW[i] + c
        LCW[i] = max(0, pCW[i] - margin)
        UCW[i] = min(1, pCW[i] + margin)

    hp = np.sort(beta.rvs(a, b, size=s))
    for j in range(s):
        for i in range(k):
            if LCW[i] < hp[j] < UCW[i]:
                cpCW[i, j] = binom.pmf(i - 1, n, hp[j])
        cppCW[j] = np.sum(cpCW[:, j])
        if t1 < cppCW[j] < t2:
            ctr += 1

    CPCW = pd.DataFrame({'hp': hp, 'cp': cppCW, 'method': "Continuity corrected Wald"})
    return CPCW

def gcovpcsc(n, alp, c, a, b, t1, t2):
    x = np.arange(0, n + 1)
    k = n + 1

    pCS = np.zeros(k)
    seCS_L = np.zeros(k)
    seCS_U = np.zeros(k)
    LCS = np.zeros(k)
    UCS = np.zeros(k)
    s = 5000
    cpCS = np.zeros((k, s))
    cppCS = np.zeros(s)
    ctr = 0

    cv = norm.ppf(1 - (alp / 2))
    cv1 = (cv ** 2) / (2 * n)
    cv2 = cv / (2 * n)

    for i in range(k):
        pCS[i] = x[i] / n
        qCS = 1 - pCS[i]
        seCS_L[i] = np.sqrt((cv ** 2) - (4 * n * (c + c ** 2)) + (4 * n * pCS[i] * (1 - pCS[i] + (2 * c))))
        seCS_U[i] = np.sqrt((cv ** 2) + (4 * n * (c - c ** 2)) + (4 * n * pCS[i] * (1 - pCS[i] - (2 * c))))
        LCS[i] = (n / (n + (cv) ** 2)) * ((pCS[i] - c + cv1) - (cv2 * seCS_L[i]))
        UCS[i] = (n / (n + (cv) ** 2)) * ((pCS[i] + c + cv1) + (cv2 * seCS_U[i]))
        LCS[i] = max(0, LCS[i])
        UCS[i] = min(1, UCS[i])

    hp = np.sort(beta.rvs(a, b, size=s))
    for j in range(s):
        for i in range(k):
            if LCS[i] < hp[j] < UCS[i]:
                cpCS[i, j] = binom.pmf(i - 1, n, hp[j])
        cppCS[j] = np.sum(cpCS[:, j])
        if t1 < cppCS[j] < t2:
            ctr += 1

    CPCS = pd.DataFrame({'hp': hp, 'cp': cppCS, 'method': "Continuity corrected Wilson"})
    return CPCS

def gcovpcas(n, alp, c, a, b, t1, t2):
    x = np.arange(0, n + 1)
    k = n + 1

    pCA = np.zeros(k)
    qCA = np.zeros(k)
    seCA = np.zeros(k)
    LCA = np.zeros(k)
    UCA = np.zeros(k)
    s = 5000
    cpCA = np.zeros((k, s))
    cppCA = np.zeros(s)
    ctr = 0

    cv = norm.ppf(1 - (alp / 2))

    for i in range(k):
        pCA[i] = x[i] / n
        qCA[i] = 1 - pCA[i]
        seCA[i] = cv / np.sqrt(4 * n)
        LCA[i] = (np.sin(np.arcsin(np.sqrt(pCA[i])) - seCA[i] - c)) ** 2
        UCA[i] = (np.sin(np.arcsin(np.sqrt(pCA[i])) + seCA[i] + c)) ** 2
        LCA[i] = max(0, LCA[i])
        UCA[i] = min(1, UCA[i])

    hp = np.sort(beta.rvs(a, b, size=s))
    for j in range(s):
        for i in range(k):
            if LCA[i] < hp[j] < UCA[i]:
                cpCA[i, j] = binom.pmf(i - 1, n, hp[j])
        cppCA[j] = np.sum(cpCA[:, j])
        if t1 < cppCA[j] < t2:
            ctr += 1

    CPCA = pd.DataFrame({'hp': hp, 'cp': cppCA, 'method': "Continuity corrected ArcSine"})
    return CPCA

def gcovpclt(n, alp, c, a, b, t1, t2):
    x = np.arange(0, n + 1)
    k = n + 1

    pCLT = np.zeros(k)
    qCLT = np.zeros(k)
    seCLT = np.zeros(k)
    lgit = np.zeros(k)
    LCLT = np.zeros(k)
    UCLT = np.zeros(k)
    s = 5000
    cpCLT = np.zeros((k, s))
    cppCLT = np.zeros(s)
    ctr = 0

    cv = norm.ppf(1 - (alp / 2))

    pCLT[0] = 0
    qCLT[0] = 1
    LCLT[0] = 0
    UCLT[0] = 1 - ((alp / 2) ** (1 / n))

    pCLT[-1] = 1
    qCLT[-1] = 0
    LCLT[-1] = (alp / 2) ** (1 / n)
    UCLT[-1] = 1

    lgiti = lambda t: np.exp(t) / (1 + np.exp(t))
    for j in range(1, k - 1):
        pCLT[j] = x[j] / n
        qCLT[j] = 1 - pCLT[j]
        lgit[j] = np.log(pCLT[j] / qCLT[j])
        seCLT[j] = np.sqrt(pCLT[j] * qCLT[j] * n)
        LCLT[j] = lgiti(lgit[j] - (cv / seCLT[j]) - c)
        UCLT[j] = lgiti(lgit[j] + (cv / seCLT[j]) + c)

    LCLT = np.clip(LCLT, 0, 1)
    UCLT = np.clip(UCLT, 0, 1)

    hp = np.sort(beta.rvs(a, b, size=s))
    for j in range(s):
        for i in range(k):
            if LCLT[i] < hp[j] < UCLT[i]:
                cpCLT[i, j] = binom.pmf(i - 1, n, hp[j])
        cppCLT[j] = np.sum(cpCLT[:, j])
        if t1 < cppCLT[j] < t2:
            ctr += 1

    CPLT = pd.DataFrame({'hp': hp, 'cp': cppCLT, 'method': "Continuity corrected Logit Wald"})
    return CPLT

def gcovpctw(n, alp, c, a, b, t1, t2):
    x = np.arange(0, n + 1)
    k = n + 1

    pCTW = np.zeros(k)
    qCTW = np.zeros(k)
    DOF = np.zeros(k)
    cv = np.zeros(k)
    seCTW = np.zeros(k)
    LCTW = np.zeros(k)
    UCTW = np.zeros(k)
    s = 5000
    cpCTW = np.zeros((k, s))
    cppCTW = np.zeros(s)
    ctr = 0

    for i in range(k):
        pCTW[i] = (x[i] + 2) / (n + 4) if x[i] in [0, n] else x[i] / n
        qCTW[i] = 1 - pCTW[i]

        f1 = lambda p, n: p * (1 - p) / n
        f2 = lambda p, n: (p * (1 - p) / (n ** 3) + (p + ((6 * n) - 7) * (p ** 2) +
                     (4 * (n - 1) * (n - 3) * (p ** 3)) - (2 * (n - 1) * ((2 * n) - 3) * (p ** 4))) / (n ** 5) -
                     (2 * (p + ((2 * n) - 3) * (p ** 2) - 2 * (n - 1) * (p ** 3))) / (n ** 4))

        DOF[i] = 2 * (f1(pCTW[i], n) ** 2) / f2(pCTW[i], n)
        cv[i] = t.ppf(1 - (alp / 2), DOF[i])
        seCTW[i] = cv[i] * np.sqrt(f1(pCTW[i], n))
        LCTW[i] = pCTW[i] - (seCTW[i] + c)
        UCTW[i] = pCTW[i] + (seCTW[i] + c)

        LCTW[i] = max(LCTW[i], 0)
        UCTW[i] = min(UCTW[i], 1)

    hp = np.sort(beta.rvs(a, b, size=s))
    for j in range(s):
        for i in range(k):
            if LCTW[i] < hp[j] < UCTW[i]:
                cpCTW[i, j] = binom.pmf(i - 1, n, hp[j])

        cppCTW[j] = np.sum(cpCTW[:, j])
        if t1 < cppCTW[j] < t2:
            ctr += 1

    CPCTW = pd.DataFrame({'hp': hp, 'cp': cppCTW, 'method': "Continuity corrected Wald-T"})
    return CPCTW
