import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

def ciWDx(x, n, alp):
    # Input validation
    if x is None:
        raise ValueError("'x' is missing")
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if not (isinstance(x, (int, float)) and 0 <= x <= n):
        raise ValueError("'x' has to be a positive integer between 0 and n")
    if not (isinstance(n, (int, float)) and n > 0):
        raise ValueError("'n' has to be greater than 0")
    if not (0 < alp < 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    
    # Critical value
    cv = stats.norm.ppf(1 - (alp / 2))
    
    # Wald method calculations
    pW = x / n
    qW = 1 - pW
    seW = np.sqrt(pW * qW / n)
    
    # Calculate lower and upper bounds
    LWDx = pW - (cv * seW)
    UWDx = pW + (cv * seW)
    
    # Adjust bounds if they exceed 0 or 1
    LABB = "YES" if LWDx < 0 else "NO"
    UABB = "YES" if UWDx > 1 else "NO"
    LWDx = max(LWDx, 0)
    UWDx = min(UWDx, 1)
    
    # Check for zero-width intervals
    ZWI = "YES" if UWDx - LWDx == 0 else "NO"
    
    # Return the results as a DataFrame
    return pd.DataFrame({
        'x': [x],
        'LWDx': [LWDx],
        'UWDx': [UWDx],
        'LABB': [LABB],
        'UABB': [UABB],
        'ZWI': [ZWI]
    })

def ciWDx(x, n, alp):
    # Input validation
    if x is None:
        raise ValueError("'x' is missing")
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if not (isinstance(x, (int, float)) and 0 <= x <= n):
        raise ValueError("'x' has to be a positive integer between 0 and n")
    if not (isinstance(n, (int, float)) and n > 0):
        raise ValueError("'n' has to be greater than 0")
    if not (0 < alp < 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    
    # Critical value
    cv =stats.norm.ppf(1 - (alp / 2))
    
    # Wald method calculations
    pW = x / n
    qW = 1 - pW
    seW = np.sqrt(pW * qW / n)
    
    # Calculate lower and upper bounds
    LWDx = pW - (cv * seW)
    UWDx = pW + (cv * seW)
    
    # Adjust bounds if they exceed 0 or 1
    LABB = "YES" if LWDx < 0 else "NO"
    UABB = "YES" if UWDx > 1 else "NO"
    LWDx = max(LWDx, 0)
    UWDx = min(UWDx, 1)
    
    # Check for zero-width intervals
    ZWI = "YES" if UWDx - LWDx == 0 else "NO"
    
    # Return the results as a DataFrame
    return pd.DataFrame({
        'x': [x],
        'LWDx': [LWDx],
        'UWDx': [UWDx],
        'LABB': [LABB],
        'UABB': [UABB],
        'ZWI': [ZWI]
    })


def ciSCx(x, n, alp):
    # Input validation
    if x is None:
        raise ValueError("'x' is missing")
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if not (isinstance(x, (int, float)) and 0 <= x <= n):
        raise ValueError("'x' has to be a positive integer between 0 and n")
    if not (isinstance(n, (int, float)) and n > 0):
        raise ValueError("'n' has to be greater than 0")
    if not (0 < alp < 1):
        raise ValueError("'alpha' has to be between 0 and 1")

    # Critical values
    cv = norm.ppf(1 - (alp / 2))
    cv1 = (cv ** 2) / (2 * n)
    cv2 = (cv / (2 * n)) ** 2
    # Score (Wilson) method calculations
    pS = x / n
    qS = 1 - pS
    seS = np.sqrt((pS * qS / n) + cv2)
    LSCx = (n / (n + cv ** 2)) * ((pS + cv1) - (cv * seS))
    USCx = (n / (n + cv ** 2)) * ((pS + cv1) + (cv * seS))

    # Adjust bounds if they exceed 0 or 1
    LABB = "YES" if LSCx < 0 else "NO"
    LSCx = max(LSCx, 0)
    UABB = "YES" if USCx > 1 else "NO"
    USCx = min(USCx, 1)

    # Check for zero-width intervals
    ZWI = "YES" if USCx - LSCx == 0 else "NO"
    
    # Return the results as a DataFrame
    return pd.DataFrame({
        'x': [x],
        'LSCx': [LSCx],
        'USCx': [USCx],
        'LABB': [LABB],
        'UABB': [UABB],
        'ZWI': [ZWI]
    })






def ciASx(x, n, alp):
    # Input validation
    if x is None:
        raise ValueError("'x' is missing")
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if not (isinstance(x, (int, float)) and 0 <= x <= n):
        raise ValueError("'x' has to be a positive integer between 0 and n")
    if not (isinstance(n, (int, float)) and n > 0):
        raise ValueError("'n' has to be greater than 0")
    if not (0 < alp < 1):
        raise ValueError("'alpha' has to be between 0 and 1")

    # Critical value
    cv = stats.norm.ppf(1 - (alp / 2))

    # Arc-Sine Method
    pA = x / n
    seA = cv / np.sqrt(4 * n)
    LASx = np.sin(np.arcsin(np.sqrt(pA)) - seA) ** 2
    UASx = np.sin(np.arcsin(np.sqrt(pA)) + seA) ** 2

    # Adjust bounds if they exceed 0 or 1
    LABB = "YES" if LASx < 0 else "NO"
    LASx = max(LASx, 0)
    UABB = "YES" if UASx > 1 else "NO"
    UASx = min(UASx, 1)

    # Check for zero-width intervals
    ZWI = "YES" if UASx - LASx == 0 else "NO"

    # Return the results as a DataFrame
    return pd.DataFrame({
        'x': [x],
        'LASx': [LASx],
        'UASx': [UASx],
        'LABB': [LABB],
        'UABB': [UABB],
        'ZWI': [ZWI]
    })




def ciLRx(x, n, alp):
    # Input validation
    if x is None:
        raise ValueError("'x' is missing")
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if not (isinstance(x, (int, float)) and 0 <= x <= n):
        raise ValueError("'x' has to be a positive integer between 0 and n")
    if not (isinstance(n, (int, float)) and n > 0):
        raise ValueError("'n' has to be greater than 0")
    if not (0 < alp < 1):
        raise ValueError("'alpha' has to be between 0 and 1")

    y = x
    # Critical value
    cv = stats.norm.ppf(1 - (alp / 2))

    # Likelihood function
    def likelhd(p):
        return stats.binom.pmf(y, n, p)
    
    # Log-likelihood function
    def loglik(p):
        return stats.binom.logpmf(y, n, p)
    
    # Maximum likelihood estimation (MLE)
    mle = optimize.minimize_scalar(lambda p: -likelhd(p), bounds=(0, 1), method='bounded').x
    
    # Log-likelihood cutoff value
    cutoff = loglik(mle) - (cv ** 2 / 2)
    
    # Optimization to find LLR and ULR
    def loglik_optim(p):
        return abs(cutoff - loglik(p))
    
    LLRx = optimize.minimize_scalar(loglik_optim, bounds=(0, mle), method='bounded').x
    ULRx = optimize.minimize_scalar(loglik_optim, bounds=(mle, 1), method='bounded').x

    # Adjust bounds if they exceed 0 or 1
    LABB = "YES" if LLRx < 0 else "NO"
    LLRx = max(LLRx, 0)
    UABB = "YES" if ULRx > 1 else "NO"
    ULRx = min(ULRx, 1)

    # Check for zero-width intervals
    ZWI = "YES" if ULRx - LLRx == 0 else "NO"

    # Return the results as a DataFrame
    return pd.DataFrame({
        'x': [x],
        'LLRx': [LLRx],
        'ULRx': [ULRx],
        'LABB': [LABB],
        'UABB': [UABB],
        'ZWI': [ZWI]
    })



import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

def ciEXx(x, n, alp, e):
    # Input validation
    if x is None:
        raise ValueError("'x' is missing")
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if e is None:
        raise ValueError("'e' is missing")
    if not (isinstance(x, (int, float)) and 0 <= x <= n):
        raise ValueError("'x' has to be a positive integer between 0 and n")
    if not (isinstance(n, (int, float)) and n > 0):
        raise ValueError("'n' has to be greater than 0")
    if not (0 < alp < 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if isinstance(e, (float, int)):
        e = [e]
    if not (isinstance(e, (list, np.ndarray)) and all(0 <= i <= 1 for i in e)):
        raise ValueError("'e' has to be a list, array, or single value with values between 0 and 1")
    if len(e) > 10:
        raise ValueError("'e' can have only 10 intervals")

    result = pd.DataFrame()


    for e_val in e:
        lu = lufn103(x, n, alp, e_val)
        result = pd.concat([result, lu])

    return result

def lufn103(x, n, alp, e):
    LEXx = exlim103l(x, n, alp, e)
    UEXx = exlim103u(x, n, alp, e)

    # Adjust limits if out of [0, 1]
    LABB = "YES" if LEXx < 0 else "NO"
    LEXx = max(LEXx, 0)
    UABB = "YES" if UEXx > 1 else "NO"
    UEXx = min(UEXx, 1)

    # Zero-width interval check
    ZWI = "YES" if UEXx - LEXx == 0 else "NO"

    # Return as DataFrame
    return pd.DataFrame({
        'x': [x],
        'LEXx': [LEXx],
        'UEXx': [UEXx],
        'LABB': [LABB],
        'UABB': [UABB],
        'ZWI': [ZWI],
        'e': [e]
    })

def exlim103l(x, n, alp, e):
    if x == 0:
        return 0
    elif x == n:
        return (alp / (2 * e)) ** (1 / n)
    else:
        z = x - 1
        y = np.arange(0, z + 1)

        def f1(p):
            return (1 - e) * stats.binom.pmf(x, n, p) + np.sum(stats.binom.pmf(y, n, p)) - (1 - (alp / 2))

        sol = optimize.root_scalar(f1, bracket=[0, 1], method='bisect')
        return sol.root if sol.converged else None

def exlim103u(x, n, alp, e):
    if x == 0:
        return 1 - (alp / (2 * e)) ** (1 / n)
    elif x == n:
        return 1
    else:
        z = x - 1
        y = np.arange(0, z + 1)

        def f2(p):
            return e * stats.binom.pmf(x, n, p) + np.sum(stats.binom.pmf(y, n, p)) - (alp / 2)
        sol = optimize.root_scalar(f2, bracket=[0, 1], method='bisect')
        return sol.root if sol.converged else None


def ciTWx(x, n, alp):
    if x is None:
        raise ValueError("'x' is missing")
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if not isinstance(x, (int, float)) or x < 0 or x > n or len(np.atleast_1d(x)) > 1:
        raise ValueError("'x' has to be a positive integer between 0 and n")
    if not isinstance(n, (int, float)) or len(np.atleast_1d(n)) > 1 or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if alp > 1 or alp < 0 or len(np.atleast_1d(alp)) > 1:
        raise ValueError("'alpha' has to be between 0 and 1")

    # MODIFIED t-WALD METHOD
    if x == 0 or x == n:
        pTWx = (x + 2) / (n + 4)
    else:
        pTWx = x / n

    # Helper functions for calculations
    def f1(p, n):
        return p * (1 - p) / n

    def f2(p, n):
        term1 = p * (1 - p) / (n**3)
        term2 = (p + ((6 * n) - 7) * (p**2) + (4 * (n - 1) * (n - 3) * (p**3)) - (2 * (n - 1) * ((2 * n) - 3) * (p**4))) / (n**5)
        term3 = (2 * (p + ((2 * n) - 3) * (p**2) - 2 * (n - 1) * (p**3))) / (n**4)
        return term1 + term2 - term3

    DOFx = 2 * (f1(pTWx, n)**2) / f2(pTWx, n)
    cvx = stats.t.ppf(1 - (alp / 2), df=DOFx)
    seTWx = cvx * np.sqrt(f1(pTWx, n))
    LTWx = pTWx - seTWx
    UTWx = pTWx + seTWx

    # Boundary checks
    LABB = "YES" if LTWx < 0 else "NO"
    LTWx = max(LTWx, 0)
    UABB = "YES" if UTWx > 1 else "NO"
    UTWx = min(UTWx, 1)
    ZWI = "YES" if UTWx - LTWx == 0 else "NO"

    result_df = pd.DataFrame({
        'x': [x],
        'LTWx': [LTWx],
        'UTWx': [UTWx],
        'LABB': [LABB],
        'UABB': [UABB],
        'ZWI': [ZWI]
    })

    return result_df






def ciLTx(x, n, alp):
    if x is None:
        raise ValueError("'x' is missing")
    if n is None:
        raise ValueError("'n' is missing")
    if alp is None:
        raise ValueError("'alpha' is missing")
    if not isinstance(x, (int, float)) or x < 0 or x > n or len(np.atleast_1d(x)) > 1:
        raise ValueError("'x' has to be a positive integer between 0 and n")
    if not isinstance(n, (int, float)) or len(np.atleast_1d(n)) > 1 or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if alp > 1 or alp < 0 or len(np.atleast_1d(alp)) > 1:
        raise ValueError("'alpha' has to be between 0 and 1")

    # CRITICAL VALUES
    cv = stats.norm.ppf(1 - (alp / 2))

    # Initialize variables
    LLTx, ULTx = 0, 0
    LABB, UABB, ZWI = "NO", "NO", "NO"

    # LOGIT-WALD METHOD
    if x == 0:
        pLTx = 0
        qLTx = 1
        LLTx = 0
        ULTx = 1 - (alp / 2) ** (1 / n)
    elif x == n:
        pLTx = 1
        qLTx = 0
        LLTx = (alp / 2) ** (1 / n)
        ULTx = 1
    else:
        pLTx = x / n
        qLTx = 1 - pLTx
        lgitx = np.log(pLTx / qLTx)
        seLTx = np.sqrt(pLTx * qLTx * n)
        LLTx = 1 / (1 + np.exp(-lgitx + (cv / seLTx)))
        ULTx = 1 / (1 + np.exp(-lgitx - (cv / seLTx)))

    # Boundary checks
    if LLTx < 0:
        LABB = "YES"
        LLTx = 0

    if ULTx > 1:
        UABB = "YES"
        ULTx = 1

    if ULTx - LLTx == 0:
        ZWI = "YES"

    # Create DataFrame
    result_df = pd.DataFrame({
        'x': [x],
        'LLTx': [LLTx],
        'ULTx': [ULTx],
        'LABB': [LABB],
        'UABB': [UABB],
        'ZWI': [ZWI]
    })

    return result_df

def hpd_beta(shape1, shape2, conf):
    def interval_length(alpha):
        lower = stats.beta.ppf(alpha, shape1, shape2)
        upper = stats.beta.ppf(alpha + conf, shape1, shape2)
        return upper - lower

    result = optimize.minimize_scalar(interval_length, bounds=(0, 1 - conf), method='bounded')

    if result.success:
        lower = stats.beta.ppf(result.x, shape1, shape2)
        upper = stats.beta.ppf(result.x + conf, shape1, shape2)
        return lower, upper
    else:
        raise RuntimeError("HPD estimation failed")



def ciWDx(x, n, alp):
    return pd.DataFrame({
        'x': [x],
        'LWDx': [0.1],
        'UWDx': [0.9],
        'LABB': ['NO'],
        'UABB': ['NO'],
        'ZWI': ['NO']
    })

def ciASx(x, n, alp):
    return pd.DataFrame({
        'x': [x],
        'LASx': [0.12],
        'UASx': [0.88],
        'LABB': ['YES'],
        'UABB': ['NO'],
        'ZWI': ['NO']
    })

def ciLRx(x, n, alp):
    return pd.DataFrame({
        'x': [x],
        'LLRx': [0.15],
        'ULRx': [0.85],
        'LABB': ['NO'],
        'UABB': ['YES'],
        'ZWI': ['NO']
    })

def ciSCx(x, n, alp):
    return pd.DataFrame({
        'x': [x],
        'LSCx': [0.2],
        'USCx': [0.8],
        'LABB': ['NO'],
        'UABB': ['NO'],
        'ZWI': ['YES']
    })

def ciLTx(x, n, alp):
    return pd.DataFrame({
        'x': [x],
        'LLTx': [0.18],
        'ULTx': [0.82],
        'LABB': ['NO'],
        'UABB': ['NO'],
        'ZWI': ['NO']
    })

def ciTWx(x, n, alp):
    return pd.DataFrame({
        'x': [x],
        'LTWx': [0.14],
        'UTWx': [0.86],
        'LABB': ['NO'],
        'UABB': ['NO'],
        'ZWI': ['NO']
    })

# Replace ciAllx function with the actual logic using the mock CIs
def ciAllx(x, n, alp):
    # Combine the outputs of the individual CI methods into a DataFrame
    WaldCI_df = ciWDx(x, n, alp)
    ArcSineCI_df = ciASx(x, n, alp)
    LRCI_df = ciLRx(x, n, alp)
    ScoreCI_df = ciSCx(x, n, alp)
    WaldLCI_df = ciLTx(x, n, alp)
    AdWaldCI_df = ciTWx(x, n, alp)

    # Combine all DataFrames
    Final_df = pd.concat([
        pd.DataFrame({
            'method': 'Wald',
            'x': WaldCI_df['x'],
            'LowerLimit': WaldCI_df['LWDx'],
            'UpperLimit': WaldCI_df['UWDx'],
            'LowerAbb': WaldCI_df['LABB'],
            'UpperAbb': WaldCI_df['UABB'],
            'ZWI': WaldCI_df['ZWI']
        }),
        pd.DataFrame({
            'method': 'ArcSine',
            'x': ArcSineCI_df['x'],
            'LowerLimit': ArcSineCI_df['LASx'],
            'UpperLimit': ArcSineCI_df['UASx'],
            'LowerAbb': ArcSineCI_df['LABB'],
            'UpperAbb': ArcSineCI_df['UABB'],
            'ZWI': ArcSineCI_df['ZWI']
        }),
        pd.DataFrame({
            'method': 'Likelihood',
            'x': LRCI_df['x'],
            'LowerLimit': LRCI_df['LLRx'],
            'UpperLimit': LRCI_df['ULRx'],
            'LowerAbb': LRCI_df['LABB'],
            'UpperAbb': LRCI_df['UABB'],
            'ZWI': LRCI_df['ZWI']
        }),
        pd.DataFrame({
            'method': 'Score',
            'x': ScoreCI_df['x'],
            'LowerLimit': ScoreCI_df['LSCx'],
            'UpperLimit': ScoreCI_df['USCx'],
            'LowerAbb': ScoreCI_df['LABB'],
            'UpperAbb': ScoreCI_df['UABB'],
            'ZWI': ScoreCI_df['ZWI']
        }),
        pd.DataFrame({
            'method': 'Logit-Wald',
            'x': WaldLCI_df['x'],
            'LowerLimit': WaldLCI_df['LLTx'],
            'UpperLimit': WaldLCI_df['ULTx'],
            'LowerAbb': WaldLCI_df['LABB'],
            'UpperAbb': WaldLCI_df['UABB'],
            'ZWI': WaldLCI_df['ZWI']
        }),
        pd.DataFrame({
            'method': 'Wald-T',
            'x': AdWaldCI_df['x'],
            'LowerLimit': AdWaldCI_df['LTWx'],
            'UpperLimit': AdWaldCI_df['UTWx'],
            'LowerAbb': AdWaldCI_df['LABB'],
            'UpperAbb': AdWaldCI_df['UABB'],
            'ZWI': AdWaldCI_df['ZWI']
        })
    ], ignore_index=True)

    return Final_df
x= 5; n=5; alp=0.05
print(ciAllx(x,n,alp))