from CoverageProb_ADJ_All_212 import *
from CoverageProb_ADJ_internal_214 import *
import pandas as pd
import numpy as np
from plotnine import *





def plotcovpAWD(n, alp, h, a, b, t1, t2):
    # Input validations
    if n is None: raise ValueError("'n' is missing")
    if alp is None: raise ValueError("'alpha' is missing")
    if h is None: raise ValueError("'h' is missing")
    if a is None: raise ValueError("'a' is missing")
    if b is None: raise ValueError("'b' is missing")
    if t1 is None: raise ValueError("'t1' is missing")
    if t2 is None: raise ValueError("'t2' is missing")

    if not isinstance(n, (int, float)) or n <= 0: raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1): raise ValueError("'alpha' has to be between 0 and 1")
    if not isinstance(h, (int, float)) or h < 0: raise ValueError("'h' has to be greater than or equal to 0")
    if not isinstance(a, (int, float)) or a < 0: raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0: raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2: raise ValueError("t1 has to be lesser than t2")
    if not (0 <= t1 <= 1): raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1): raise ValueError("'t2' has to be between 0 and 1")

    # Generating data
    Waldcovp_df = covpAWD(n, alp, h, a, b, t1, t2)
    print(Waldcovp_df)
    nndf = gcovpAWD(n, alp, h, a, b, t1, t2)

    # Add t1, t2, and alp to nndf
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp
    nndf['mcp'] = Waldcovp_df['mcpAW'].iloc[0]
    nndf['micp'] = Waldcovp_df['micpAW'].iloc[0]

    # Plot using plotnine
    plot = (ggplot(nndf, aes(x='hp', y='cp'))
         + labs(title="Coverage Probability of the Adjusted Wald Method", y="Coverage Probability", x="p")
         + geom_line(aes(color='method'))
         + geom_hline(aes(yintercept='micp', color='"Minimum Coverage"'))
         + geom_hline(aes(yintercept='mcp', color='"Mean Coverage"'))
         + geom_hline(aes(yintercept=t1), color="#FF0000", linetype="dashed")
         + geom_hline(aes(yintercept=t2), color="#0000FF", linetype="dashed")
         + geom_text(aes(y=t1, label='"\\nLower tolerance (t1)"', x=0.1), color="#FF0000")
         + geom_text(aes(y=t2, label='"Higher tolerance (t2)"', x=0.1), color="#0000FF")
         + guides(colour=guide_legend("Heading"))
         + geom_hline(aes(yintercept=1 - alp), linetype="dashed"))

    plot.show()





def plotcovpAAS(n, alp, h, a, b, t1, t2):
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if h < 0 or a < 0 or b < 0:
        raise ValueError("'h', 'a', and 'b' have to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be less than or equal to 't2'")
    if not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' and 't2' have to be between 0 and 1")
    nndf = gcovpAAS(n, alp, h, a, b, t1, t2)
    ArcSinecovp_df = covpAAS(n, alp, h, a, b, t1, t2)

    # Apply mean and min coverage values
    nndf['mcp'] = ArcSinecovp_df['mcpAA'].iloc[0]
    nndf['micp'] = ArcSinecovp_df['micpAA'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plotting
    plot = (
        ggplot(nndf, aes(x='hp', y='cp'))
        + labs(title="Coverage Probability of the adjusted ArcSine method", y="Coverage Probability", x="p")
        + geom_line(aes(color='method'))
        + geom_hline(aes(yintercept='micp', color='"Minimum Coverage"'))
        + geom_hline(aes(yintercept='mcp', color='"Mean Coverage"'))
        + geom_hline(yintercept=t1, color="#FF0000", linetype="dashed")
        + geom_hline(yintercept=t2, color="#0000FF", linetype="dashed")
        + geom_text(aes(y=t1, label='"Lower tolerance (t1)"', x=0.1), color="#FF0000")
        + geom_text(aes(y=t2, label='"Higher tolerance (t2)"', x=0.1), color="#0000FF")
        + guides(color=guide_legend(title="Heading"))
        + geom_hline(yintercept=1 - alp, linetype="dashed")
    )
    plot.show()
def plotcovpALR(n, alp, h, a, b, t1, t2):
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if h < 0 or not float(h).is_integer():
        raise ValueError("'h' has to be an integer greater than or equal to 0")
    if a < 0 or b < 0:
        raise ValueError("'a' and 'b' have to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be less than or equal to 't2'")
    if not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' and 't2' have to be between 0 and 1")

    # Calling functions to create dataframes (replace with actual function implementations)
    nndf = gcovpALR(n, alp, h, a, b, t1, t2)
    LRcovp_df = covpALR(n, alp, h, a, b, t1, t2)

    # Apply mean and min coverage values
    nndf['mcp'] = LRcovp_df['mcpAL'].iloc[0]
    nndf['micp'] = LRcovp_df['micpAL'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plotting
    plot = (
        ggplot(nndf, aes(x='hp', y='cp'))
        + labs(title="Coverage Probability of the adjusted Likelihood Ratio method", y="Coverage Probability", x="p")
        + geom_line(aes(color='method'))
        + geom_hline(aes(yintercept='micp', color='"Minimum Coverage"'))
        + geom_hline(aes(yintercept='mcp', color='"Mean Coverage"'))
        + geom_hline(yintercept=t1, color="#FF0000", linetype="dashed")
        + geom_hline(yintercept=t2, color="#0000FF", linetype="dashed")
        + geom_text(aes(y=t1, label='"Lower tolerance (t1)"', x=0.1), color="#FF0000")
        + geom_text(aes(y=t2, label='"Higher tolerance (t2)"', x=0.1), color="#0000FF")
        + guides(color=guide_legend(title="Heading"))
        + geom_hline(yintercept=1 - alp, linetype="dashed")
    )
    plot.show()





def plotcovpASC(n, alp, h, a, b, t1, t2):
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if h < 0:
        raise ValueError("'h' has to be greater than or equal to 0")
    if a < 0 or b < 0:
        raise ValueError("'a' and 'b' have to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be less than or equal to 't2'")
    if not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' and 't2' have to be between 0 and 1")

    # Calling functions to create dataframes
    nndf = gcovpASC(n, alp, h, a, b, t1, t2)
    Scorecovp_df = covpASC(n, alp, h, a, b, t1, t2)

    # Apply mean and min coverage values
    nndf['mcp'] = Scorecovp_df['mcpAS'].iloc[0]
    nndf['micp'] = Scorecovp_df['micpAS'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plotting
    plot = (
        ggplot(nndf, aes(x='hp', y='cp'))
        + labs(title="Coverage Probability of the adjusted Score method", y="Coverage Probability", x="p")
        + geom_line(aes(color='method'))
        + geom_hline(aes(yintercept='micp', color='"Minimum Coverage"'))
        + geom_hline(aes(yintercept='mcp', color='"Mean Coverage"'))
        + geom_hline(yintercept=t1, color="#FF0000", linetype="dashed")
        + geom_hline(yintercept=t2, color="#0000FF", linetype="dashed")
        + geom_text(aes(y=t1, label='"Lower tolerance (t1)"', x=0.1), color="#FF0000")
        + geom_text(aes(y=t2, label='"Higher tolerance (t2)"', x=0.1), color="#0000FF")
        + guides(color=guide_legend(title="Heading"))
        + geom_hline(yintercept=1 - alp, linetype="dashed")
    )
    plot.show()





def plotcovpALT(n, alp, h, a, b, t1, t2):
    # Error checks
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if h < 0:
        raise ValueError("'h' has to be greater than or equal to 0")
    if a < 0 or b < 0:
        raise ValueError("'a' and 'b' have to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be less than or equal to 't2'")
    if not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' and 't2' have to be between 0 and 1")

    # Call functions to create dataframes
    nndf = gcovpALT(n, alp, h, a, b, t1, t2)
    WaldLcovp_df = covpALT(n, alp, h, a, b, t1, t2)

    # Apply mean and min coverage values
    nndf['mcp'] = WaldLcovp_df['mcpALT'].iloc[0]
    nndf['micp'] = WaldLcovp_df['micpALT'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plotting
    plot = (
        ggplot(nndf, aes(x='hp', y='cp'))
        + labs(title="Coverage Probability of the adjusted Logistic Wald method", y="Coverage Probability", x="p")
        + geom_line(aes(color='method'))
        + geom_hline(aes(yintercept='micp', color='"Minimum Coverage"'))
        + geom_hline(aes(yintercept='mcp', color='"Mean Coverage"'))
        + geom_hline(yintercept=t1, color="#FF0000", linetype="dashed")
        + geom_hline(yintercept=t2, color="#0000FF", linetype="dashed")
        + geom_text(aes(y=t1, label='"Lower tolerance (t1)"', x=0.1), color="#FF0000")
        + geom_text(aes(y=t2, label='"Higher tolerance (t2)"', x=0.1), color="#0000FF")
        + guides(color=guide_legend(title="Heading"))
        + geom_hline(yintercept=1 - alp, linetype="dashed")
    )
    plot.show()




def plotcovpATW(n, alp, h, a, b, t1, t2):
    # Error checks
    if n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if h < 0:
        raise ValueError("'h' has to be greater than or equal to 0")
    if a < 0 or b < 0:
        raise ValueError("'a' and 'b' have to be greater than or equal to 0")
    if t1 > t2:
        raise ValueError("'t1' has to be less than or equal to 't2'")
    if not (0 <= t1 <= 1) or not (0 <= t2 <= 1):
        raise ValueError("'t1' and 't2' have to be between 0 and 1")

    # Call functions to create dataframes
    nndf = gcovpATW(n, alp, h, a, b, t1, t2)
    AdWaldcovp_df = covpATW(n, alp, h, a, b, t1, t2)

    # Apply mean and min coverage values
    nndf['mcp'] = AdWaldcovp_df['mcpATW'].iloc[0]
    nndf['micp'] = AdWaldcovp_df['micpATW'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plotting
    plot = (
        ggplot(nndf, aes(x='hp', y='cp'))
        + labs(title="Coverage Probability of the adjusted Wald-T method", y="Coverage Probability", x="p")
        + geom_line(aes(color='method'))
        + geom_hline(aes(yintercept='micp', color='"Minimum Coverage"'))
        + geom_hline(aes(yintercept='mcp', color='"Mean Coverage"'))
        + geom_hline(yintercept=t1, color="#FF0000", linetype="dashed")
        + geom_hline(yintercept=t2, color="#0000FF", linetype="dashed")
        + geom_text(aes(y=t1, label='"Lower tolerance (t1)"', x=0.1), color="#FF0000")
        + geom_text(aes(y=t2, label='"Higher tolerance (t2)"', x=0.1), color="#0000FF")
        + guides(color=guide_legend(title="Heading"))
        + geom_hline(yintercept=1 - alp, linetype="dashed")
    )
    plot.show()








#SJTP