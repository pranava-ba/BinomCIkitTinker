import pandas as pd
import numpy as np
from debugpy.launcher.debuggee import describe
from plotnine import *
from CoverageProb_BASE_internal_203 import *
from CoverageProb_BASE_All_201 import *




def plotcovpEX(n, alp, e, a, b, t1, t2):
    if n is None: raise ValueError("'n' is missing")
    if alp is None: raise ValueError("'alpha' is missing")
    if e is None: raise ValueError("'e' is missing")
    if a is None: raise ValueError("'a' is missing")
    if b is None: raise ValueError("'b' is missing")
    if t1 is None: raise ValueError("'t1' is missing")
    if t2 is None: raise ValueError("'t2' is missing")
    if not isinstance(n, (int, float)) or n <= 0: raise ValueError("'n' has to be greater than 0")
    if not (0 <= alp <= 1): raise ValueError("'alpha' has to be between 0 and 1")
    if isinstance(e, (list, np.ndarray)):
        e = np.asarray(e)
    elif isinstance(e, (int, float)):
        e = np.array([e])
    else:
        raise ValueError("'e' has to be a float, list, or numpy array")
    if any(val > 1 or val < 0 for val in e):
        raise ValueError("'e' has to be between 0 and 1")

    if len(e) > 10: raise ValueError("Plot of only 10 intervals of 'e' is possible")
    if not isinstance(a, (int, float)) or a < 0: raise ValueError("'a' has to be greater than or equal to 0")
    if not isinstance(b, (int, float)) or b < 0: raise ValueError("'b' has to be greater than or equal to 0")
    if t1 > t2: raise ValueError("t1 has to be lesser than t2")
    if not (0 <= t1 <= 1): raise ValueError("'t1' has to be between 0 and 1")
    if not (0 <= t2 <= 1): raise ValueError("'t2' has to be between 0 and 1")
    dfex=gcovpEX(n, alp, e, a, b, t1, t2)
    exdf = dfex.iloc[:, [0, 1, 4]].copy()
    exdf['e'] = exdf['e'].astype('category')
    exdf['t1'] = t1
    exdf['t2'] = t2
    exdf['alp'] = alp
    
    # Create label DataFrame for the tolerance lines
    label_df = pd.DataFrame({
        'x': [0.1, 0.1],
        'y': [t1, t2],
        'label': ['Lower tolerance (t1)', 'Upper tolerance (t2)'],
        'color': ['#FF0000', '#0000FF']
    })

    if len(e) > 1:
        plot = (ggplot(exdf, aes(x='hp', y='cpp', color='e'))
                + labs(y="Coverage Probability",
                       title="Coverage Probability for Exact Method for Multiple e Values",
                       x="p")
                + geom_hline(yintercept=t1, color="#FF0000", linetype="dashed")
                + geom_hline(yintercept=t2, color="#0000FF", linetype="dashed")
                + geom_text(data=label_df,
                            mapping=aes(x='x', y='y', label='label'),
                            color=label_df['color'].tolist())
                + geom_line()
                + geom_hline(yintercept=1 - alp, linetype="dashed", color="#A52A2A"))
    else:
        dfex1 = pd.DataFrame({
            'micp': [dfex['micpEX'].iloc[0]],
            'mcp': [dfex['mcpEX'].iloc[0]]
        })
        dfex1['alp'] = alp

        plot = (ggplot(dfex, aes(x='hp', y='cpp'))
                + labs(title="Coverage Probability of Exact Method",
                       y="Coverage Probability",
                       x="p")
                + geom_line(color="#FF0000")
                + geom_point(color="#FF0000")
                + geom_hline(yintercept=1 - alp, color="#A52A2A", linetype="dashed")
                + geom_hline(data=dfex1, mapping=aes(yintercept='micp'), color="#000000")
                + geom_hline(data=dfex1, mapping=aes(yintercept='mcp'), color="#0000FF")
                + scale_colour_manual(name='Heading',
                                      values={'Coverage Probability': '#FF0000',
                                              'CP Values': '#FF0000',
                                              'Minimum Coverage': '#000000',
                                              'Mean Coverage': '#0000FF',
                                              'Confidence Level': '#A52A2A'},
                                      guide='legend')
                + guides(colour=guide_legend(override_aes={'linetype': ['dashed', 'solid', 'solid', 'solid', 'solid'],
                                                           'shape': [None, None, 16, None, None]})))
    plot.show()



def plotcovpWD(n, alp, a, b, t1, t2):
    # Input validation
    if n is None or n <= 0:
        raise ValueError("'n' is missing or invalid; it must be greater than 0")
    if alp is None or not (0 <= alp <= 1):
        raise ValueError("'alpha' is missing or out of range; it must be between 0 and 1")
    if a is None or a < 0:
        raise ValueError("'a' is missing or invalid; it must be greater than or equal to 0")
    if b is None or b < 0:
        raise ValueError("'b' is missing or invalid; it must be greater than or equal to 0")
    if t1 is None or not (0 <= t1 <= 1):
        raise ValueError("'t1' is missing or out of range; it must be between 0 and 1")
    if t2 is None or not (0 <= t2 <= 1):
        raise ValueError("'t2' is missing or out of range; it must be between 0 and 1")
    if t1 > t2:
        raise ValueError("t1 has to be less than t2")

    Waldcovp_df = covpWD(n, alp, a, b, t1, t2)
    nndf = gcovpW(n, alp, a, b, t1, t2)
    nndf['mcp'] = Waldcovp_df['mcpW'].iloc[0]
    nndf['micp'] = Waldcovp_df['micpW'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plot creation using plotnine
    plot = (ggplot(nndf, aes(x='hp', y='cp'))
            + ggtitle("Coverage Probability for Wald method")
            + labs(x="p", y="Coverage Probability")
            + geom_hline(aes(yintercept='t1'), color="#FF0000", linetype="dashed")
            + geom_hline(aes(yintercept='t2'), color="#0000FF", linetype="dashed")
            + geom_text(aes(x=0.1, y='t1'), label=f"Lower tolerance (t1 = {t1})", color="#FF0000", ha='left')
            + geom_text(aes(x=0.1, y='t2'), label=f"Higher tolerance (t2 = {t2})", color="#0000FF", ha='left')
            + geom_hline(aes(yintercept='micp', color="'Minimum Coverage'"))
            + geom_hline(aes(yintercept='mcp', color="'Mean Coverage'"))
            + geom_line(aes(color="'Coverage Probability'"))
            + scale_color_manual(name="Heading", values={
                'Coverage Probability': '#FF0000',
                'Minimum Coverage': '#000000',
                'Mean Coverage': '#0000FF'
            })
            + geom_hline(yintercept=1 - alp, linetype="dashed")
            + xlim(0, 1)  # Set x-axis range from 0 to 1
            + ylim(0, 1)  # Set y-axis range from 0 to 1
            )
    plot.show()


def plotcovpAS(n, alp, a, b, t1, t2):
    # Input validation
    if n is None or not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if alp is None or not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a is None or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b is None or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 is None or not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if t2 is None or not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")
    if t1 > t2:
        raise ValueError("t1 has to be less than t2")

    # Generate data
    ArcSinecovp_df = covpAS(n, alp, a, b, t1, t2)
    nndf = gcovpA(n, alp, a, b, t1, t2)
    nndf['mcp'] = ArcSinecovp_df['mcpA'].iloc[0]
    nndf['micp'] = ArcSinecovp_df['micpA'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plot creation
    plot = (ggplot(nndf, aes(x='hp', y='cp'))
            + ggtitle("Coverage Probability for ArcSine method")
            + labs(x="p", y="Coverage Probability")
            + geom_hline(aes(yintercept='t1'), color="red", linetype="dashed")
            + geom_hline(aes(yintercept='t2'), color="blue", linetype="dashed")
            + geom_text(aes(x=0.1, y='t1'), label="Lower tolerance (t1)", color="red", ha='left')
            + geom_text(aes(x=0.1, y='t2'), label="Higher tolerance (t2)", color="blue", ha='left')
            + geom_hline(aes(yintercept='micp', color="'Minimum Coverage'"))
            + geom_hline(aes(yintercept='mcp', color="'Mean Coverage'"))
            + geom_line(aes(color="'Coverage Probability'"))
            + scale_color_manual(name="Heading", values={
                'Coverage Probability': '#FF0000',
                'Minimum Coverage': '#000000',
                'Mean Coverage': '#0000FF'
            })
            + geom_hline(yintercept=1 - alp, linetype="dashed")
            + xlim(0, 1)  # Set x-axis range from 0 to 1
            + ylim(0, 1)  # Set y-axis range from 0 to 1
            )

    plot.show()



def plotcovpLR(n, alp, a, b, t1, t2):
    # Input validation
    if n is None or not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if alp is None or not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a is None or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b is None or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 is None or not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if t2 is None or not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")
    if t1 > t2:
        raise ValueError("t1 has to be less than t2")

    # Generate data
    LRcovp_df = covpLR(n, alp, a, b, t1, t2)
    nndf = gcovpL(n, alp, a, b, t1, t2)
    nndf['mcp'] = LRcovp_df['mcpL'].iloc[0]
    nndf['micp'] = LRcovp_df['micpL'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plot creation
    plot = (ggplot(nndf, aes(x='hp', y='cp'))
            + ggtitle("Coverage Probability for Likelihood Ratio method")
            + labs(x="p", y="Coverage Probability")
            + geom_hline(aes(yintercept='t1'), color="red", linetype="dashed")
            + geom_hline(aes(yintercept='t2'), color="blue", linetype="dashed")
            + geom_text(aes(x=0.1, y='t1'), label="Lower tolerance (t1)", color="red", ha='left')
            + geom_text(aes(x=0.1, y='t2'), label="Higher tolerance (t2)", color="blue", ha='left')
            + geom_hline(aes(yintercept='micp', color="'Minimum Coverage'"))
            + geom_hline(aes(yintercept='mcp', color="'Mean Coverage'"))
            + geom_line(aes(color="'Coverage Probability'"))
            + scale_color_manual(name="Heading", values={
                'Coverage Probability': '#FF0000',
                'Minimum Coverage': '#000000',
                'Mean Coverage': '#0000FF'
            })
            + geom_hline(yintercept=1 - alp, linetype="dashed")
            + xlim(0, 1)  # Set x-axis range from 0 to 1
            + ylim(0, 1)  # Set y-axis range from 0 to 1
            )

    plot.show()

def plotcovpSC(n, alp, a, b, t1, t2):
    # Input validation
    if n is None or not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if alp is None or not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a is None or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b is None or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 is None or not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if t2 is None or not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")
    if t1 > t2:
        raise ValueError("t1 has to be less than t2")

    # Generate data
    Scorecovp_df = covpSC(n, alp, a, b, t1, t2)
    nndf = gcovpS(n, alp, a, b, t1, t2)
    nndf['mcp'] = Scorecovp_df['mcpS'].iloc[0]
    nndf['micp'] = Scorecovp_df['micpS'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp

    # Plot creation
    plot = (ggplot(nndf, aes(x='hp', y='cp'))
            + ggtitle("Coverage Probability for Score method")
            + labs(x="p", y="Coverage Probability")
            + geom_hline(aes(yintercept='t1'), color="red", linetype="dashed")
            + geom_hline(aes(yintercept='t2'), color="blue", linetype="dashed")
            + geom_text(aes(x=0.1, y='t1'), label="Lower tolerance (t1)", color="red", ha='left')
            + geom_text(aes(x=0.1, y='t2'), label="Higher tolerance (t2)", color="blue", ha='left')
            + geom_hline(aes(yintercept='micp', color="'Minimum Coverage'"))
            + geom_hline(aes(yintercept='mcp', color="'Mean Coverage'"))
            + geom_line(aes(color="'Coverage Probability'"))
            + scale_color_manual(name="Heading", values={
                'Coverage Probability': '#FF0000',
                'Minimum Coverage': '#000000',
                'Mean Coverage': '#0000FF'
            })
            + geom_hline(yintercept=1 - alp, linetype="dashed")
            )

    plot.show()


def plotcovpLT(n, alp, a, b, t1, t2):
    if n is None or not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if alp is None or not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a is None or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b is None or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 is None or not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if t2 is None or not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")
    if t1 > t2:
        raise ValueError("t1 has to be less than t2")

    # Generate data
    WaldLcovp_df = covpLT(n, alp, a, b, t1, t2)
    nndf = gcovpLT(n, alp, a, b, t1, t2)
    nndf['mcp'] = WaldLcovp_df['mcpLT'].iloc[0]
    nndf['micp'] = WaldLcovp_df['micpLT'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp
    plot = (ggplot(nndf, aes(x='hp', y='cp'))
            + ggtitle("Coverage Probability for Logit Wald method")
            + labs(x="p", y="Coverage Probability")
            + geom_hline(aes(yintercept='t1'), color="red", linetype="dashed")
            + geom_hline(aes(yintercept='t2'), color="blue", linetype="dashed")
            + geom_text(aes(x=0.1, y='t1'), label="Lower tolerance (t1)", color="red", ha='left')
            + geom_text(aes(x=0.1, y='t2'), label="Higher tolerance (t2)", color="blue", ha='left')
            + geom_hline(aes(yintercept='micp', color="'Minimum Coverage'"))
            + geom_hline(aes(yintercept='mcp', color="'Mean Coverage'"))
            + geom_line(aes(color="'Coverage Probability'"))
            + scale_color_manual(name="Heading", values={
                'Coverage Probability': '#FF0000',
                'Minimum Coverage': '#000000',
                'Mean Coverage': '#0000FF'
            })
            + geom_hline(yintercept=1 - alp, linetype="dashed")
            )

    plot.show()




def plotcovpTW(n, alp, a, b, t1, t2):
    if n is None or not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("'n' has to be greater than 0")
    if alp is None or not (0 <= alp <= 1):
        raise ValueError("'alpha' has to be between 0 and 1")
    if a is None or a < 0:
        raise ValueError("'a' has to be greater than or equal to 0")
    if b is None or b < 0:
        raise ValueError("'b' has to be greater than or equal to 0")
    if t1 is None or not (0 <= t1 <= 1):
        raise ValueError("'t1' has to be between 0 and 1")
    if t2 is None or not (0 <= t2 <= 1):
        raise ValueError("'t2' has to be between 0 and 1")
    if t1 > t2:
        raise ValueError("t1 has to be less than t2")
    AdWaldcovp_df = covpTW(n, alp, a, b, t1, t2)
    nndf = gcovpTW(n, alp, a, b, t1, t2)
    nndf['mcp'] = AdWaldcovp_df['mcpTW'].iloc[0]
    nndf['micp'] = AdWaldcovp_df['micpTW'].iloc[0]
    nndf['t1'] = t1
    nndf['t2'] = t2
    nndf['alp'] = alp
    plot = (ggplot(nndf, aes(x='hp', y='cp'))
            + ggtitle("Coverage Probability for Wald-T method")
            + labs(x="p", y="Coverage Probability")
            + geom_hline(aes(yintercept='t1'), color="red", linetype="dashed")
            + geom_hline(aes(yintercept='t2'), color="blue", linetype="dashed")
            + geom_text(aes(x=0.1, y='t1'), label="Lower tolerance (t1)", color="red", ha='left')
            + geom_text(aes(x=0.1, y='t2'), label="Higher tolerance (t2)", color="blue", ha='left')
            + geom_hline(aes(yintercept='micp', color="'Minimum Coverage'"))
            + geom_hline(aes(yintercept='mcp', color="'Mean Coverage'"))
            + geom_line(aes(color="'Coverage Probability'"))
            + scale_color_manual(name="Heading", values={
                'Coverage Probability': '#FF0000',
                'Minimum Coverage': '#000000',
                'Mean Coverage': '#0000FF'
            })
            + geom_hline(yintercept=1 - alp, linetype="dashed")
            )
    plot.show()








#SJTP