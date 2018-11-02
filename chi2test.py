from scipy import stats

def chisquare(obs, exp, std, ddof=1):
    length = len(obs)
    chis = sum([((obs[i] - exp[i])/(std[i]))**2 for i in range(0,length-1)])
    pVal = 1 - stats.chi2.cdf(chis, length-ddof if length-ddof > 1 else 1)
    return (chis, pVal)
