# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 04:15:58 2017

@author: rcortez
"""
#Python for Finance (safariBooks)
#%%
import numpy as np
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from portfolio_gen import download_hist_data, download_symbol, load_symbols
from portfolio_gen import get_data_all, get_data, symbol_to_path
import datetime
import os
#%matplotlib inline


#proj_path ='/home/rcortez/projects/python/projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup' 
cwd = os.getcwd()
print cwd
#os.chdir('../')

proj_path = os.getcwd()
ptf_fnames = dict()
ptf_fnames['buffett'] = 'buffett_port_syms.pickle'        
#%%
def gen_paths(S0, r, sigma, T, M, I):
            ''' Generates Monte Carlo paths for geometric Brownian motion.

            Parameters
            ==========
            S0 : float
                initial stock/index value
            r : float
            sigma : float
                constant volatility
            T : float
                final time horizon
            M : int
                number of time steps/intervals
            I : int
                number of paths to be simulated

            Returns
            =======
            paths : ndarray, shape (M + 1, I)
                simulated paths given the parameters
            '''
            dt = float(T) / M
            paths = np.zeros((M + 1, I), np.float64)
            paths[0] = S0
            for t in range(1, M + 1):
                rand = np.random.standard_normal(I)
                rand = (rand - rand.mean()) / rand.std()
                paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                                 sigma * np.sqrt(dt) * rand)
            return paths
            
def print_statistics(array):
            ''' Prints selected statistics.

            Parameters
            ==========
            array: ndarray
                object to generate statistics on
            '''
            sta = scs.describe(array)
            print "%14s %15s" % ('statistic', 'value')
            print 30 * "-"
            print "%14s %15.5f" % ('size', sta[0])
            print "%14s %15.5f" % ('min', sta[1][0])
            print "%14s %15.5f" % ('max', sta[1][1])
            print "%14s %15.5f" % ('mean', sta[2])
            print "%14s %15.5f" % ('std', np.sqrt(sta[3]))
            print "%14s %15.5f" % ('skew', sta[4])
            print "%14s %15.5f" % ('kurtosis', sta[5])      
            
def normality_tests(arr):
             ''' Tests for normality distribution of given data set.

             Parameters
             ==========
             array: ndarray
                 object to generate statistics on
             '''
             print "Skew of data set  %14.3f" % scs.skew(arr)
             print "Skew test p-value %14.3f" % scs.skewtest(arr)[1]
             print "Kurt of data set  %14.3f" % scs.kurtosis(arr)
             print "Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1]
             print "Norm test p-value %14.3f" % scs.normaltest(arr)[1]

            
            
if __name__ == "__main__":
    S0 = 100.
    r = 0.05
    sigma = 0.2
    T = 1.0
    M = 50
    I = 250000         
    paths = gen_paths(S0, r, sigma, T, M, I)
    plt.plot(paths[:, :10])
    plt.grid(True)
    plt.xlabel('time steps')
    plt.ylabel('index level')    
    log_returns = np.log(paths[1:] / paths[0:-1])
    paths[:, 0].round(4)
    log_returns[:, 0].round(4)    
    print_statistics(log_returns.flatten())
    
    plt.hist(log_returns.flatten(), bins=70, normed=True, label='frequency')
    plt.grid(True)
    plt.xlabel('log-return')
    plt.ylabel('frequency')
    x = np.linspace(plt.axis()[0], plt.axis()[1])
    plt.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M)),
              'r', lw=2.0, label='pdf')
    plt.legend()    
    
    sm.qqplot(log_returns.flatten()[::500], line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')    
    
    normality_tests(log_returns.flatten())    
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.hist(paths[-1], bins=30)
    ax1.grid(True)
    ax1.set_xlabel('index level')
    ax1.set_ylabel('frequency')
    ax1.set_title('regular data')
    ax2.hist(np.log(paths[-1]), bins=30)
    ax2.grid(True)
    ax2.set_xlabel('log index level')
    ax2.set_title('log data')    
    
    print_statistics(paths[-1])    
    print_statistics(np.log(paths[-1]))    
    normality_tests(np.log(paths[-1]))    

    log_data = np.log(paths[-1])
    plt.hist(log_data, bins=70, normed=True, label='observed')
    plt.grid(True)
    plt.xlabel('index levels')
    plt.ylabel('frequency')
    x = np.linspace(plt.axis()[0], plt.axis()[1])
    plt.plot(x, scs.norm.pdf(x, log_data.mean(), log_data.std()),
             'r', lw=2.0, label='pdf')
    plt.legend()    
    
    sm.qqplot(log_data, line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')    
    
#==============================================================================
#     Real-World Data
#==============================================================================
    import pandas as pd
    import pandas.io.data as web
    
    symbols = ['SPY', 'AAPL', 'IBM', 'LEE']
    start_date = datetime.datetime(2013, 4, 5)
    end_date = datetime.datetime(2015, 6, 2) 
    dates = pd.date_range(start_date, end_date)  
#    symbols = download_hist_data('buffett', start_date, end_date )
    
    data = get_data(symbols, dates)
#    for sym in symbols:
#        data[sym] = get_data_all(sym, dates)
#        print 'dataframe for {}'.format(sym)
#         data[sym] = web.DataReader(sym, data_source='yahoo',
#                                     start='1/1/2006')['Adj Close']
    data = data.dropna()    
    data.info()
    data.head()
    (data / data.ix[0] * 100).plot(figsize=(8, 6))
    
    log_returns = np.log(data / data.shift(1))
    log_returns.head()  
    log_returns.hist(bins=50, figsize=(9, 6))       
    
    for sym in symbols:
        print "\nResults for symbol %s" % sym
        print 30 * "-"
        log_data = np.array(log_returns[sym].dropna())
        print_statistics(log_data)    
        
    sm.qqplot(log_returns['SPY'].dropna(), line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')        
    
    sm.qqplot(log_returns['AAPL'].dropna(), line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')    
    
#All this leads us finally to the formal normality tests:

    for sym in symbols:
         print "\nResults for symbol %s" % sym
         print 32 * "-"
         log_data = np.array(log_returns[sym].dropna())
         normality_tests(log_data)    
         
#==============================================================================
#          Portfolio Optimization
#==============================================================================
    import numpy as np
    import pandas as pd
    import pandas.io.data as web
    import matplotlib.pyplot as plt
     
     
    symbols = ['SPY', 'AAPL', 'IBM', 'LEE']    
    noa = len(symbols)     
     
    
#    for sym in symbols:
#         data[sym] = web.DataReader(sym, data_source='yahoo',
#                                        end='2014-09-12')['Adj Close']
#    data.columns = symbols
    data = get_data(symbols, dates)

    (data / data.ix[0] * 100).plot(figsize=(8, 5))     
    rets = np.log(data / data.shift(1))   
    rets.mean() * 252       
    rets.cov() * 252    
    
#==============================================================================
#     The Basic Theory
#==============================================================================
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    
    np.sum(rets.mean() * weights) * 252
    # expected portfolio return    
    
    np.dot(weights.T, np.dot(rets.cov() * 252, weights))
    # expected portfolio variance    
    
    np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    # expected portfolio standard deviation/volatility    
    
    prets = []
    pvols = []
    for p in range (2500):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        prets.append(np.sum(rets.mean() * weights) * 252)
        pvols.append(np.sqrt(np.dot(weights.T,
                             np.dot(rets.cov() * 252, weights))))
    prets = np.array(prets)
    pvols = np.array(pvols)    
    
    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets, c=prets / pvols, marker='o')
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')    
    

#==============================================================================
#   Portfolio Optimization    
#==============================================================================
    import scipy.optimize as sco        
    def statistics(weights):
        ''' Returns portfolio statistics.

        Parameters
        ==========
        weights : array-like
            weights for different securities in portfolio
        Returns
        =======
        pret : float
            expected portfolio return
        pvol : float
            expected portfolio volatility
        pret / pvol : float
            Sharpe ratio for rf=0
        '''
        weights = np.array(weights)
        pret = np.sum(rets.mean() * weights) * 252
        pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
        return np.array([pret, pvol, pret / pvol])    

#The derivation of the optimal portfolios is a constrained optimization problem for which we use the function minimize from the scipy.optimize sublibrary (cf. Chapter 9):

    import scipy.optimize as sco

#The minimization function minimize is quite general and allows for (in)equality constraints and bounds for the parameters. Let us start with the maximization of the Sharpe ratio. Formally, we minimize the negative value of the Sharpe ratio:

    def min_func_sharpe(weights):
        return -statistics(weights)[2]

#The constraint is that all parameters (weights) add up to 1. This can be formulated as follows using the conventions of the minimize function (cf. the documentation for this function).[42]

    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})

#We also bound the parameter values (weights) to be within 0 and 1. These values are provided to the minimization function as a tuple of tuples in this case:

    bnds = tuple((0, 1) for x in range(noa))

#The only input that is missing for a call of the optimization function is a starting parameter list (initial guesses for the weights). We simply use an equal distribution:

    noa * [1. / noa,]

    
#Calling the function returns not only optimal parameter values, but much more. We store the results in an object we call opts:

#    %%time
    opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',
                                bounds=bnds, constraints=cons)

#Here are the results:

    opts


#Our main interest lies in getting the optimal portfolio composition. To this end, we access the results object by providing the key of interest—i.e., x in our case. The optimization yields a portfolio that only consists of three out of the five assets:

    opts['x'].round(3)



#Using the portfolio weights from the optimization, the following statistics emerge:

    statistics(opts['x']).round(3)

#The expected return is about 23.5%, the expected volatility is about 22.2%, and the resulting optimal Sharpe ratio is 1.06.

#Next, let us minimize the variance of the portfolio. This is the same as minimizing the volatility, but we will define a function to minimize the variance:

    def min_func_variance(weights):
        return statistics(weights)[1] ** 2

#Everything else can remain the same for the call of the minimize function:

    optv = sco.minimize(min_func_variance, noa * [1. / noa,],
                                method='SLSQP', bounds=bnds,
                                constraints=cons)

#This time a fourth asset is added to the portfolio. This portfolio mix leads to the absolute minimum variance portfolio:

    optv['x'].round(3)

#For the expected return, volatility, and Sharpe ratio, we get:

    statistics(optv['x']).round(3)
    
#The expected return is about 23.5%, the expected volatility is about 22.2%, and the resulting optimal Sharpe ratio is 1.06.

#Next, let us minimize the variance of the portfolio. This is the same as minimizing the volatility, but we will define a function to minimize the variance:

    def min_func_variance(weights):
        return statistics(weights)[1] ** 2

#Everything else can remain the same for the call of the minimize function:

    optv = sco.minimize(min_func_variance, noa * [1. / noa,],
                                method='SLSQP', bounds=bnds,
                                constraints=cons)

#This time a fourth asset is added to the portfolio. This portfolio mix leads to the absolute minimum variance portfolio:

    optv['x'].round(3)

#For the expected return, volatility, and Sharpe ratio, we get:

    statistics(optv['x']).round(3)    


#==============================================================================
#   Efficient Frontier
#==============================================================================
  
#The derivation of all optimal portfolios—i.e., all portfolios with minimum volatility for a given target return level (or all portfolios with maximum return for a given risk level)—is similar to the previous optimizations. The only difference is that we have to iterate over multiple starting conditions. The approach we take is that we fix a target return level and derive for each such level those portfolio weights that lead to the minimum volatility value. For the optimization, this leads to two conditions: one for the target return level tret and one for the sum of the portfolio weights as before. The boundary values for each parameter stay the same:

    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
                 {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    bnds = tuple((0, 1) for x in weights)

#For clarity, we define a dedicated function min_func for use in the minimization procedure. It merely returns the volatility value from the statistics function:

    def min_func_port(weights):
        return statistics(weights)[1]

#When iterating over different target return levels (trets), one condition for the minimization changes. That is why the conditions dictionary is updated during every loop:

#%%time
    trets = np.linspace(0.0, 0.25, 50)
    tvols = []
    for tret in trets:
        cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
                     {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                                bounds=bnds, constraints=cons)
        tvols.append(res['fun'])
    tvols = np.array(tvols)

#Out[64]: CPU times: user 4.35 s, sys: 4 ms, total: 4.36 s
#         Wall time: 4.36 s

#Figure 11-13 shows the optimization results. Crosses indicate the optimal 
#portfolios given a certain target return; the dots are, as before, the random 
#portfolios. In addition, the figure shows two larger stars: one for the minimum 
#volatility/variance portfolio (the leftmost portfolio) and one for the portfolio 
#with the maximum Sharpe ratio:

    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets,
                     c=prets / pvols, marker='o')
                     # random portfolio composition
    plt.scatter(tvols, trets,
                     c=trets / tvols, marker='x')
                     # efficient frontier
    plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
                  'r*', markersize=15.0)
                     # portfolio with highest Sharpe ratio
    plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
                  'y*', markersize=15.0)
                     # minimum variance portfolio
    plt.grid(True)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')

#Minimum risk portfolios for given return level (crosses)
#Figure 11-13. Minimum risk portfolios for given return level (crosses)

#The efficient frontier is comprised of all optimal portfolios with a higher return than the absolute minimum variance portfolio. These portfolios dominate all other portfolios in terms of expected returns given a certain risk level.
#%%
#==============================================================================
#   Capital Market Line
#==============================================================================

#In addition to risky securities like stocks or commodities (such as gold), 
#there is in general one universal, riskless investment opportunity available: 
#cash or cash accounts. In an idealized world, money held in a cash account 
#with a large bank can be considered riskless (e.g., through public deposit 
#insurance schemes). The downside is that such a riskless investment generally 
#yields only a small return, sometimes close to zero.
#However, taking into account such a riskless asset enhances the efficient 
#investment opportunity set for investors considerably. The basic idea is that 
#investors first determine an efficient portfolio of risky assets and then add 
#the riskless asset to the mix. By adjusting the proportion of the investor’s 
#wealth to be invested in the riskless asset it is possible to achieve any 
#risk-return profile that lies on the straight line (in the risk-return space) 
#between the riskless asset and the efficient portfolio.
#Which efficient portfolio (out of the many options) is to be taken to invest 
#in optimal fashion? It is the one portfolio where the tangent line of the efficient 
#frontier goes exactly through the risk-return point of the riskless portfolio. 
#For example, consider a riskless interest rate of rf = 0.01. We look for that 
#portfolio on the efficient frontier for which the tangent goes through the 
#point (𝜎f,rf) = (0,0.01) in risk-return space.
#For the calculations to follow, we need a functional approximation and the first 
#derivative for the efficient frontier. We use cubic splines interpolation to 
#this end (cf. Chapter 9):

    import scipy.interpolate as sci

#For the spline interpolation, we only use the portfolios from the efficient frontier. The following code selects exactly these portfolios from our previously used sets tvols and trets:

    ind = np.argmin(tvols)
    evols = tvols[ind:]
    erets = trets[ind:]

#The new ndarray objects evols and erets are used for the interpolation:

    tck = sci.splrep(evols, erets)

#Via this numerical route we end up being able to define a continuously differentiable function f(x) for the efficient frontier and the respective first derivative function df(x):

    def f(x):
        ''' Efficient frontier function (splines approximation). '''
        return sci.splev(x, tck, der=0)
    def df(x):
        ''' First derivative of efficient frontier function. '''
        return sci.splev(x, tck, der=1)

#What we are looking for is a function t(x) = a + b · x describing the line that 
#passes through the riskless asset in risk-return space and that is tangent to 
#the efficient frontier. Equation 11-4 describes all three conditions that the function t(x) has to satisfy.
#Equation 11-4. Mathematical conditions for capital market line
#Mathematical conditions for capital market line

#Since we do not have a closed formula for the efficient frontier or the first 
#derivative of it, we have to solve the system of equations in Equation 11-4 numerically. 
#To this end, we define a Python function that returns the values of all three 
#equations given the parameter set p = (a,b,x):

    def equations(p, rf=0.01):
        eq1 = rf - p[0]
        eq2 = rf + p[1] * p[2] - f(p[2])
        eq3 = p[1] - df(p[2])
        return eq1, eq2, eq3

#The function fsolve from scipy.optimize is capable of solving such a system of equations. We provide an initial parameterization in addition to the function equations. Note that success or failure of the optimization might depend on the initial parameterization, which therefore has to be chosen carefully—generally by a combination of educated guesses with trial and error:

    opt = sco.fsolve(equations, [0.01, 0.5, 0.15])

#The numerical optimization yields the following values. As desired, we have a = rf = 0.01:

#The three equations are also, as desired, zero:

    np.round(equations(opt), 6)

#Out[73]: array([ 0., -0., -0.])

#Figure 11-14 presents the results graphically: the star represents the optimal 
#portfolio from the efficient frontier where the tangent line passes through the 
#riskless asset point (0,rf = 0.01). The optimal portfolio has an expected 
#volatility of 20.5% and an expected return of 17.6%. The plot is generated with the following code:

    plt.figure(figsize=(8, 4))
    plt.scatter(pvols, prets,
                c=(prets - 0.01) / pvols, marker='o')
                # random portfolio composition
    plt.plot(evols, erets, 'g', lw=4.0)
                # efficient frontier
    cx = np.linspace(0.0, 0.3)
    plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
                # capital market line
    plt.plot(opt[2], f(opt[2]), 'b*', markersize=15.0)
    plt.grid(True)
    plt.axhline(0, color='k', ls='--', lw=2.0)
    plt.axvline(0, color='k', ls='--', lw=2.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')

#Capital market line and tangency portfolio (star) for risk-free rate of 1%
#Figure 11-14. Capital market line and tangency portfolio (star) for risk-free rate of 1%
#
#The portfolio weights of the optimal (tangent) portfolio are as follows. Only three of the five assets are in the mix:

    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
                 {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                                bounds=bnds, constraints=cons)

    res['x'].round(3)


#%%
#==============================================================================
# Principal Component Analysis
#==============================================================================

#Principal component analysis (PCA) has become a popular tool in finance. 
#Wikipedia defines the technique as follows:
#    Principal component analysis (PCA) is a statistical procedure that uses 
#orthogonal transformation to convert a set of observations of possibly correlated 
#variables into a set of values of linearly uncorrelated variables called principal 
#components. The number of principal components is less than or equal to the number 
#of original variables. This transformation is defined in such a way that the first
# principal component has the largest possible variance (that is, accounts for as 
#much of the variability in the data as possible), and each succeeding component 
#in turn has the highest variance possible under the constraint that it is 
#orthogonal to (i.e., uncorrelated with) the preceding components.
#Consider, for example, a stock index like the German DAX index, composed of 30 
#different stocks. The stock price movements of all stocks taken together determine
# the movement in the index (via some well-documented formula). In addition, the 
#stock price movements of the single stocks are generally correlated, for example, 
#due to general economic conditions or certain developments in certain sectors.
#For statistical applications, it is generally quite hard to use 30 correlated 
#factors to explain the movements of a stock index. This is where PCA comes into 
#play. It derives single, uncorrelated components that are “well suited” to 
#explain the movements in the stock index. One can think of these components as 
#linear combinations (i.e., portfolios) of selected stocks from the index. 
#Instead of working with 30 correlated index constituents, one can then work with 
#maybe 5, 3, or even only 1 principal component.
#The example of this section illustrates the use of PCA in such a context. 
#We retrieve data for both the German DAX index and all stocks that make up the index. 
#We then use PCA to derive principal components, which we use to construct what we call a pca_index.
#First, some imports. In particular, we use the KernelPCA function of the 
#scikit-learn machine learning library (cf. the documentation for KernelPCA):

    import numpy as np
    import pandas as pd
    import pandas.io.data as web
    from sklearn.decomposition import KernelPCA

#The DAX Index and Its 30 Stocks

#The following list object contains the 30 symbols for the stocks contained in the German DAX index, as well as the symbol for the index itself:

#    symbols = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE',
#                   'BMW.DE', 'CBK.DE', 'CON.DE', 'DAI.DE', 'DB1.DE',
#                   'DBK.DE', 'DPW.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
#                   'FRE.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'LHA.DE',
#                   'LIN.DE', 'LXS.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE',
#                   'SAP.DE', 'SDF.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE',
#                   'SPY']

#We work only with the closing values of each data set that we retrieve (for details on how to retrieve stock data with pandas, see Chapter 6):


#    data = pd.DataFrame()
#    for sym in symbols:
#        data[sym] = web.DataReader(sym, data_source='yahoo')['Close']
#    data = data.dropna()

#==============================================================================
    stocks = load_symbols('buffett_port_syms.pickle' )    
    symbols  = download_hist_data(stocks, start_date, end_date )
#    symbols = download_hist_data('buffett', start_date, end_date )
    start_date = datetime.datetime(2014, 4, 5)
    end_date = datetime.datetime(2015, 6, 2) 
    dates = pd.date_range(start_date, end_date)
    data = get_data(symbols, dates)
    
    data = data.dropna(axis = 1) #eliminates el entire column when it finds a nan
#==============================================================================

#CPU times: user 408 ms, sys: 68 ms, total: 476 ms
#        Wall time: 5.61 s

#Let us separate the index data since we need it regularly:

#    dax = pd.DataFrame(data.pop('^GDAXI'))
    dax = pd.DataFrame(data.pop('SPY'))

#The DataFrame object data now has log return data for the 30 DAX stocks:

    data[data.columns[:6]].head()

#%%
#==============================================================================
# Applying PCA
#==============================================================================

#Usually, PCA works with normalized data sets. Therefore, the following convenience function proves helpful:

    scale_function = lambda x: (x - x.mean()) / x.std()

#For the beginning, consider a PCA with multiple components (i.e., we do not restrict the number of components):[43]

    pca = KernelPCA().fit(data.apply(scale_function))

#The importance or explanatory power of each component is given by its Eigenvalue. These are found in an attribute of the KernelPCA object. The analysis gives too many components:

    len(pca.lambdas_)


#Therefore, let us only have a look at the first 10 components. The tenth component already has almost negligible influence:

    pca.lambdas_[:10].round()

#We are mainly interested in the relative importance of each component, so we will normalize these values. Again, we use a convenience function for this:

    get_we = lambda x: x / x.sum()
    get_we(pca.lambdas_)[:10]

#With this information, the picture becomes much clearer. The first component already explains about 60% of the variability in the 30 time series. The first five components explain about 95% of the variability:
    get_we(pca.lambdas_)[:5].sum()


#Constructing a PCA Index
#Next, we use PCA to construct a PCA (or factor) index over time and compare it with the original index. First, we have a PCA index with a single component only:

    pca = KernelPCA(n_components=1).fit(data.apply(scale_function))
    dax['PCA_1'] = pca.transform(data)

#Figure 11-15 shows the results for normalized data—already not too bad, given the rather simple application of the approach:

    import matplotlib.pyplot as plt
#    %matplotlib inline
    dax.apply(scale_function).plot(figsize=(8, 4))

#German DAX index and PCA index with one component
#Figure 11-15. German DAX index and PCA index with one component

#Let us see if we can improve the results by adding more components. To this end, we need to calculate a weighted average from the single resulting components:

    pca = KernelPCA(n_components=5).fit(data.apply(scale_function))
    pca_components = pca.transform(data)
    weights = get_we(pca.lambdas_)
    dax['PCA_5'] = np.dot(pca_components, weights)

#The results as presented in Figure 11-16 are still “good,” but not that much better than before—at least upon visual inspection:

    import matplotlib.pyplot as plt
#    %matplotlib inline
    dax.apply(scale_function).plot(figsize=(8, 4))

#German DAX index and PCA indices with one and five components
#Figure 11-16. German DAX index and PCA indices with one and five components

#In view of the results so far, we want to inspect the relationship between the DAX index and the PCA index in a different way—via a scatter plot, adding date information to the mix. First, we convert the DatetimeIndex of the DataFrame object to a matplotlib-compatible format:

    import matplotlib as mpl
#    mpl_dates = mpl.dates.date2num(data.index)
    mpl_dates = mpl.dates.date2num(data.index.to_pydatetime())    
    mpl_dates

#This new date list can be used for a scatter plot, highlighting through different colors which date each data point is from. Figure 11-17 shows the data in this fashion:

    plt.figure(figsize=(8, 4))
    plt.scatter(dax['PCA_5'], dax['SPY'], c=mpl_dates)
    lin_reg = np.polyval(np.polyfit(dax['PCA_5'],
                                         dax['SPY'], 1),
                                         dax['PCA_5'])
    plt.plot(dax['PCA_5'], lin_reg, 'r', lw=3)
    plt.grid(True)
    plt.xlabel('PCA_5')
    plt.ylabel('SPY')
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                         format=mpl.dates.DateFormatter('%d %b %y'))

#DAX return values against PCA return values with linear regression
#Figure 11-17. DAX return values against PCA return values with linear regression
#
#Figure 11-17 reveals that there is obviously some kind of structural break sometime in the middle of 2011. If the PCA index were to perfectly replicate the DAX index, we would expect all the points to lie on a straight line and to see the regression line going through these points. Perfection is hard to achieve, but we can maybe do better.
#
#To this end, let us divide the total time frame into two subintervals. We can then implement an early and a late regression:

    cut_date = '2015/03/01'
    early_pca = dax[dax.index < cut_date]['PCA_5']
    early_reg = np.polyval(np.polyfit(early_pca,
                         dax['SPY'][dax.index < cut_date], 1),
                         early_pca)

    late_pca = dax[dax.index >= cut_date]['PCA_5']
    late_reg = np.polyval(np.polyfit(late_pca,
                         dax['SPY'][dax.index >= cut_date], 1),
                         late_pca)

#Figure 11-18 shows the new regression lines, which indeed display the high explanatory power both before our cutoff date and thereafter. This heuristic approach will be made a bit more formal in the next section on Bayesian statistics:

    plt.figure(figsize=(8, 4))
    plt.scatter(dax['PCA_5'], dax['SPY'], c=mpl_dates)
    plt.plot(early_pca, early_reg, 'r', lw=3)
    plt.plot(late_pca, late_reg, 'r', lw=3)
    plt.grid(True)
    plt.xlabel('PCA_5')
    plt.ylabel('SPY')
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                         format=mpl.dates.DateFormatter('%d %b %y'))

#DAX index values against PCA index values with early and late regression (regime switch)
#Figure 11-18. DAX index values against PCA index values with early and late regression (regime switch)
#%%
#==============================================================================
# Bayesian Regression
#==============================================================================

#Bayesian statistics nowadays is a cornerstone in empirical finance. This chapter cannot lay the foundations for all concepts of the field. You should therefore consult, if needed, a textbook like that by Geweke (2005) for a general introduction or Rachev (2008) for one that is financially motivated.
#Bayes’s Formula
#
#The most common interpretation of Bayes’ formula in finance is the diachronic interpretation. This mainly states that over time we learn new information about certain variables or parameters of interest, like the mean return of a time series. Equation 11-5 states the theorem formally. Here, H stands for an event, the hypothesis, and D represents the data an experiment or the real world might present.[44] On the basis of these fundamental definitions, we have:
#
#    p(H) is called the prior probability.
#    p(D) is the probability for the data under any hypothesis, called the normalizing constant.
#    p(D|H) is the likelihood (i.e., the probability) of the data under hypothesis H.
#    p(H|D) is the posterior probability; i.e., after we have seen the data. 

#Equation 11-5. Bayes’s formula
#Bayes’s formula
#
#Consider a simple example. We have two boxes, B 1 and B 2. Box B 1 contains 20 black balls and 70 red balls, while box B 2 contains 40 black balls and 50 red balls. We randomly draw a ball from one of the two boxes. Assume the ball is black. What are the probabilities for the hypotheses “H 1: Ball is from box B 1” and “H 2: Ball is from box B 2,” respectively?
#
#Before we randomly draw the ball, both hypotheses are equally likely. After it is clear that the ball is black, we have to update the probability for both hypotheses according to Bayes’ formula. Consider hypothesis H 1:
#
#    Prior: p(H 1) = 0.5
#    Normalizing constant: p(D) = 0.5 · 0.2 + 0.5 · 0.4 = 0.3
#    Likelihood: p(D|H 1) = 0.2 
#
#This gives for the updated probability of H 1 .
#
#This result also makes sense intuitively. The probability for drawing a black ball from box B 2 is twice as high as for the same event happening with box B 1. Therefore, having drawn a black ball, the hypothesis H 2 has with an updated probability two times as high as the updated probability for hypothesis H 1.
#PyMC3
#
#With PyMC3 the Python ecosystem provides a powerful and performant library to technically implement Bayesian statistics. PyMC3 is (at the time of this writing) not part of the Anaconda distribution recommended in Chapter 2. On a Linux or a Mac OS X operating system, the installation comprises mainly the following steps.
#
#First, you need to install the Theano compiler package needed for PyMC3 (cf. http://bit.ly/install_theano). In the shell, execute the following commands:

#$ git clone git://github.com/Theano/Theano.git
#$ sudo python Theano/python.py install
#
#On a Mac OS X system you might need to add the following line to your .bash_profile file (to be found in your home/user directory):
#
#export DYLD_FALLBACK_LIBRARY_PATH= \
#$DYLD_FALLBACK_LIBRARY_PATH:/Library/anaconda/lib:
#
#Once Theano is installed, the installation of PyMC3 is straightforward:
#
#$ git clone https://github.com/pymc-devs/pymc.git
#$ cd pymc
#$ sudo python setup.py install
#
#If successful, you should be able to import the library named pymc as usual:
#
#In [22]: import warnings
#         warnings.simplefilter('ignore')
#         import pymc as pm
#         import numpy as np
#         np.random.seed(1000)
#         import matplotlib.pyplot as plt
#         %matplotlib inline
#
#PyMC3
#
#PyMC3 is already a powerful library at the time of this writing. However, it is still in its early stages, so you should expect further enhancements, changes to the API, etc. Make sure to stay up to date by regularly checking the website when using PyMC3.
#Introductory Example
#
#Consider now an example where we have noisy data around a straight line:[45]
#
#In [23]: x = np.linspace(0, 10, 500)
#         y = 4 + 2 * x + np.random.standard_normal(len(x)) * 2
#
#As a benchmark, consider first an ordinary least-squares regression given the noisy data, using NumPy’s polyfit function (cf. Chapter 9). The regression is implemented as follows:
#
#In [24]: reg = np.polyfit(x, y, 1)
#           # linear regression
#
#Figure 11-19 shows the data and the regression line graphically:
#
#In [25]: plt.figure(figsize=(8, 4))
#         plt.scatter(x, y, c=y, marker='v')
#         plt.plot(x, reg[1] + reg[0] * x, lw=2.0)
#         plt.colorbar()
#         plt.grid(True)
#         plt.xlabel('x')
#         plt.ylabel('y')
#
#Sample data points and regression line
#Figure 11-19. Sample data points and regression line
#
#The result of the “standard” regression approach is fixed values for the parameters of the regression line:
#
#In [26]: reg
#
#Out[26]: array([ 2.03384161,  3.77649234])
#
#Note that the highest-order monomial factor (in this case, the slope of the regression line) is at index level 0 and that the intercept is at index level 1. The original parameters 2 and 4 are not perfectly recovered, but this of course is due to the noise included in the data.
#
#Next, the Bayesian regression. Here, we assume that the parameters are distributed in a certain way. For example, consider the equation describing the regression line ŷ(x) = 𝛼 + 𝛽 · x. We now assume the following priors:
#
#    𝛼 is normally distributed with mean 0 and a standard deviation of 20.
#    𝛽 is normally distributed with mean 0 and a standard deviation of 20. 
#
#For the likelihood, we assume a normal distribution with mean of ŷ(x) and a uniformly distributed standard deviation between 0 and 10.
#
#A major element of Bayesian regression is (Markov Chain) Monte Carlo (MCMC) sampling.[46] In principle, this is the same as drawing balls multiple times from boxes, as in the previous simple example—just in a more systematic, automated way.
#
#For the technical sampling, there are three different functions to call:
#
#    find_MAP finds the starting point for the sampling algorithm by deriving the local maximum a posteriori point.
#    NUTS implements the so-called “efficient No-U-Turn Sampler with dual averaging” (NUTS) algorithm for MCMC sampling given the assumed priors.
#    sample draws a number of samples given the starting value from find_MAP and the optimal step size from the NUTS algorithm. 
#
#All this is to be wrapped into a PyMC3 Model object and executed within a with statement:
#
#In [27]: with pm.Model() as model:
#                 # model specifications in PyMC3
#                 # are wrapped in a with statement
#             # define priors
#             alpha = pm.Normal('alpha', mu=0, sd=20)
#             beta = pm.Normal('beta', mu=0, sd=20)
#             sigma = pm.Uniform('sigma', lower=0, upper=10)
#
#             # define linear regression
#             y_est = alpha + beta * x
#
#             # define likelihood
#             likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)
#
#             # inference
#             start = pm.find_MAP()
#               # find starting value by optimization
#             step = pm.NUTS(state=start)
#               # instantiate MCMC sampling algorithm
#             trace = pm.sample(100, step, start=start, progressbar=False)
#               # draw 100 posterior samples using NUTS sampling
#
#Have a look at the estimates from the first sample:
#
#In [28]: trace[0]
#
#Out[28]: {'alpha': 3.8783781152509031,
#          'beta': 2.0148472296530033,
#          'sigma': 2.0078134493352975}
#
#All three values are rather close to the original values (4, 2, 2). However, the whole procedure yields, of course, many more estimates. They are best illustrated with the help of a trace plot, as in Figure 11-20—i.e., a plot showing the resulting posterior distribution for the different parameters as well as all single estimates per sample. The posterior distribution gives us an intuitive sense about the uncertainty in our estimates:
#
#In [29]: fig = pm.traceplot(trace, lines={'alpha': 4, 'beta': 2, 'sigma': 2})
#         plt.figure(figsize=(8, 8))
#
#Trace plots for alpha, beta, and sigma
#Figure 11-20. Trace plots for alpha, beta, and sigma
#
#Taking only the alpha and beta values from the regression, we can draw all resulting regression lines as shown in Figure 11-21:
#
#In [30]: plt.figure(figsize=(8, 4))
#         plt.scatter(x, y, c=y, marker='v')
#         plt.colorbar()
#         plt.grid(True)
#         plt.xlabel('x')
#         plt.ylabel('y')
#         for i in range(len(trace)):
#             plt.plot(x, trace['alpha'][i] + trace['beta'][i] * x)
#
#Sample data and regression lines from Bayesian regression
#Figure 11-21. Sample data and regression lines from Bayesian regression
#Real Data
#
#Having seen Bayesian regression with PyMC3 in action with dummy data, we now move on to real market data. In this context, we introduce yet another Python library: zipline (cf. https://github.com/quantopian/zipline and https://pypi.python.org/pypi/zipline). zipline is a Pythonic, open source algorithmic trading library that powers the community backtesting platform Quantopian.
#
#It is also to be installed separately, e.g., by using pip:
#
#$ pip install zipline
#
#After installation, import zipline as well pytz and datetime as follows:
#
#In [31]: import warnings
#         warnings.simplefilter('ignore')
#         import zipline
#         import pytz
#         import datetime as dt
#
#Similar to pandas, zipline provides a convenience function to load financial data from different sources. Under the hood, zipline also uses pandas.
#
#The example we use is a “classical” pair trading strategy, namely with gold and stocks of gold mining companies. These are represented by ETFs with the following symbols, respectively:
#
#    GLD
#    GDX 
#
#We can load the data using zipline as follows:
#
#In [32]: data = zipline.data.load_from_yahoo(stocks=['GLD', 'GDX'],
#                  end=dt.datetime(2014, 3, 15, 0, 0, 0, 0, pytz.utc)).dropna()
#         data.info()
#
#Out[32]: GLD
#         GDX
#         <class 'pandas.core.frame.DataFrame'>
#         DatetimeIndex: 1967 entries, 2006-05-22 00:00:00+00:00 to 2014-03-14 00
#         :00:00+00:00
#         Data columns (total 2 columns):
#         GDX    1967 non-null float64
#         GLD    1967 non-null float64
#         dtypes: float64(2)
#
#Figure 11-22 shows the historical data for both ETFs:
#
#In [33]: data.plot(figsize=(8, 4))
#
#Comovements of trading pair
#Figure 11-22. Comovements of trading pair
#
#The absolute performance differs significantly:
#
#In [34]: data.ix[-1] / data.ix[0] - 1
#
#Out[34]: GDX   -0.216002
#         GLD    1.038285
#         dtype: float64
#
#However, both time series seem to be quite strongly positively correlated when inspecting Figure 11-22, which is also reflected in the correlation data:
#
#In [35]: data.corr()
#
#Out[35]:           GDX       GLD
#         GDX  1.000000  0.466962
#         GLD  0.466962  1.000000
#
#As usual, the DatetimeIndex object of the DataFrame object consists of Timestamp objects:
#
#In [36]: data.index
#
#Out[36]: <class 'pandas.tseries.index.DatetimeIndex'>
#         [2006-05-22, ..., 2014-03-14]
#         Length: 1967, Freq: None, Timezone: UTC
#
#To use the date-time information with matplotlib in the way we want to in the following, we have to first convert it to an ordinal date representation:
#
#In [37]: import matplotlib as mpl
#         mpl_dates = mpl.dates.date2num(data.index)
#         mpl_dates
#
#Out[37]: array([ 732453.,  732454.,  732455., ...,  735304.,  735305.,  735306.])
#
#Figure 11-23 shows a scatter plot of the time series data, plotting the GLD values against the GDX values and illustrating the dates of each data pair with different colorings:[47]
#
#In [38]: plt.figure(figsize=(8, 4))
#         plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
#         plt.grid(True)
#         plt.xlabel('GDX')
#         plt.ylabel('GLD')
#         plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
#                      format=mpl.dates.DateFormatter('%d %b %y'))
#
#Scatter plot of prices for GLD and GDX
#Figure 11-23. Scatter plot of prices for GLD and GDX
#
#Let us implement a Bayesian regression on the basis of these two time series. The parameterizations are essentially the same as in the previous example with dummy data; we just replace the dummy data with the real data we now have available:
#
#In [39]: with pm.Model() as model:
#             alpha = pm.Normal('alpha', mu=0, sd=20)
#             beta = pm.Normal('beta', mu=0, sd=20)
#             sigma = pm.Uniform('sigma', lower=0, upper=50)
#
#             y_est = alpha + beta * data['GDX'].values
#
#             likelihood = pm.Normal('GLD', mu=y_est, sd=sigma,
#                                    observed=data['GLD'].values)
#
#             start = pm.find_MAP()
#             step = pm.NUTS(state=start)
#             trace = pm.sample(100, step, start=start, progressbar=False)
#
#Figure 11-24 shows the results from the MCMC sampling procedure given the assumptions about the prior probability distributions for the three parameters:
#
#In [40]: fig = pm.traceplot(trace)
#         plt.figure(figsize=(8, 8))
#
#Trace plots for alpha, beta, and sigma based on GDX and GLD data
#Figure 11-24. Trace plots for alpha, beta, and sigma based on GDX and GLD data
#
#Figure 11-25 adds all the resulting regression lines to the scatter plot from before. All the regression lines are pretty close to each other:
#
#In [41]: plt.figure(figsize=(8, 4))
#         plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
#         plt.grid(True)
#         plt.xlabel('GDX')
#         plt.ylabel('GLD')
#         for i in range(len(trace)):
#             plt.plot(data['GDX'], trace['alpha'][i] + trace['beta'][i] * data
#                      ['GDX'])
#         plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
#                      format=mpl.dates.DateFormatter('%d %b %y'))
#
#Scatter plot with “simple” regression lines
#Figure 11-25. Scatter plot with “simple” regression lines
#
#The figure reveals a major drawback of the regression approach used: the approach does not take into account evolutions over time. That is, the most recent data is treated the same way as the oldest data.
#
#As pointed out at the beginning of this section, the Bayesian approach in finance is generally most useful when seen as diachronic—i.e., in the sense that new data revealed over time allows for better regressions and estimates.
#
#To incorporate this concept in the current example, we assume that the regression parameters are not only random and distributed in some fashion, but that they follow some kind of random walk over time. It is the same generalization used when making the transition in finance theory from random variables to stochastic processes (which are essentially ordered sequences of random variables):
#
#To this end, we define a new PyMC3 model, this time specifying parameter values as random walks with the variance parameter values transformed to log space (for better sampling characteristics).
#
#In [42]: model_randomwalk = pm.Model()
#         with model_randomwalk:
#             # std of random walk best sampled in log space
#             sigma_alpha, log_sigma_alpha = \
#                     model_randomwalk.TransformedVar('sigma_alpha',
#                                     pm.Exponential.dist(1. / .02, testval=.1),
#                                     pm.logtransform)
#             sigma_beta, log_sigma_beta = \
#                     model_randomwalk.TransformedVar('sigma_beta',
#                                     pm.Exponential.dist(1. / .02, testval=.1),
#                                     pm.logtransform)
#
#After having specified the distributions of the random walk parameters, we can proceed with specifying the random walks for alpha and beta. To make the whole procedure more efficient, 50 data points at a time share common coefficients:
#
#In [43]: from pymc.distributions.timeseries import GaussianRandomWalk
#         # to make the model simpler, we will apply the same coefficients
#         # to 50 data points at a time
#         subsample_alpha = 50
#         subsample_beta = 50
#
#         with model_randomwalk:
#             alpha = GaussianRandomWalk('alpha', sigma_alpha**-2,
#                                        shape=len(data) / subsample_alpha)
#             beta = GaussianRandomWalk('beta', sigma_beta**-2,
#                                       shape=len(data) / subsample_beta)
#
#             # make coefficients have the same length as prices
#             alpha_r = np.repeat(alpha, subsample_alpha)
#             beta_r = np.repeat(beta, subsample_beta)
#
#The time series data sets have a length of 1,967 data points:
#
#In [44]: len(data.dropna().GDX.values)  # a bit longer than 1,950
#
#Out[44]: 1967
#
#For the sampling to follow, the number of data points must be divisible by 50. Therefore, only the first 1,950 data points are taken for the regression:
#
#In [45]: with model_randomwalk:
#             # define regression
#             regression = alpha_r + beta_r * data.GDX.values[:1950]
#
#             # assume prices are normally distributed
#             # the mean comes from the regression
#             sd = pm.Uniform('sd', 0, 20)
#             likelihood = pm.Normal('GLD',
#                                    mu=regression,
#                                    sd=sd,
#                                    observed=data.GLD.values[:1950])
#
#All these definitions are a bit more involved than before due to the use of random walks instead of a single random variable. However, the inference steps with the MCMC remain essentially the same. Note, though, that the computational burden increases substantially since we have to estimate per random walk sample 1,950 / 50 = 39 parameter pairs (instead of 1, as before):
#
#In [46]: import scipy.optimize as sco
#         with model_randomwalk:
#             # first optimize random walk
#             start = pm.find_MAP(vars=[alpha, beta], fmin=sco.fmin_l_bfgs_b)
#
#             # sampling
#             step = pm.NUTS(scaling=start)
#             trace_rw = pm.sample(100, step, start=start, progressbar=False)
#
#In total, we have 100 estimates with 39 time intervals:
#
#In [47]: np.shape(trace_rw['alpha'])
#
#Out[47]: (100, 39)
#
#We can illustrate the evolution of the regression factors alpha and beta over time by plotting a subset of the estimates and the average over all samples, as in Figure 11-26:
#
#In [48]: part_dates = np.linspace(min(mpl_dates), max(mpl_dates), 39)
#
#In [49]: fig, ax1 = plt.subplots(figsize=(10, 5))
#         plt.plot(part_dates, np.mean(trace_rw['alpha'], axis=0),
#                  'b', lw=2.5, label='alpha')
#         for i in range(45, 55):
#             plt.plot(part_dates, trace_rw['alpha'][i], 'b-.', lw=0.75)
#         plt.xlabel('date')
#         plt.ylabel('alpha')
#         plt.axis('tight')
#         plt.grid(True)
#         plt.legend(loc=2)
#         ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b %y') )
#         ax2 = ax1.twinx()
#         plt.plot(part_dates, np.mean(trace_rw['beta'], axis=0),
#                  'r', lw=2.5, label='beta')
#         for i in range(45, 55):
#             plt.plot(part_dates, trace_rw['beta'][i], 'r-.', lw=0.75)
#         plt.ylabel('beta')
#         plt.legend(loc=4)
#         fig.autofmt_xdate()
#
#Evolution of (mean) alpha and (mean) beta over time (updated estimates over time)
#Figure 11-26. Evolution of (mean) alpha and (mean) beta over time (updated estimates over time)
#Absolute Price Data Versus Relative Return Data
#
#Both when presenting the PCA analysis implementation and for this example about Bayesian statistics, we’ve worked with absolute price levels instead of relative (log) return data. This is for illustration purposes only, because the respective graphical results are easier to understand and interpret (they are visually “more appealing”). However, for real-world financial applications you would instead rely on relative return data.
#
#Using the mean alpha and beta values, we can illustrate how the regression is updated over time. Figure 11-27 again shows the data points as a scatter plot. In addition, the 39 regression lines resulting from the mean alpha and beta values are displayed. It is obvious that updating over time increases the regression fit (for the current/most recent data) tremendously—in other words, every time period needs its own regression:
#
#In [50]: plt.figure(figsize=(10, 5))
#         plt.scatter(data['GDX'], data['GLD'], c=mpl_dates, marker='o')
#         plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
#                      format=mpl.dates.DateFormatter('%d %b %y'))
#         plt.grid(True)
#         plt.xlabel('GDX')
#         plt.ylabel('GLD')
#         x = np.linspace(min(data['GDX']), max(data['GDX']))
#         for i in range(39):
#             alpha_rw = np.mean(trace_rw['alpha'].T[i])
#             beta_rw = np.mean(trace_rw['beta'].T[i])
#             plt.plot(x, alpha_rw + beta_rw * x, color=plt.cm.jet(256 * i / 39))
#
#Scatter plot with time-dependent regression lines (updated estimates)
#Figure 11-27. Scatter plot with time-dependent regression lines (updated estimates)
#
#This concludes the section on Bayesian regression, which shows that Python offers with PyMC3 a powerful library to implement different approaches from Bayesian statistics. Bayesian regression in particular is a tool that has become quite popular and important recently in quantitative finance.
#Conclusions
#
#Statistics is not only an important discipline in its own right, but also provides indispensible tools for many other disciplines, like finance and the social sciences. It is impossible to give a broad overview of statistics in a single chapter. This chapter therefore concentrates on four important topics, illustrating the use of Python and several statistics libraries on the basis of realistic examples:
#
#Normality tests
#    The normality assumption with regard to financial market returns is an important one for many financial theories and applications; it is therefore important to be able to test whether certain time series data conforms to this assumption. As we have seen—via graphical and statistical means—real-world return data generally is not normally distributed. 
#Modern portfolio theory
#    MPT, with its focus on the mean and variance/volatility of returns, can be considered one of the major conceptual and intellectual successes of statistics in finance; the important concept of investment diversification is beautifully illustrated in this context. 
#Principal component analysis
#    PCA provides a pretty helpful method to reduce complexity for factor/component analysis tasks; we have shown that five principal components—constructed from the 30 stocks contained in the DAX index—suffice to explain more than 95% of the index’s variability. 
#Bayesian regression
#    Bayesian statistics in general (and Bayesian regression in particular) has become a popular tool in finance, since this approach overcomes shortcomings of other approaches, as introduced in Chapter 9; even if the mathematics and the formalism are more involved, the fundamental ideas—like the updating of probability/distribution beliefs over time—are easily grasped intuitively. 
#
#Further Reading
#
#The following online resources are helpful:
#
#    Information about the SciPy statistical functions is found here: http://docs.scipy.org/doc/scipy/reference/stats.html.
#    Also consult the documentation of the statsmodels library: http://statsmodels.sourceforge.net/stable/.
#    For the optimization functions used in this chapter, refer to http://docs.scipy.org/doc/scipy/reference/optimize.html.
#    There is a short tutorial available for PyMC3; at the time of this writing the library is still in early release mode and not yet fully documented. 
#
#Useful references in book form are:
#
#    Copeland, Thomas, Fred Weston, and Kuldeep Shastri (2005): Financial Theory and Corporate Policy, 4th ed. Pearson, Boston, MA.
#    Downey, Allen (2013): Think Bayes . O’Reilly, Sebastopol, CA.
#    Geweke, John (2005): Contemporary Bayesian Econometrics and Statistics. John Wiley & Sons, Hoboken, NJ.
#    Rachev, Svetlozar et al. (2008): Bayesian Methods in Finance. John Wiley & Sons, Hoboken, NJ. 
#
#
#[41] Cf. Markowitz, Harry (1952): “Portfolio Selection.” Journal of Finance, Vol. 7, 77-91.
#
#[42] An alternative to np.sum(x) - 1 would be to write np.sum(x) == 1 taking into account that with Python the Boolean True value equals 1 and the False value equals 0.
#
#[43] Note that we work here—and in the section to follow on Bayesian statistics—with absolute stock prices and not with return data, which would be more statistically sound. The reason for this is that it simplifies intuition and makes graphical plots easier to interpret. In real-world applications, you would use return data.
#
#[44] For a Python-based introduction into these and other fundamental concepts of Bayesian statistics, refer to Downey (2013).
#
#[45] This example and the one in the following subsection are from a presentation by Thomas Wiecki, one of the lead developers of PyMC3; he allowed me to use them for this chapter, for which I am most grateful.
#
#[46] Cf. http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo. For example, the Monte Carlo algorithms used throughout the book and analyzed in detail in Chapter 10 all generate so-called Markov chains, since the immediate next step/value only depends on the current state of the process and not on any other historic state or value.
#
#[47] Note also here that we are working with absolute price levels and not return data, which would be statistically more sound. For a real-world (trading) application, you would rather choose the return data to implement such an analysis.
#                                