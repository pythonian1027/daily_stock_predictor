import os
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

cwd = os.getcwd()
print cwd
os.chdir('../')
cwd = os.getcwd()
#path = '/home/rcortez/projects/python/projects/umlNanoDegee/machine-learning/projects/machineLearningTradingBackup'
#path = os.path.abspath(os.path.join(cwd, os.pardir))
#print path


def symbol_to_path(symbol, base_dir=cwd + '/data/'):
    """Return CSV file path given ticker symbol."""    
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, na_values=['nan'])        
        df = df.dropna()

    return df
    
def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    plt.show()    
    
def get_stats(s, n=252):
    s = s.dropna()
    wins = len(s[s>0])
    losses = len(s[s<0])
    evens = len(s[s==0])
    mean_w = round(s[s>0].mean(), 3)
    mean_l = round(s[s<0].mean(), 3)
    win_r = round(wins/losses, 3)
    mean_trd = round(s.mean(), 3)
    sd = round(np.std(s), 3)
    max_l = round(s.min(), 3)
    max_w = round(s.max(), 3)
    sharpe_r = round((s.mean()/np.std(s))*np.sqrt(n), 4)
    cnt = len(s)
    print 'Trades:', cnt, '\nWins:', wins,'\nLosses:', losses,
    print '\nBreakeven:', evens,'\nWin/Loss Ratio', win_r,
    print '\nMean Win:', mean_w,'\nMean Loss:', mean_l,
    print '\nMean', mean_trd,'\nStd Dev:', sd,
    print '\nMax Loss:', max_l,'\nMax Win:', max_w,
    print '\nSharpe Ratio:', sharpe_r

if __name__ == "__main__":
    dates = pd.date_range('2000-01-01', '2016-03-01')
    spy = get_data(['SPY'],dates )
    plot_data(spy['Close'])
    
    spy['Daily Change'] = pd.Series(spy['Close'] - spy['Open'])
    print spy['Daily Change']
    
    #sum those changes over the period
    print spy['Daily Change'].sum()
    
    print np.std(spy['Daily Change'])
    
    #daily returns 
    daily_rtn = ((spy['Close'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
#    daily_rtn.hist(bins=50, color='lightblue', figsize=(12,8))
    
    # intra day returns
    id_rtn = ((spy['Close'] - spy['Open'])/spy['Open'])*100
#    id_rtn.hist(bins=50, color='lightblue', figsize=(12,8))
    
    # overnight returns
    on_rtn = ((spy['Open'] - spy['Close'].shift(1))/spy['Close'].shift(1))*100
#    on_rtn.hist(bins=50, color='lightblue', figsize=(12,8))

    get_stats(daily_rtn)
    get_stats(id_rtn)
    get_stats(on_rtn)    
    
#   creating random signals    
#    def get_signal(x):
#        val = np.random.rand()
#        if val > .5:
#            return 1
#        else:
#            return 0
#
#    for i in range(1000):
#        spy['Signal_' + str(i)] = spy.apply(get_signal, axis=1) 
    
# Support Vector Regressor
    sp = spy
    print sp.shape
    for i in range(1, 21, 1):
        sp.loc[:,'Close Minus ' + str(i)] = sp['Close'].shift(i)
        sp20 = sp[[x for x in sp.columns if 'Close Minus' in x or x == 'Close']].iloc[20:,]
    
    sp20 = sp20.iloc[:,::-1]
    from sklearn.svm import SVR
    clf = SVR(kernel='linear')            

    X_train = sp20[:-2000]
    print 'sp20', sp20.shape
    print len(X_train)
    y_train = sp20['Close'].shift(-1)[:-2000]

    X_test = sp20[-2000:-1000]
    y_test = sp20['Close'].shift(-1)[-2000:-1000]

    model = clf.fit(X_train, y_train)       
    preds = model.predict(X_test)
    
    t = np.arange(0, len(preds))
    plt.plot(t, preds, 'r', t, y_test, 'b')
    plt.show()