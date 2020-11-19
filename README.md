# bu-projects
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_acf

def ad_hoc_position(s, upper_bound, lower_bound):
    ''' buy when reach lower bound, sell when reach upper bound
    '''
    position = [0]*len(s)
    
    for i in range(len(s)-1):
        if s[i] <= lower_bound[i]:
            position[i+1] = 1
        elif s[i] >= upper_bound[i]:
            position[i+1] = 0
        else:
            position[i+1] = position[i]
    position[-1] = 0 # close the position in the end
    '''
    s.plot()
    plt.plot(upper_bound)
    plt.plot(lower_bound)
    plt.xlabel('time')
    plt.ylabel('price')
    #plt.title('Ad hoc bands')
    plt.show()
    '''
    return pd.Series(position)


def ad_hoc_capital_process(initial_capital, s, position):
    
    wealth = [1]*len(s)
    r = np.log(s/s.shift(1))
    
    for i in range(1, len(s)):
        wealth[i] = wealth[i-1]*np.exp(r[i]*position[i])
        returns = r[i]*position[i]
    
    re = returns.cumsum()
    mu = np.mean(returns)
    variance = np.var(returns)
    sr = mu/np.sqrt(variance)
    
    wealth = pd.Series(wealth)
    capital = initial_capital*wealth
    
    
    #capital.plot()
    #plt.title('capital process')
    return capital, re[len(re)-1], mu, variance, sr

def buy_hold(initial_capital, s):
    
    r = np.log(s/s.shift(1))
    returns = r.cumsum()
    returns.plot(title = 'Cumulative Returns')
    capital_process=np.exp(returns)*initial_capital
    capital_process[0] = initial_capital
    mu = np.mean(r)
    variance = np.var(r)
    sr = mu/np.sqrt(variance)
    capital = initial_capital*np.exp(returns[len(s)-1])
    
    return capital_process, returns[len(s)-1], mu, variance, sr


def long_short(initial_capital, s, upper_bound, lower_bound):
    
    x = initial_capital/s[0] # short x units of stock
    m = initial_capital-s[len(s)-1]*x
    position = ad_hoc_position(s, upper_bound, lower_bound)
    #position=position.shift(1)
    #position[0]=0
    c = ad_hoc_capital_process(initial_capital, s, position)[0]
    c[len(c)-1]+=m
    
    return c

    

if __name__ == '__main__':
    
    # read the data(dataframe)
    #df = pd.read_csv('/Users/liusihang/Downloads/^GSPC (1).csv')
    df = pd.read_csv('/Users/liusihang/Desktop/821project/coding/^GSPC.csv')
    data = df[['Adj Close']]



    #########################################################################
    
    #Calculate the log return of SP500 index using the “adj close” price.
    #Draw the histogram of the log return, fit to a normal distribution, compare the tail of the fitted
    #distribution with the empirical distribution.
    
    s = data['Adj Close']
    r = np.log(s/s.shift(1))
    data['return'] = r
    plt.hist(r, bins=15)
    mu, std = norm.fit(r[1:])
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    plt.plot(x, norm.pdf(x, mu, std))
    plt.xlabel('log return')
    plt.ylabel('frequency')
    #plt.title('Histogram of the log return&Fitted normal distribution')
    plt.show()
    
    # Split the data into a portfolio formation period and a portfolio testing period
    train_s = s[:-15]
    #train_s = s[-30:-15]
    test_s = s[-15:]
    test_s.index = range(len(test_s))
    #train_s.index = range(len(train_s))
    ########################################################################
    
    # In formation period, fit the SP500 price as a linear function of time.
    
    linear_f = np.polyfit(train_s.index, train_s, 1)
    s_ave = linear_f[0] * train_s.index + linear_f[1]
    a = linear_f[0]
    b = linear_f[1]
    y = train_s - s_ave
    plt.xlabel('time')
    plt.ylabel('price deviation from s_ave')
    plt.plot(y)
    
    plt.plot(train_s.index, s_ave, c='red')
    plt.scatter(train_s.index, train_s)
    plt.xlabel('time')
    plt.ylabel('adj close')
    
    # AR(1) model on ΔY(t)
    dy = y - y.shift(1)
    ar1_model = ARMA(dy[1:], order=(1,0))
    result = ar1_model.fit()
    print(result.summary())
    
    # plot acf of ΔY(t)
    plot_acf(dy[1:], lags= 10, alpha=0.05) # confidence: 95%
    plt.xlabel('lag')
    plt.ylabel('acf')
    
    #plt.title('Auto-correlation of ΔY')
    plt.show()
    
    
    # continuous time mean reverting model for Y, Δt=1
    
    A, B = result.params
    k = 1-B
    theta=A/k
    
    # Use the sample volatility of Y(t) in the portfolio formation period as the volatility parameter in your model
    sigma = np.sqrt(np.mean(dy**2)) # realized volatility
    
    ##########################################################################
    
    # formation period
    
    
    n = np.array([i*0.01 for i in range(200)])# change n for different ad hoc bands
    payoff = [0] * 200
    for i in range(len(n)):
        
        yl = theta - n[i]*sigma
        yu = theta + n[i]*sigma
        upper = s_ave + yu
        lower = s_ave + yl
    
        position = ad_hoc_position(train_s, upper, lower)
        initial_capital = 1000
        capital = ad_hoc_capital_process(initial_capital, train_s, position)[0]
        payoff[i] = capital[len(capital)-1] - initial_capital
    plt.plot(payoff)
    plt.xlabel("100n")
    plt.ylabel("payoff")
    #plt.title('Ad-hoc payoff')
    plt.show()
    
    ###############################################################################
    # buy&hold
    pay, re, mean, var, sr = buy_hold(initial_capital, train_s)
    
    # testing period
    
    
    window = 3 # change window for different rolling windows
    rolling_mean = test_s.rolling(window).mean()
    n = np.array([i*0.01 for i in range(200)])# change n for different ad hoc bands
    payoff = [0] * 200
    for i in range(len(n)):
        
        yl = theta - n[i]*sigma
        yu = theta + n[i]*sigma
        upper = rolling_mean + yu
        lower = rolling_mean + yl
    
        position = ad_hoc_position(test_s, upper, lower)
        # delay 1 day
        #position=position.shift(1)
        #position[0]=0
        initial_capital = 1000
        capital = ad_hoc_capital_process(initial_capital, test_s, position)[0]
        payoff[i] = capital[len(capital)-1] - initial_capital
    plt.plot(payoff)
    plt.xlabel("100n")
    plt.ylabel("payoff")
    #plt.title('Ad-hoc payoff')
    plt.show()
    
    # same ad hoc bands as in training period: n*std
    # choose the best n
    n=0.09
    yl = theta - n*sigma
    yu = theta + n*sigma
    upper_bound = rolling_mean + yu
    lower_bound = rolling_mean +yl
    
    
    position = ad_hoc_position(test_s, upper_bound, lower_bound)
    # delay 1 day
    #position=position.shift(1)
    #position[0]=0
    capital = ad_hoc_capital_process(initial_capital, test_s, position)[0]


    buy_hold_capital=buy_hold(initial_capital, test_s)[0]
    plt.xlabel("time")
    plt.ylabel("capital")
    
    
    
    long_short_c = long_short(initial_capital, test_s, upper_bound, lower_bound)
    
    plt.plot(capital, label='ad-hoc')
    plt.plot(buy_hold_capital, label='buy-hold')
    plt.plot(long_short_c, label='long-short')
    plt.xlabel("time")
    plt.ylabel("capital")
    plt.legend()
    
    
    
    ######################################################################
    
    # optimal bands
    
    
    
    sigma=sigma*252**0.5

    T=15/252
    Nt=4000
    smax=4000
    Ns=400


    dt = T/Nt
    ds = smax/Ns

    H = np.zeros((Nt+1, Ns+1))
    s = np.arange(0, smax+ds, ds)
    t = np.arange(0, T+dt, dt)
    # boundary
    # H(t,s)
    H[:,0]=0-0.01
    H[:,Ns]=smax-0.01
    H[Nt,]=s-0.01

    for i in range(Nt-1,-1,-1):
    
        for j in range(1, Ns):
            v=((k*(theta-s[j]+a*t[i]*252+b)+a)*dt/ds/2 + 0.5*sigma**2*dt/ds**2)*H[i+1,j+1]+(1-sigma**2*dt/ds**2)*H[i+1,j]+(-(k*(theta-s[j]+a*t[i]*252+b)+a)*dt/ds/2 + 0.5*sigma**2*dt/ds**2)*H[i+1,j-1]
            H[i, j]=max(v, s[j])


    upper=[]      
    for i in range(Nt):
        qs = H[i,]-s
        index = 0
        for q in qs[1:]:
            if q>0:
                index+=1
        upper+=[s[index]]
    upper=upper[:-1]   
            
    # 
    G = np.zeros((Nt+1, Ns+1))

    G[:,0]=H[:,0]-s[0]-0.01
    G[:,Ns]=H[:,Ns]-smax-0.01
    G[Nt,]=H[Nt,]-s-0.01
    lower=[0]*(Nt+1)
    m=np.zeros((Nt+1, Ns+1))
    for i in range(Nt-1,-1,-1):
    
        for j in range(1, Ns):
            v=(k*(theta-s[j])*dt/ds/2 + 0.5*sigma**2*dt/ds**2)*G[i+1,j+1]+(1-sigma**2*dt/ds**2)*G[i+1,j]+(-k*(theta-s[j])*dt/ds/2 + 0.5*sigma**2*dt/ds**2)*G[i+1,j-1]
            G[i, j]=max(v, H[i,j]-s[j])
            m[i, j]=H[i,j]-s[j]
    lower=[]
    for i in range(Nt):
        qs=G[i,]-m[i,]
        index=0
        for q in qs[1:]:
            if q==0:
                index+=1
            else:
                break
        lower+=[s[index]]
    
    lower=lower[:-2]
    
    
    
    
    
    plt.xlabel('time step')
    plt.ylabel('price')
    plt.plot(train_s.index*4000/15,train_s,label='s')
    plt.plot(a*252*t+b, label='s_ave')
    plt.plot(upper,label='upper bound')     
    plt.plot(lower,label='lower bound')  
    plt.legend()
    len(lower)
    
    
    s_ave=list([a*i+b for i in range(15)])
    yu=[0]*len(s_ave)
    for i in range(15):
        yu[i]=upper[i*4000//15]-s_ave[i]
    yl=[0]*len(s_ave)
    for i in range(15):
        yl[i]=lower[i*4000//15]-s_ave[i]
        
        
    
    
    s_ave_test=test_s.rolling(3).mean()
    
    su=[0]*len(s_ave_test)
    for i in range(15):
        su[i]=yu[i]+s_ave_test[i]
    sl=[0]*len(s_ave)
    for i in range(15):
        sl[i]=yl[i]+s_ave_test[i]
        
        
    position=ad_hoc_position(test_s, su, sl)
    #position=position.shift(1)
    #position[0]=0
    
    plt.plot(ad_hoc_capital_process(initial_capital, test_s, position)[0])
    position1=position.shift(1)
    position1[0]=0
    plt.plot(ad_hoc_capital_process(initial_capital, test_s, position1)[0],label='1-day delay')
    plt.xlabel("time")
    plt.ylabel("capital")
    plt.legend()

    '''
    p = s[-30:-15]
    plt.plot(p)
    '''
    l=buy_hold(initial_capital, test_s)
    long_short_c = long_short(initial_capital, test_s, su, sl)
    long_short_c.plot(label='long-short')
    
    l[0].plot(label='buy-hold')
    plt.xlabel("time")
    plt.ylabel("capital")
    plt.legend()
    
