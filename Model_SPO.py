
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from statsmodels.tsa.api import adfuller
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GM
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import confusion_matrix, classification_report

notowania = pd.read_csv('D:/Тимчасове/Projects/QI/Quant_Invest_Fundusze.csv',sep=";", index_col=0, parse_dates=True)
import itertools as itertools
tickery=list(notowania.columns.values)
pary=list(itertools.product(tickery,tickery))
print(pary)
nmbr=19

def possible_combinations(n):
    possible_pairs=(n*(n-1))
    return possible_pairs

possible_combinations(7)
def create_pairs(symbolList):
    pairs=[]
    x=0
    y=0
    for count,symbol in enumerate(symbolList):
        for nextCount,nextSymbol in enumerate(symbolList):
            x=symbol
            y=nextSymbol
            if x !=y:
                pairs.append([x,y])
    return pairs

pary=create_pairs(tickery)
def get_cointegrated(all_pairs,training_df):

    cointegrated=[]



    for count, pair in enumerate(all_pairs):
        try:

            ols=linregress(training_df[str(pair[1])],training_df[str(pair[0])])


            slope=ols[0]



            spread=training_df[str(pair[1])]-(slope*training_df[str(pair[0])])


            cadf=adfuller(spread,1)


            if cadf[0] < cadf[4]['1%']:
                print('Pair Cointegrated at 99% Confidence Interval')

                cointegrated.append([pair[0],pair[1]])
            elif cadf[0] < cadf[4]['5%']:
                print('Pair Cointegrated at 95% Confidence Interval')

                cointegrated.append([pair[0],pair[1]])
            elif cadf[0] < cadf[4]['10%']:
                print('Pair Cointegrated at 90% Confidence Interval')
                cointegrated.append(pair[0],pair[1])
            else:
                print('Pair Not Cointegrated ')
                continue
        except:
            print('Exception: Symbol not in Dataframe')
            continue

    return cointegrated



from scipy.stats import linregress

notowaniaw=notowania.resample('1W').last()
notowaniam=notowania.resample('1M').last()
notowaniaq=notowania.resample('3M').last()
notowaniay=notowania.resample('1Y').last()


cointegrated_from_cluster_0=get_cointegrated(pary,notowaniay['2000-01-01':'2018-01-01'])
cointegrated_from_cluster_0

notowania_tren=notowaniaw['2000-01-01':'2011-12-31']
OP_tren=np.array(notowania_tren['OP'])
AP_tren=np.array(notowania_tren['AP'])
ORR_tren=np.array(notowania_tren['ORR'])
G_tren=np.array(notowania_tren['G'])

notowania_test= notowaniaw['2012-01-01':'2017-01-01']
OP_test=np.array(notowania_test['OP'])
AP_test=np.array(notowania_test['AP'])
ORR_test=np.array(notowania_test['ORR'])
G_test=np.array(notowania_test['G'])




class statarb(object):

    def __init__(self,df1, df2,ma,floor, ceiling,beta_lookback,start,end,exit_zscore=0):
        self.df1=df1
        self.df2=df2
        self.ma=ma
        self.floor=floor
        self.ceiling=ceiling
        self.Close='Close Long'
        self.Cover='Cover Short'
        self.exit_zscore=exit_zscore
        self.beta_lookback=beta_lookback
        self.start=start
        self.end=end
    def create_spread(self):
        self.df=pd.DataFrame(index=range(0,len(self.df1)))

        try:
            self.df['X']=self.df1
            self.df['Y']=self.df2
        except:
            print('Length of self.df:')
            print(len(self.df))
            print('')
            print('Length of self.df1:')
            print(len(self.df1))
            print('')
            print('Length of self.df2:')
            print(len(self.df2))
        ols=linregress(self.df['Y'],self.df['X'])
        self.df['Beta']=ols[0]
        self.df['Spread']=self.df['Y']+(self.df['Beta'].rolling(window=self.beta_lookback).mean()*self.df['X'])
        return self.df.head()


    def generate_signals(self):

            self.df['Z-Score']=(self.df['Spread']-self.df['Spread'].rolling(window=self.ma).mean())/self.df['Spread'].rolling(window=self.ma).std()
            self.df['Prior Z-Score']=self.df['Z-Score'].shift(1)
            self.df['Longs']=(self.df['Z-Score']<=self.floor)*1.0
            self.df['Shorts']=(self.df['Z-Score']>=self.ceiling)*1.0
            self.df['Exit']=(np.abs(self.df['Z-Score'])<=self.exit_zscore)*1.0
            self.df['Long_Market']=0.0
            self.df['Short_Market']=0.0
            self.long_market=0
            self.short_market=0

            for i,value in enumerate(self.df.iterrows()):
                if (value[1]['Longs']==1.0):
                        self.long_market=1

                if value[1]['Shorts']==1.0:
                    self.short_market=0

                if value[1]['Exit']==1.0:

                    self.long_market=0
                    self.short_market=0

                self.df.iloc[i]['Long_Market']=self.long_market
                self.df.iloc[i]['Short_Market']=self.short_market



            return

    def create_returns(self, allocation,pair_number):
            self.allocation=allocation
            self.pair=pair_number

            self.portfolio=pd.DataFrame(index=self.df.index)
            self.portfolio['Positions']=self.df['Long_Market']-self.df['Short_Market']
            self.portfolio['X']=-0.0*self.df['X']*self.portfolio['Positions']
            self.portfolio['Y']=self.df['Y']*self.portfolio['Positions']
            self.portfolio['Total']=self.portfolio['X']+self.portfolio['Y']


            self.portfolio['Returns']=self.portfolio['Total'].pct_change()
            self.portfolio['Returns'].fillna(0.0,inplace=True)
            self.portfolio['Returns'].replace([np.inf,-np.inf],0.0,inplace=True)
            self.portfolio['Returns'].replace(-1.0,0.0,inplace=True)


            self.mu=(self.portfolio['Returns'].mean())
            self.sigma=(self.portfolio['Returns'].std())
            self.portfolio['Win']=np.where(self.portfolio['Returns']>0,1,0)
            self.portfolio['Loss']=np.where(self.portfolio['Returns']<0,1,0)
            self.wins=self.portfolio['Win'].sum()
            self.losses=self.portfolio['Loss'].sum()
            self.total_trades=self.wins+self.losses

            self.win_loss_ratio=(self.wins/self.losses)

            self.prob_of_win=(self.wins/self.total_trades)
            self.prob_of_loss=(self.losses/self.total_trades)

            self.avg_win_return=(self.portfolio['Returns']>0).mean()
            self.avg_loss_return=(self.portfolio['Returns']<0).mean()
            self.payout_ratio=(self.avg_win_return/self.avg_loss_return)

            self.portfolio['Returns']=(self.portfolio['Returns']+1.0).cumprod()
            self.portfolio['Trade Returns']=(self.portfolio['Total'].pct_change())
            self.portfolio['Portfolio Value']=(self.allocation*self.portfolio['Returns'])
            self.portfolio['Portfolio Returns']=self.portfolio['Portfolio Value'].pct_change()
            self.portfolio['Initial Value']=self.allocation

            with plt.style.context(['ggplot','seaborn-paper']):
                plt.plot(self.portfolio['Portfolio Value'])
                plt.plot(self.portfolio['Initial Value'])
                plt.title('%s Strategy Returns '%(self.pair))
                plt.legend(loc=0)
                plt.show()


            return



OP_AP=statarb(OP_test,AP_test,nmbr,-2,2,nmbr,'2012-01-01','2017-01-01')
OP_AP.create_spread()
OP_AP.generate_signals()
OP_AP.create_returns(10000,'OP_AP')


ORR_G=statarb(ORR_test,G_test,nmbr,-2,2,nmbr,'2012-01-01','2017-01-01')
ORR_G.create_spread()
ORR_G.generate_signals()
ORR_G.create_returns(10000,'ORR_G')

iloscJUg=10000/notowania_test['G'][0]

equally_weighted=pd.DataFrame()
equally_weighted['OP_AP']=OP_AP.portfolio['Portfolio Value']
equally_weighted['ORR_G']=ORR_G.portfolio['Portfolio Value']
equally_weighted['G']=np.array(iloscJUg*notowania_test['G'])
equally_weighted['Total Portfolio Value']=equally_weighted['OP_AP']+equally_weighted['ORR_G']+equally_weighted['G']
equally_weighted['Returns']=np.log(equally_weighted['Total Portfolio Value']/equally_weighted['Total Portfolio Value'].shift(1))

equally_weighted_mu=equally_weighted['Returns'].mean()
equally_weighted_sigma=equally_weighted['Returns'].std()

rate=0.015
equally_weighted_Sharpe=round((equally_weighted_mu-rate)/equally_weighted_sigma,2)

print('Equally Weighted Portfolio Sharpe:',equally_weighted_Sharpe)

plt.figure(figsize=(10,6))
plt.plot(equally_weighted['Total Portfolio Value'])
plt.title('Equally Weighted Portfolio Equity Curve')
plt.show()



OP_AP_mu=OP_AP.mu
OP_AP_sigma=OP_AP.sigma
ORR_G_mu=ORR_G.mu
ORR_G_sigma=ORR_G.sigma


returns=np.log(equally_weighted[['OP_AP','ORR_G']]/equally_weighted[['OP_AP','ORR_G']].shift(1))

avg_returns_52=returns.mean()*52
covariance_matrix=returns.cov()*52
weights=np.random.random(len(returns.columns))
weights/=np.sum(weights)
weights


import scipy.optimize as sco

def efficient_frontier(returns,rate=0.015):

    portfolio_returns=[]
    portfolio_volatility=[]
    p_sharpes=[]


    for i in range(500):
        weights=np.random.random(len(returns.columns))
        weights/=np.sum(weights)

        current_return=np.sum(returns.mean()*weights)*52
        portfolio_returns.append(current_return)

        variance=np.dot(weights.T,np.dot(returns.cov()*52,weights))
        volatility=np.sqrt(variance)
        portfolio_volatility.append(volatility)

        ratio=(current_return-rate)/volatility
        p_sharpes.append(ratio)

    p_returns=np.array(portfolio_returns)
    p_volatility=np.array(portfolio_volatility)
    p_sharpes=np.array(p_sharpes)

    plt.figure(figsize=(10,6))
    plt.scatter(p_volatility,p_returns,c=p_sharpes, marker='o')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

    return


efficient_frontier(returns.fillna(0))
def stats(weights,rate=0.015):
    weights=np.array(weights)
    p_returns=np.sum(returns.mean()*weights)*52
    p_volatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*52,weights)))
    p_sharpe=(p_returns-rate)/p_volatility

    return np.array([p_returns,p_volatility,p_sharpe])

stats(weights)


def TargetVol_const_lower(weights,target_vol=0.1) :
        weights=np.array(weights)
        p_returns=np.sum(returns.mean()*weights)*52
        p_volatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*52,weights)))
        p_vol=np.sqrt(np.dot(weights.T,np.dot(returns.cov(),weights)))
        vol_diffs = p_vol - (target_vol * 0.9)
        return(vol_diffs)

def TargetVol_const_upper(weights,target_vol=0.1) :
        weights=np.array(weights)
        p_returns=np.sum(returns.mean()*weights)*52
        p_volatility=np.sqrt(np.dot(weights.T,np.dot(returns.cov()*52,weights)))
        p_vol=np.sqrt(np.dot(weights.T,np.dot(returns.cov(),weights)))
        vol_diffs = (target_vol * 1.2) - p_vol
        return(vol_diffs)


def minimize_func(weights):
        return stats(weights)[1]

minimize_func(weights)


def get_optimal_weights(weights):
    constraints=({'type':'eq','fun':lambda x: np.sum(x)-1},
                 {'type': 'ineq', 'fun': TargetVol_const_lower},
                 {'type': 'ineq', 'fun': TargetVol_const_upper})
    options = {'ftol': 1e-20, 'maxiter': 5000}
    bounds=tuple((0,1) for x in range(len(returns.columns)))

    starting_weights=len(returns.columns)*[1./len(returns.columns)]
    most_optimal=sco.minimize(minimize_func,starting_weights, method='SLSQP', bounds=bounds, constraints=constraints,options = options)
    best_weights=most_optimal['x'].round(3)

    return best_weights, print('Weights:',best_weights)



optimal_weights=get_optimal_weights(weights)
total_allocation=20000

OP_AP_allocation=round(total_allocation*optimal_weights[0][0],2)
ORR_G_allocation=round(total_allocation*optimal_weights[0][1],2)
OP_AP_2=statarb(OP_test,AP_test,nmbr,-2,2,nmbr,'2012-01-01','2017-01-01')
OP_AP_2.create_spread()
OP_AP_2.generate_signals()
OP_AP_2.create_returns(OP_AP_allocation,'OP_AP_Portfolio _2')

ORR_G_2=statarb(G_test,ORR_test,nmbr,-2,2,nmbr,'2012-01-01','2017-01-01')
ORR_G_2.create_spread()
ORR_G_2.generate_signals()
ORR_G_2.create_returns(ORR_G_allocation,'ORR_G_Portfolio _2')



efficient_frontier_portfolio=pd.DataFrame()
efficient_frontier_portfolio['OP_AP']=OP_AP_2.portfolio['Portfolio Value']
efficient_frontier_portfolio['ORR_G']=ORR_G_2.portfolio['Portfolio Value']
efficient_frontier_portfolio['G']=np.array(iloscJUg*notowania_test['G'])
efficient_frontier_portfolio['Total Portfolio Value']=efficient_frontier_portfolio['OP_AP']+efficient_frontier_portfolio['ORR_G']+efficient_frontier_portfolio['G']

efficient_frontier_portfolio['Returns']=np.log(efficient_frontier_portfolio['Total Portfolio Value']/efficient_frontier_portfolio['Total Portfolio Value'].shift(1))

plt.figure(figsize=(10,6))
plt.plot(efficient_frontier_portfolio['Total Portfolio Value'])
plt.title('Efficient Frontier Portfolio Equity Curve')
plt.show()



efficient_frontier_portfolio_mu=efficient_frontier_portfolio['Returns'].mean()
efficient_frontier_portfolio_sigma=efficient_frontier_portfolio['Returns'].std()

efficient_frontier_portfolio_sharpe=(efficient_frontier_portfolio_mu-rate)/efficient_frontier_portfolio_sigma




len(notowania_tren)
round(len(notowania_tren)*.80)
round(len(notowania_tren)*-.20)

OP_AP_historical=statarb(OP_tren,AP_tren,nmbr,-2,2,nmbr,notowania_tren.iloc[0],notowania_tren.iloc[-1])
OP_AP_historical.create_spread()
OP_AP_historical.generate_signals()
OP_AP_historical.create_returns(10000,'OP_AP_ Over Training Period')


OP_AP_historical_rets=OP_AP_historical.portfolio['Returns']


ORR_G_historical=statarb(G_tren,ORR_tren,nmbr,-2,2,nmbr,notowania_tren.iloc[0],notowania_tren.iloc[-1])
ORR_G_historical.create_spread()
ORR_G_historical.generate_signals()
ORR_G_historical.create_returns(10000,'ORR_G_ Over Hist. Train Period')


ORR_G_historical_rets=ORR_G_historical.portfolio['Returns']

OP_AP_rets_len=len(OP_AP_historical_rets)
OP_AP_rets_train=OP_AP_historical_rets[0:520]
OP_AP_rets_test=OP_AP_historical_rets[-175:]

ORR_G_rets_len=len(ORR_G_historical_rets)
ORR_G_rets_train=ORR_G_historical_rets[0:520]
ORR_G_rets_test=ORR_G_historical_rets[-175:]


class gmm_randomForests(object):
    def __init__(self,historical_rets_train,historical_rets_test,base_portfolio_rets,gmm_components,df,base_portfolio_df,
                 internal_test_start,internal_test_end):
            self.historical_rets_train=historical_rets_train
            self.historical_rets_test=historical_rets_test
            self.base_portfolio_rets=base_portfolio_rets
            self.gmm_components=gmm_components
            self.max_iter=300
            self.random_state=101
            self.df=df
            self.base_portfolio_df=base_portfolio_df
            self.internal_test_start=internal_test_start
            self.internal_test_end=internal_test_end
            self.volatility=self.historical_rets_train.rolling(window=5).std()
            self.negative_volatility=np.where(self.historical_rets_train<0,self.historical_rets_train.rolling(window=5).std(),0)


    def make_gmm(self):
        model_kwds=dict(n_components=self.gmm_components,max_iter=self.max_iter,n_init=100,random_state=self.random_state)

        gmm=GM(**model_kwds)
        return gmm

    def analyze_historical_regimes(self):
        self.gmm=self.make_gmm()

        self.gmm_XTrain=np.array(self.historical_rets_train).reshape(-1,1)
        self.gmm.fit(self.gmm_XTrain.astype(int))
        self.gmm_historical_predictions=self.gmm.predict(self.gmm_XTrain.astype(int))
        self.gmm_XTest=np.array(self.historical_rets_test).reshape(-1,1)
        self.gmm_training_test_predictions=self.gmm.predict(self.gmm_XTest.astype(int))
        self.gmm_Actual=np.array(self.base_portfolio_rets).reshape(-1,1)
        self.base_portfolio_predictions=self.gmm.predict(self.gmm_Actual)


        return

    def historical_regime_returns_volatility(self,plotTitle):
        self.plotTitle=plotTitle
        data=pd.DataFrame({'Volatility':self.volatility,'Regime':self.gmm_historical_predictions,'Returns':self.historical_rets_train})

        with plt.style.context(['classic','seaborn-paper']):
            fig,ax=plt.subplots(figsize=(15,10),nrows=1, ncols=2)

            left   =  0.125
            right  =  0.9
            bottom =  .125
            top    =  0.9
            wspace =  .5
            hspace =  1.1

            plt.subplots_adjust(
                left    =  left,
                bottom  =  bottom,
                right   =  right,
                top     =  top,
                wspace  =  wspace,
                hspace  =  hspace
            )


            y_title_margin = 2

            plt.suptitle(self.plotTitle, y = 1, fontsize=20)

            plt.subplot(121)
            sns.swarmplot(x='Regime',y='Volatility',data=data)#,ax=ax[0][0])
            plt.title('Regime to Volatility')

            plt.subplot(122)
            sns.swarmplot(x='Regime',y='Returns',data=data)#, ax=ax[0][1])
            plt.title('Regime to Returns')
            plt.tight_layout()
            plt.show()

            return


    def train_random_forests(self):

            self.df['6 X Vol']=self.df['X'].rolling(window=6).std()
            self.df['6 Y Vol']=self.df['Y'].rolling(window=6).std()
            self.df['6 Spread Vol']=self.df['Spread'].rolling(window=6).std()
            self.df['6 Z-Score Vol']=self.df['Z-Score'].rolling(window=6).std()

            self.df['12 X Vol']=self.df['X'].rolling(window=12).std()
            self.df['12 Y Vol']=self.df['Y'].rolling(window=12).std()
            self.df['12 Spread Vol']=self.df['Spread'].rolling(window=12).std()
            self.df['12 Z-Score Vol']=self.df['Z-Score'].rolling(window=12).std()

            self.df['15 X Vol']=self.df['X'].rolling(window=15).std()
            self.df['15 Y Vol']=self.df['Y'].rolling(window=15).std()
            self.df['15 Spread Vol']=self.df['Spread'].rolling(window=15).std()
            self.df['15 Z-Score Vol']=self.df['Z-Score'].rolling(window=15).std()

            self.base_portfolio_df['6 X Vol']=self.df['X'].rolling(window=6).std()
            self.base_portfolio_df['6 Y Vol']=self.df['Y'].rolling(window=6).std()
            self.base_portfolio_df['6 Spread Vol']=self.df['Spread'].rolling(window=6).std()
            self.base_portfolio_df['6 Z-Score Vol']=self.df['Z-Score'].rolling(window=6).std()

            self.base_portfolio_df['12 X Vol']=self.df['X'].rolling(window=12).std()
            self.base_portfolio_df['12 Y Vol']=self.df['Y'].rolling(window=12).std()
            self.base_portfolio_df['12 Spread Vol']=self.df['Spread'].rolling(window=12).std()
            self.base_portfolio_df['12 Z-Score Vol']=self.df['Z-Score'].rolling(window=12).std()

            self.base_portfolio_df['15 X Vol']=self.df['X'].rolling(window=15).std()
            self.base_portfolio_df['15 Y Vol']=self.df['Y'].rolling(window=15).std()
            self.base_portfolio_df['15 Spread Vol']=self.df['Spread'].rolling(window=15).std()
            self.base_portfolio_df['15 Z-Score Vol']=self.df['Z-Score'].rolling(window=15).std()


            self.df.fillna(0, inplace=True)
            self.RF_X_TRAIN=self.df[0:8000][['6 X Vol','6 Y Vol','6 Spread Vol','6 Z-Score Vol','12 X Vol','12 Y Vol',
                                               '12 Spread Vol','12 Z-Score Vol','15 X Vol','15 Y Vol','15 Spread Vol','15 Z-Score Vol']]
            self.RF_Y_TRAIN=self.gmm_historical_predictions
            self.RF_X_TEST=self.base_portfolio_df[['6 X Vol','6 Y Vol','6 Spread Vol','6 Z-Score Vol','12 X Vol','12 Y Vol',
                                               '12 Spread Vol','12 Z-Score Vol','15 X Vol','15 Y Vol','15 Spread Vol','15 Z-Score Vol']]\

            self.RF_Y_TEST=self.base_portfolio_predictions

            self.RF_MODEL=RF(n_estimators=100)
            self.RF_MODEL.fit(self.RF_X_TRAIN.fillna(0),self.RF_Y_TRAIN)

            self.RF_BASE_PORTFOLIO_PREDICTIONS=self.RF_MODEL.predict(self.RF_X_TEST.fillna(0))

            return






OP_AP_gmm_rf=gmm_randomForests(OP_AP_historical_rets,OP_AP_rets_test,OP_AP_2.portfolio['Returns'],5,
                                 OP_AP_historical.df, OP_AP.df,520,-175)
OP_AP_gmm_rf.analyze_historical_regimes()
OP_AP_gmm_rf.historical_regime_returns_volatility('OP_AP GMM Analysis')


OP_AP_gmm_rf.train_random_forests()


ORR_G_gmm_rf=gmm_randomForests(ORR_G_historical_rets,ORR_G_rets_test,ORR_G_2.portfolio['Returns'],5,
                                 ORR_G_historical.df, ORR_G.df,520,-175)
ORR_G_gmm_rf.analyze_historical_regimes()
ORR_G_gmm_rf.historical_regime_returns_volatility('ORR_G GMM Analysis')


ORR_G_gmm_rf.train_random_forests()


OP_AP_regime_predictions=OP_AP_gmm_rf.base_portfolio_predictions
ORR_G_regime_predictions=ORR_G_gmm_rf.base_portfolio_predictions



class statarb_update(object):

     def __init__(self,df1, df2, ptype,ma,floor, ceiling,beta_lookback,start,end,regimePredictions,p2Objective,avoid1=0,target1=0,
                  exit_zscore=0):
        self.df1=df1
        self.df2=df2
        self.df=pd.DataFrame(index=df1.index)
        self.ptype=ptype
        self.ma=ma
        self.floor=floor
        self.ceiling=ceiling
        self.Close='Close Long'
        self.Cover='Cover Short'
        self.exit_zscore=exit_zscore
        self.beta_lookback=beta_lookback
        self.start=start
        self.end=end
        self.regimePredictions=regimePredictions.reshape(-1,1)
        self.avoid1=avoid1
        self.target1=target1
        self.p2Objective=p2Objective


     def create_spread(self):
            if self.ptype==1:
                self.df['X']=self.df1
                self.df['Y']=self.df2

                self.ols=linregress(self.df['Y'],self.df['X'])

                self.df['Hedge Ratio']=self.ols[0]

                self.df['Spread']=self.df['Y']-(self.df['Hedge Ratio']*self.df['X'])

            if self.ptype==2:
                self.df['X']=self.df1
                self.df['Y']=self.df2


                self.ols=linregress(self.df['Y'],self.df['X'])
                self.df['Hedge Ratio']=self.ols[0]
                self.df['Spread']=self.df['Y']+(self.df['Hedge Ratio']*self.df['X'])

                self.df['Z-Score']=(self.df['Spread']-self.df['Spread'].rolling(window=self.ma).mean())/self.df['Spread'].rolling(window=self.ma).std()

                self.df['6 X Vol']=self.df['X'].rolling(window=6).std()
                self.df['6 Y Vol']=self.df['Y'].rolling(window=6).std()
                self.df['6 Spread Vol']=self.df['Spread'].rolling(window=6).std()
                self.df['6 Z-Score Vol']=self.df['Z-Score'].rolling(window=6).std()

                self.df['12 X Vol']=self.df['X'].rolling(window=12).std()
                self.df['12 Y Vol']=self.df['Y'].rolling(window=12).std()
                self.df['12 Spread Vol']=self.df['Spread'].rolling(window=12).std()
                self.df['12 Z-Score Vol']=self.df['Z-Score'].rolling(window=12).std()

                self.df['15 X Vol']=self.df['X'].rolling(window=15).std()
                self.df['15 Y Vol']=self.df['Y'].rolling(window=15).std()
                self.df['15 Spread Vol']=self.df['Spread'].rolling(window=15).std()
                self.df['15 Z-Score Vol']=self.df['Z-Score'].rolling(window=15).std()
                self.df['Regime']=0
                self.df['Regime']=self.regimePredictions.astype(int)




            return


     def generate_signals(self):
            if self.ptype==1:

                self.df['Z-Score']=(self.df['Spread']+self.df['Spread'].rolling(window=self.ma).mean())/self.df['Spread'].rolling(window=self.ma).std()

                self.df['Prior Z-Score']=self.df['Z-Score'].shift(1)


                self.df['Longs']=(self.df['Z-Score']<=self.floor)*1.0
                self.df['Shorts']=(self.df['Z-Score']>=self.ceiling)*1.0
                self.df['Exit']=(self.df['Z-Score']<=self.exit_zscore)*1.0

                self.df['Long_Market']=0.0
                self.df['Short_Market']=0.0

                self.long_market=0
                self.short_market=0

                for i,value in enumerate(self.df.iterrows()):
                    if value[1]['Longs']==1.0:
                        self.long_market=1

                    if value[1]['Shorts']==1.0:
                        self.short_market=0

                    if value[1]['Exit']==1.0:

                        self.long_market=0
                        self.short_market=0

                    self.df.iloc[i]['Long_Market']=self.long_market
                    self.df.iloc[i]['Short_Market']=self.short_market


            if self.ptype==2:



                self.df['Longs']=(self.df['Z-Score']<=self.floor)*1.0
                self.df['Shorts']=(self.df['Z-Score']>=self.ceiling)*1.0
                self.df['Exit']=(self.df['Z-Score']<=self.exit_zscore)*1.0
                self.df['Long_Market']=0.0
                self.df['Short_Market']=0.0

                self.long_market=0
                self.short_market=0
                for i,value in enumerate(self.df.iterrows()):
                    if self.p2Objective=='Avoid':
                        if value[1]['Regime']!= self.avoid1:

                            if value[1]['Longs']==1.0:
                                self.long_market=1

                            if value[1]['Shorts']==1.0:
                                self.short_market=0

                            if value[1]['Exit']==1.0:

                                self.long_market=0
                                self.short_market=0

                        self.df.iloc[i]['Long_Market']=value[1]['Longs']
                        self.df.iloc[i]['Short_Market']=value[1]['Shorts']

                    elif self.p2Objective=='Target':
                        if value[1]['Regime']==self.target1:
                            if value[1]['Longs']==1.0:
                                self.long_market=1

                            if value[1]['Shorts']==1.0:
                                self.short_market=1

                            if value[1]['Exit']==1.0:

                                self.long_market=0
                                self.short_market=0

                        self.df.iloc[i]['Long_Market']=value[1]['Longs']
                        self.df.iloc[i]['Short_Market']=value[1]['Shorts']

                    elif self.p2Objective=='None':

                        if value[1]['Longs']==1.0:
                            self.long_market=1
                        if value[1]['Shorts']==1.0:
                            self.short_market=1

                        if value[1]['Exit']==1.0:

                            self.long_market=0
                            self.short_market=0

                        self.df.iloc[i]['Long_Market']=value[1]['Longs']
                        self.df.iloc[i]['Short_Market']=value[1]['Shorts']




            return self.df

     def create_returns(self, allocation,pair_number):
        if self.ptype==1:
            self.allocation=allocation
            self.pair=pair_number

            self.portfolio=pd.DataFrame(index=self.df.index)
            self.portfolio['Positions']=self.df['Long_Market']-self.df['Short_Market']
            self.portfolio['X']=-1.0*self.df['X']*self.portfolio['Positions']
            self.portfolio['Y']=self.df['Y']*self.portfolio['Positions']
            self.portfolio['Total']=self.portfolio['X']+self.portfolio['Y']


            self.portfolio['Returns']=self.portfolio['Total'].pct_change()
            self.portfolio['Returns'].fillna(0.0,inplace=True)
            self.portfolio['Returns'].replace([np.inf,-np.inf],0.0,inplace=True)
            self.portfolio['Returns'].replace(-1.0,0.0,inplace=True)


            self.mu=(self.portfolio['Returns'].mean())
            self.sigma=(self.portfolio['Returns'].std())
            self.portfolio['Win']=np.where(self.portfolio['Returns']>0,1,0)
            self.portfolio['Loss']=np.where(self.portfolio['Returns']<0,1,0)
            self.wins=self.portfolio['Win'].sum()
            self.losses=self.portfolio['Loss'].sum()
            self.total_trades=self.wins+self.losses
            self.win_loss_ratio=(self.wins/self.losses)

            self.prob_of_win=(self.wins/self.total_trades)
            self.prob_of_loss=(self.losses/self.total_trades)

            self.avg_win_return=(self.portfolio['Returns']>0).mean()
            self.avg_loss_return=(self.portfolio['Returns']<0).mean()
            self.payout_ratio=(self.avg_win_return/self.avg_loss_return)

            self.portfolio['Returns']=(self.portfolio['Returns']+1.0).cumprod()
            self.portfolio['Trade Returns']=(self.portfolio['Total'].pct_change())
            self.portfolio['Portfolio Value']=(self.allocation*self.portfolio['Returns'])
            self.portfolio['Portfolio Returns']=self.portfolio['Portfolio Value'].pct_change()
            self.portfolio['Initial Value']=self.allocation

            with plt.style.context(['ggplot','seaborn-paper']):
                plt.plot(self.portfolio['Portfolio Value'])
                plt.plot(self.portfolio['Initial Value'])
                plt.title('%s Strategy Returns '%(self.pair))
                plt.legend(loc=0)
                plt.show()




        if self.ptype==2:
            self.allocation=allocation
            self.pair=pair_number

            self.portfolio=pd.DataFrame(index=self.df.index)
            self.portfolio['Positions']=self.df['Longs']-self.df['Shorts']
            self.portfolio['X']=-1.0*self.df['X']*self.portfolio['Positions']
            self.portfolio['Y']=self.df['Y']*self.portfolio['Positions']
            self.portfolio['Total']=self.portfolio['X']+self.portfolio['Y']

            self.portfolio.fillna(0.0,inplace=True)


            self.portfolio['Returns']=self.portfolio['Total'].pct_change()
            self.portfolio['Returns'].fillna(0.0,inplace=True)
            self.portfolio['Returns'].replace([np.inf,-np.inf],0.0,inplace=True)
            self.portfolio['Returns'].replace(-1.0,0.0,inplace=True)


            self.mu=(self.portfolio['Returns'].mean())
            self.sigma=(self.portfolio['Returns'].std())
            self.portfolio['Win']=np.where(self.portfolio['Returns']>0,1,0)
            self.portfolio['Loss']=np.where(self.portfolio['Returns']<0,1,0)
            self.wins=self.portfolio['Win'].sum()
            self.losses=self.portfolio['Loss'].sum()
            self.total_trades=self.wins+self.losses
            self.win_loss_ratio=(self.wins/self.losses)

            self.prob_of_win=(self.wins/self.total_trades)
            self.prob_of_loss=(self.losses/self.total_trades)

            self.avg_win_return=(self.portfolio['Returns']>0).mean()
            self.avg_loss_return=(self.portfolio['Returns']<0).mean()
            self.payout_ratio=(self.avg_win_return/self.avg_loss_return)

            self.portfolio['Returns']=(self.portfolio['Returns']+1.0).cumprod()
            self.portfolio['Trade Returns']=(self.portfolio['Total'].pct_change())
            self.portfolio['Portfolio Value']=(self.allocation*self.portfolio['Returns'])
            self.portfolio['Portfolio Returns']=self.portfolio['Portfolio Value'].pct_change()
            self.portfolio['Initial Value']=self.allocation

            with plt.style.context(['ggplot','seaborn-paper']):
                plt.plot(self.portfolio['Portfolio Value'])
                plt.plot(self.portfolio['Initial Value'])
                plt.title('%s Strategy Returns '%(self.pair))
                plt.legend(loc=0)
                plt.show()




        return




OP_AP_spo=statarb_update(pd.DataFrame(OP_test), pd.DataFrame(AP_test), 2,nmbr,-2, 2,nmbr,notowania_test.iloc[0],notowania_test.iloc[-1],OP_AP_regime_predictions,'Target',avoid1=0,target1=1,
                  exit_zscore=0)
ORR_G_spo=statarb_update(pd.DataFrame(AP_test), pd.DataFrame(ORR_test), 2,nmbr,-2, 2,nmbr,notowania_test.iloc[0],notowania_test.iloc[-1],ORR_G_regime_predictions,'Target',avoid1=0,target1=1,
                  exit_zscore=0)

OP_AP_spo.create_spread()
ORR_G_spo.create_spread()
OP_AP_spo.generate_signals()
ORR_G_spo.generate_signals()

OP_AP_spo.create_returns(OP_AP_allocation,'OP_AP SPO Framework')
ORR_G_spo.create_returns(ORR_G_allocation, 'ORR_G SPO Framework')
spo_portfolio=pd.DataFrame()
spo_portfolio['OP_AP']=OP_AP_spo.portfolio['Portfolio Value']
spo_portfolio['ORR_G']= ORR_G_spo.portfolio['Portfolio Value']
spo_portfolio['G']=np.array(iloscJUg*notowania_test['G'])
spo_portfolio['Total Portfolio Value']=spo_portfolio['OP_AP']+ spo_portfolio['ORR_G']+spo_portfolio['G']
spo_portfolio['Returns']=np.log(spo_portfolio['Total Portfolio Value']/spo_portfolio['Total Portfolio Value'].shift(1))
spo_portfolio_mu=spo_portfolio['Returns'].mean()
spo_portfolio_sigma=spo_portfolio['Returns'].std()

spo_portfolio_sharpe=(spo_portfolio_mu-rate)/spo_portfolio_sigma

plt.figure(figsize=(10,6))
plt.plot(spo_portfolio['Total Portfolio Value'])
plt.title('SPO Portfolio Equity Curve')
plt.show()

print(spo_portfolio['Total Portfolio Value'])

print(equally_weighted['Total Portfolio Value'])

print(efficient_frontier_portfolio['Total Portfolio Value'])


spo_portfolio.index=notowania_test.index
spo_portfoliomret=spo_portfolio["Returns"].resample("1M").sum()
spo_portfoliomret.std()
