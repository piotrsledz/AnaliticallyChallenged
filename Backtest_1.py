#Test wsteczny 1: okres kryzysu
#Lata trenowania modelu 2000-2006
#Lata testowania modelu 2007-2009



notowania_tren3=notowaniaw['2000-01-01':'2006-12-31']
OP_tren3=np.array(notowania_tren3['OP'])
AP_tren3=np.array(notowania_tren3['AP'])
ORR_tren3=np.array(notowania_tren3['ORR'])
G_tren3=np.array(notowania_tren3['G'])

notowania_test3= notowaniaw['2007-01-01':'2009-12-31']
OP_test3=np.array(notowania_test3['OP'])
AP_test3=np.array(notowania_test3['AP'])
ORR_test3=np.array(notowania_test3['ORR'])
G_test3=np.array(notowania_test3['G'])

OP_AP_2007=statarb(OP_test3,AP_test3,nmbr,-1,2,nmbr,'2007-01-01','2009-12-31')
OP_AP_2007.create_spread()
OP_AP_2007.generate_signals()
OP_AP_2007.create_returns(10000,'OP_AP 2007')


ORR_G_2007=statarb(ORR_test3,ORR_test3,nmbr,-1,2,nmbr,'2007-01-01','2009-12-31')
ORR_G_2007.create_spread()
ORR_G_2007.generate_signals()
ORR_G_2007.create_returns(10000,'ORR_G 2007')

iloscJUg3 =10000/notowania_test3['G'][0]

equally_weighted_2007=pd.DataFrame()
equally_weighted_2007['OP_AP']=OP_AP_2007.portfolio['Portfolio Value']
equally_weighted_2007['ORR_G']=ORR_G_2007.portfolio['Portfolio Value']
equally_weighted_2007['G']=np.array(iloscJUg3*notowania_test3['G'])
equally_weighted_2007['Total Portfolio Value']=equally_weighted_2007['OP_AP']+equally_weighted_2007['ORR_G']+equally_weighted_2007['G']
equally_weighted_2007['Returns']=np.log(equally_weighted_2007['Total Portfolio Value']/equally_weighted_2007['Total Portfolio Value'].shift(1))

plt.figure(figsize=(10,6))
plt.plot(equally_weighted_2007['Total Portfolio Value'])
plt.title('Equally Weighted Portfolio Equity Curve in 2007')
plt.show()

print(equally_weighted_2007['Total Portfolio Value'])

print(efficient_frontier_portfolio_2007['Total Portfolio Value'])

OP_AP_allocation=round(total_allocation*optimal_weights[0][0],2)
ORR_G_allocation=round(total_allocation*optimal_weights[0][1],2)
OP_AP_2_2007=statarb(OP_test3,AP_test3,nmbr,-1,2,nmbr,'2007-01-01','2009-12-31')
OP_AP_2_2007.create_spread()
OP_AP_2_2007.generate_signals()
OP_AP_2_2007.create_returns(OP_AP_allocation,'OP_AP_Portfolio _2 in 2007')

ORR_G_2_2007=statarb(G_test3,ORR_test3,nmbr,-1,2,nmbr,'2007-01-01','2009-12-31')
ORR_G_2_2007.create_spread()
ORR_G_2_2007.generate_signals()
ORR_G_2_2007.create_returns(ORR_G_allocation,'ORR_G_Portfolio _2 in 2007')



efficient_frontier_portfolio_2007=pd.DataFrame()
efficient_frontier_portfolio_2007['OP_AP']=OP_AP_2_2007.portfolio['Portfolio Value']
efficient_frontier_portfolio_2007['ORR_G']=ORR_G_2_2007.portfolio['Portfolio Value']
efficient_frontier_portfolio_2007['G']=np.array(iloscJUg3*notowania_test3['G'])
efficient_frontier_portfolio_2007['Total Portfolio Value']=efficient_frontier_portfolio_2007['OP_AP']+efficient_frontier_portfolio_2007['ORR_G']+efficient_frontier_portfolio_2007['G']
efficient_frontier_portfolio_2007['Returns']=np.log(efficient_frontier_portfolio_2007['Total Portfolio Value']/efficient_frontier_portfolio_2007['Total Portfolio Value'].shift(1))

plt.figure(figsize=(10,6))
plt.plot(efficient_frontier_portfolio_2007['Total Portfolio Value'])
plt.title('Efficient Frontier Portfolio Equity Curve in 2007')
plt.show()



OP_AP_historical3=statarb(OP_tren3,AP_tren3,nmbr,-2,2,nmbr,notowania_tren3.iloc[0],notowania_tren3.iloc[-1])
OP_AP_historical3.create_spread()
OP_AP_historical3.generate_signals()
OP_AP_historical3.create_returns(10000,'OP_AP_ Over Training Period 2007')


OP_AP_historical_rets3=OP_AP_historical3.portfolio['Returns']


ORR_G_historical3=statarb(G_tren3,ORR_tren3,nmbr,-2,2,nmbr,notowania_tren3.iloc[0],notowania_tren3.iloc[-1])
ORR_G_historical3.create_spread()
ORR_G_historical3.generate_signals()
ORR_G_historical3.create_returns(10000,'ORR_G_ Over Hist. Train Period 2007')


ORR_G_historical_rets3=ORR_G_historical3.portfolio['Returns']

OP_AP_rets_len2=len(OP_AP_historical_rets3)
OP_AP_rets_train2=OP_AP_historical_rets3[0:295]
OP_AP_rets_test3=OP_AP_historical_rets3[-75:]

ORR_G_rets_len2=len(ORR_G_historical_rets3)
ORR_G_rets_train2=ORR_G_historical_rets3[0:295]
ORR_G_rets_test3=ORR_G_historical_rets3[-75:]




OP_AP_gmm_rf_2007=gmm_randomForests(OP_AP_historical_rets3,OP_AP_rets_test3,OP_AP_2_2007.portfolio['Returns'],5,
                                 OP_AP_historical3.df, OP_AP_2007.df,295,-75)
OP_AP_gmm_rf_2007.analyze_historical_regimes()
OP_AP_gmm_rf_2007.historical_regime_returns_volatility('OP_AP GMM Analysis 2007')


OP_AP_gmm_rf_2007.train_random_forests()


ORR_G_gmm_rf_2007=gmm_randomForests(ORR_G_historical_rets3,ORR_G_rets_test3,ORR_G_2_2007.portfolio['Returns'],5,
                                 ORR_G_historical3.df, ORR_G_2007.df,295,-75)
ORR_G_gmm_rf_2007.analyze_historical_regimes()
ORR_G_gmm_rf_2007.historical_regime_returns_volatility('ORR_G GMM Analysis 2007')


ORR_G_gmm_rf_2007.train_random_forests()


OP_AP_regime_predictions_2007=OP_AP_gmm_rf_2007.base_portfolio_predictions
ORR_G_regime_predictions_2007=ORR_G_gmm_rf_2007.base_portfolio_predictions





OP_AP_spo_2007=statarb_update(pd.DataFrame(OP_test3), pd.DataFrame(AP_test3),2,nmbr,-1,2,nmbr,notowania_test3.iloc[0],notowania_test3.iloc[-1],OP_AP_regime_predictions_2007,'Target',avoid1=0,target1=1,
                  exit_zscore=0)
ORR_G_spo_2007=statarb_update(pd.DataFrame(G_test3), pd.DataFrame(ORR_test3),2,nmbr,-1,2,nmbr,notowania_test3.iloc[0],notowania_test3.iloc[-1],ORR_G_regime_predictions_2007,'Target',avoid1=1,target1=0,
                  exit_zscore=0)

OP_AP_spo_2007.create_spread()
ORR_G_spo_2007.create_spread()
OP_AP_spo_2007.generate_signals()
ORR_G_spo_2007.generate_signals()

OP_AP_spo_2007.create_returns(OP_AP_allocation,'OP_AP SPO Framework')
ORR_G_spo_2007.create_returns(ORR_G_allocation, 'ORR_G SPO Framework')
spo_portfolio_2007=pd.DataFrame()
spo_portfolio_2007['OP_AP']=OP_AP_spo_2007.portfolio['Portfolio Value']
spo_portfolio_2007['ORR_G']= ORR_G_spo_2007.portfolio['Portfolio Value']
spo_portfolio_2007['G']=np.array(iloscJUg3*notowania_test3['G'])
spo_portfolio_2007['Total Portfolio Value']=spo_portfolio_2007['OP_AP']+ spo_portfolio_2007['ORR_G']+spo_portfolio_2007['G']
spo_portfolio_2007['Returns']=np.log(spo_portfolio_2007['Total Portfolio Value']/spo_portfolio_2007['Total Portfolio Value'].shift(1))


plt.figure(figsize=(10,6))
plt.plot(spo_portfolio_2007['Total Portfolio Value'])
plt.title('SPO Portfolio Equity Curve in 2007')
plt.show()
