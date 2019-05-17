#Test wsteczny 3: skrócenie okresu testowania modelu oraz wydłużenie okresu trenowania
#Lata trenowania modelu 2000-2011
#Lata testowania modelu 2012-2015



notowania_tren2=notowaniaw['2000-01-01':'2014-12-31']
OP_tren2=np.array(notowania_tren2['OP'])
AP_tren2=np.array(notowania_tren2['AP'])
ORR_tren2=np.array(notowania_tren2['ORR'])
G_tren2=np.array(notowania_tren2['G'])

notowania_test2= notowaniaw['2015-01-01':'2018-12-31']
OP_test2=np.array(notowania_test2['OP'])
AP_test2=np.array(notowania_test2['AP'])
ORR_test2=np.array(notowania_test2['ORR'])
G_test2=np.array(notowania_test2['G'])

OP_AP_2015=statarb(OP_test2,AP_test2,nmbr,-1,2,nmbr,'2015-01-01','2018-12-31')
OP_AP_2015.create_spread()
OP_AP_2015.generate_signals()
OP_AP_2015.create_returns(10000,'OP_AP 2015')


ORR_G_2015=statarb(ORR_test2,ORR_test2,nmbr,-1,2,nmbr,'2015-01-01','2018-12-31')
ORR_G_2015.create_spread()
ORR_G_2015.generate_signals()
ORR_G_2015.create_returns(10000,'ORR_G 2015')

iloscJUg2 =10000/notowania_test2['G'][0]

equally_weighted_2015=pd.DataFrame()
equally_weighted_2015['OP_AP']=OP_AP_2015.portfolio['Portfolio Value']
equally_weighted_2015['ORR_G']=ORR_G_2015.portfolio['Portfolio Value']
equally_weighted_2015['G']=np.array(iloscJUg2*notowania_test2['G'])
equally_weighted_2015['Total Portfolio Value']=equally_weighted_2015['OP_AP']+equally_weighted_2015['ORR_G']+equally_weighted_2015['G']
equally_weighted_2015['Returns']=np.log(equally_weighted_2015['Total Portfolio Value']/equally_weighted_2015['Total Portfolio Value'].shift(1))

plt.figure(figsize=(10,6))
plt.plot(equally_weighted_2015['Total Portfolio Value'])
plt.title('Equally Weighted Portfolio Equity Curve in 2015')
plt.show()




OP_AP_allocation=round(total_allocation*optimal_weights[0][0],2)
ORR_G_allocation=round(total_allocation*optimal_weights[0][1],2)
OP_AP_2_2015=statarb(OP_test2,AP_test2,nmbr,-1,2,nmbr,'2015-01-01','2018-12-31')
OP_AP_2_2015.create_spread()
OP_AP_2_2015.generate_signals()
OP_AP_2_2015.create_returns(OP_AP_allocation,'OP_AP_Portfolio _2 in 2015')

ORR_G_2_2015=statarb(G_test2,ORR_test2,nmbr,-1,2,nmbr,'2015-01-01','2018-12-31')
ORR_G_2_2015.create_spread()
ORR_G_2_2015.generate_signals()
ORR_G_2_2015.create_returns(ORR_G_allocation,'ORR_G_Portfolio _2 in 2015')



efficient_frontier_portfolio_2015=pd.DataFrame()
efficient_frontier_portfolio_2015['OP_AP']=OP_AP_2_2015.portfolio['Portfolio Value']
efficient_frontier_portfolio_2015['ORR_G']=ORR_G_2_2015.portfolio['Portfolio Value']
efficient_frontier_portfolio_2015['G']=np.array(iloscJUg2*notowania_test2['G'])
efficient_frontier_portfolio_2015['Total Portfolio Value']=efficient_frontier_portfolio_2015['OP_AP']+efficient_frontier_portfolio_2015['ORR_G']+efficient_frontier_portfolio_2015['G']
efficient_frontier_portfolio_2015['Returns']=np.log(efficient_frontier_portfolio_2015['Total Portfolio Value']/efficient_frontier_portfolio_2015['Total Portfolio Value'].shift(1))

plt.figure(figsize=(10,6))
plt.plot(efficient_frontier_portfolio_2015['Total Portfolio Value'])
plt.title('Efficient Frontier Portfolio Equity Curve in 2015')
plt.show()



OP_AP_historical2=statarb(OP_tren2,AP_tren2,nmbr,-2,2,nmbr,notowania_tren2.iloc[0],notowania_tren2.iloc[-1])
OP_AP_historical2.create_spread()
OP_AP_historical2.generate_signals()
OP_AP_historical2.create_returns(10000,'OP_AP_ Over Training Period 2015')


OP_AP_historical_rets2=OP_AP_historical2.portfolio['Returns']


ORR_G_historical2=statarb(G_tren2,ORR_tren2,nmbr,-2,2,nmbr,notowania_tren2.iloc[0],notowania_tren2.iloc[-1])
ORR_G_historical2.create_spread()
ORR_G_historical2.generate_signals()
ORR_G_historical2.create_returns(10000,'ORR_G_ Over Hist. Train Period 2015')


ORR_G_historical_rets2=ORR_G_historical2.portfolio['Returns']

OP_AP_rets_len2=len(OP_AP_historical_rets2)
OP_AP_rets_train2=OP_AP_historical_rets2[0:626]
OP_AP_rets_test2=OP_AP_historical_rets2[-160:]

ORR_G_rets_len2=len(ORR_G_historical_rets2)
ORR_G_rets_train2=ORR_G_historical_rets2[0:626]
ORR_G_rets_test2=ORR_G_historical_rets2[-160:]




OP_AP_gmm_rf_2015=gmm_randomForests(OP_AP_historical_rets2,OP_AP_rets_test2,OP_AP_2_2015.portfolio['Returns'],5,
                                 OP_AP_historical2.df, OP_AP_2015.df,626,-160)
OP_AP_gmm_rf_2015.analyze_historical_regimes()
OP_AP_gmm_rf_2015.historical_regime_returns_volatility('OP_AP GMM Analysis 2015')


OP_AP_gmm_rf_2015.train_random_forests()


ORR_G_gmm_rf_2015=gmm_randomForests(ORR_G_historical_rets2,ORR_G_rets_test2,ORR_G_2_2015.portfolio['Returns'],5,
                                 ORR_G_historical2.df, ORR_G_2015.df,626,-160)
ORR_G_gmm_rf_2015.analyze_historical_regimes()
ORR_G_gmm_rf_2015.historical_regime_returns_volatility('ORR_G GMM Analysis 2015')


ORR_G_gmm_rf_2015.train_random_forests()


OP_AP_regime_predictions_2015=OP_AP_gmm_rf_2015.base_portfolio_predictions
ORR_G_regime_predictions_2015=ORR_G_gmm_rf_2015.base_portfolio_predictions





OP_AP_spo_2015=statarb_update(pd.DataFrame(OP_test2), pd.DataFrame(AP_test2),2,nmbr,-1,2,nmbr,notowania_test2.iloc[0],notowania_test2.iloc[-1],OP_AP_regime_predictions_2015,'Target',avoid1=0,target1=1,
                  exit_zscore=0)
ORR_G_spo_2015=statarb_update(pd.DataFrame(G_test2), pd.DataFrame(ORR_test2),2,nmbr,-1,2,nmbr,notowania_test2.iloc[0],notowania_test2.iloc[-1],ORR_G_regime_predictions_2015,'Target',avoid1=1,target1=0,
                  exit_zscore=0)

OP_AP_spo_2015.create_spread()
ORR_G_spo_2015.create_spread()
OP_AP_spo_2015.generate_signals()
ORR_G_spo_2015.generate_signals()

OP_AP_spo_2015.create_returns(OP_AP_allocation,'OP_AP SPO Framework')
ORR_G_spo_2015.create_returns(ORR_G_allocation, 'ORR_G SPO Framework')
spo_portfolio_2015=pd.DataFrame()
spo_portfolio_2015['OP_AP']=OP_AP_spo_2015.portfolio['Portfolio Value']
spo_portfolio_2015['ORR_G']= ORR_G_spo_2015.portfolio['Portfolio Value']
spo_portfolio_2015['G']=np.array(iloscJUg2*notowania_test2['G'])
spo_portfolio_2015['Total Portfolio Value']=spo_portfolio_2015['OP_AP']+ spo_portfolio_2015['ORR_G']+spo_portfolio_2015['G']
spo_portfolio_2015['Returns']=np.log(spo_portfolio_2015['Total Portfolio Value']/spo_portfolio_2015['Total Portfolio Value'].shift(1))


plt.figure(figsize=(10,6))
plt.plot(spo_portfolio_2015['Total Portfolio Value'])
plt.title('SPO Portfolio Equity Curve in 2015')
plt.show()
