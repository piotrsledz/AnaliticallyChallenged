#Test wsteczny 2: skrócenie okresu testowania modelu
#Lata trenowania modelu 2000-2011
#Lata testowania modelu 2012-2015



notowania_tren4=notowaniaw['2000-01-01':'2010-12-31']
OP_tren4=np.array(notowania_tren4['OP'])
AP_tren4=np.array(notowania_tren4['AP'])
ORR_tren4=np.array(notowania_tren4['ORR'])
G_tren4=np.array(notowania_tren4['G'])

notowania_test4= notowaniaw['2012-01-01':'2015-12-31']
OP_test4=np.array(notowania_test4['OP'])
AP_test4=np.array(notowania_test4['AP'])
ORR_test4=np.array(notowania_test4['ORR'])
G_test4=np.array(notowania_test4['G'])

OP_AP_2012=statarb(OP_test4,AP_test4,nmbr,-1,2,nmbr,'2012-01-01','2015-12-31')
OP_AP_2012.create_spread()
OP_AP_2012.generate_signals()
OP_AP_2012.create_returns(10000,'OP_AP 2012')


ORR_G_2012=statarb(ORR_test4,ORR_test4,nmbr,-1,2,nmbr,'2012-01-01','2015-12-31')
ORR_G_2012.create_spread()
ORR_G_2012.generate_signals()
ORR_G_2012.create_returns(10000,'ORR_G 2012')

iloscJUg4 =10000/notowania_test4['G'][0]

equally_weighted_2012=pd.DataFrame()
equally_weighted_2012['OP_AP']=OP_AP_2012.portfolio['Portfolio Value']
equally_weighted_2012['ORR_G']=ORR_G_2012.portfolio['Portfolio Value']
equally_weighted_2012['G']=np.array(iloscJUg4*notowania_test4['G'])
equally_weighted_2012['Total Portfolio Value']=equally_weighted_2012['OP_AP']+equally_weighted_2012['ORR_G']+equally_weighted_2012['G']
equally_weighted_2012['Returns']=np.log(equally_weighted_2012['Total Portfolio Value']/equally_weighted_2012['Total Portfolio Value'].shift(1))

plt.figure(figsize=(10,6))
plt.plot(equally_weighted_2012['Total Portfolio Value'])
plt.title('Equally Weighted Portfolio Equity Curve in 2012')
plt.show()




OP_AP_allocation=round(total_allocation*optimal_weights[0][0],2)
ORR_G_allocation=round(total_allocation*optimal_weights[0][1],2)
OP_AP_2_2012=statarb(OP_test4,AP_test4,nmbr,-1,2,nmbr,'2012-01-01','2015-12-31')
OP_AP_2_2012.create_spread()
OP_AP_2_2012.generate_signals()
OP_AP_2_2012.create_returns(OP_AP_allocation,'OP_AP_Portfolio _2 in 2012')

ORR_G_2_2012=statarb(G_test4,ORR_test4,nmbr,-1,2,nmbr,'2012-01-01','2015-12-31')
ORR_G_2_2012.create_spread()
ORR_G_2_2012.generate_signals()
ORR_G_2_2012.create_returns(ORR_G_allocation,'ORR_G_Portfolio _2 in 2012')



efficient_frontier_portfolio_2012=pd.DataFrame()
efficient_frontier_portfolio_2012['OP_AP']=OP_AP_2_2012.portfolio['Portfolio Value']
efficient_frontier_portfolio_2012['ORR_G']=ORR_G_2_2012.portfolio['Portfolio Value']
efficient_frontier_portfolio_2012['G']=np.array(iloscJUg4*notowania_test4['G'])
efficient_frontier_portfolio_2012['Total Portfolio Value']=efficient_frontier_portfolio_2012['OP_AP']+efficient_frontier_portfolio_2012['ORR_G']+efficient_frontier_portfolio_2012['G']
efficient_frontier_portfolio_2012['Returns']=np.log(efficient_frontier_portfolio_2012['Total Portfolio Value']/efficient_frontier_portfolio_2012['Total Portfolio Value'].shift(1))

plt.figure(figsize=(10,6))
plt.plot(efficient_frontier_portfolio_2012['Total Portfolio Value'])
plt.title('Efficient Frontier Portfolio Equity Curve in 2012')
plt.show()



OP_AP_historical4=statarb(OP_tren4,AP_tren4,nmbr,-2,2,nmbr,notowania_tren4.iloc[0],notowania_tren4.iloc[-1])
OP_AP_historical4.create_spread()
OP_AP_historical4.generate_signals()
OP_AP_historical4.create_returns(10000,'OP_AP_ Over Training Period 2012')


OP_AP_historical_rets4=OP_AP_historical4.portfolio['Returns']


ORR_G_historical4=statarb(G_tren4,ORR_tren4,nmbr,-2,2,nmbr,notowania_tren4.iloc[0],notowania_tren4.iloc[-1])
ORR_G_historical4.create_spread()
ORR_G_historical4.generate_signals()
ORR_G_historical4.create_returns(10000,'ORR_G_ Over Hist. Train Period 2012')


ORR_G_historical_rets4=ORR_G_historical4.portfolio['Returns']

OP_AP_rets_len2=len(OP_AP_historical_rets4)
OP_AP_rets_train2=OP_AP_historical_rets4[0:465]
OP_AP_rets_test4=OP_AP_historical_rets4[-116:]

ORR_G_rets_len2=len(ORR_G_historical_rets4)
ORR_G_rets_train2=ORR_G_historical_rets4[0:465]
ORR_G_rets_test4=ORR_G_historical_rets4[-116:]




OP_AP_gmm_rf_2012=gmm_randomForests(OP_AP_historical_rets4,OP_AP_rets_test4,OP_AP_2_2012.portfolio['Returns'],5,
                                 OP_AP_historical4.df, OP_AP_2012.df,465,-116)
OP_AP_gmm_rf_2012.analyze_historical_regimes()
OP_AP_gmm_rf_2012.historical_regime_returns_volatility('OP_AP GMM Analysis 2012')


OP_AP_gmm_rf_2012.train_random_forests()


ORR_G_gmm_rf_2012=gmm_randomForests(ORR_G_historical_rets4,ORR_G_rets_test4,ORR_G_2_2012.portfolio['Returns'],5,
                                 ORR_G_historical4.df, ORR_G_2012.df,465,-116)
ORR_G_gmm_rf_2012.analyze_historical_regimes()
ORR_G_gmm_rf_2012.historical_regime_returns_volatility('ORR_G GMM Analysis 2012')


ORR_G_gmm_rf_2012.train_random_forests()


OP_AP_regime_predictions_2012=OP_AP_gmm_rf_2012.base_portfolio_predictions
ORR_G_regime_predictions_2012=ORR_G_gmm_rf_2012.base_portfolio_predictions





OP_AP_spo_2012=statarb_update(pd.DataFrame(OP_test4), pd.DataFrame(AP_test4),2,nmbr,-1,2,nmbr,notowania_test4.iloc[0],notowania_test4.iloc[-1],OP_AP_regime_predictions_2012,'Target',avoid1=0,target1=1,
                  exit_zscore=0)
ORR_G_spo_2012=statarb_update(pd.DataFrame(G_test4), pd.DataFrame(ORR_test4),2,nmbr,-1,2,nmbr,notowania_test4.iloc[0],notowania_test4.iloc[-1],ORR_G_regime_predictions_2012,'Target',avoid1=1,target1=0,
                  exit_zscore=0)

OP_AP_spo_2012.create_spread()
ORR_G_spo_2012.create_spread()
OP_AP_spo_2012.generate_signals()
ORR_G_spo_2012.generate_signals()

OP_AP_spo_2012.create_returns(OP_AP_allocation,'OP_AP SPO Framework')
ORR_G_spo_2012.create_returns(ORR_G_allocation, 'ORR_G SPO Framework')
spo_portfolio_2012=pd.DataFrame()
spo_portfolio_2012['OP_AP']=OP_AP_spo_2012.portfolio['Portfolio Value']
spo_portfolio_2012['ORR_G']= ORR_G_spo_2012.portfolio['Portfolio Value']
spo_portfolio_2012['G']=np.array(iloscJUg*notowania_test4['G'])
spo_portfolio_2012['Total Portfolio Value']=spo_portfolio_2012['OP_AP']+ spo_portfolio_2012['ORR_G']+spo_portfolio_2012['G']
spo_portfolio_2012['Returns']=np.log(spo_portfolio_2012['Total Portfolio Value']/spo_portfolio_2012['Total Portfolio Value'].shift(1))


plt.figure(figsize=(10,6))
plt.plot(spo_portfolio_2012['Total Portfolio Value'])
plt.title('SPO Portfolio Equity Curve in 2012')
plt.show()

