#Analiza wrażliwości


import statsmodels as statsmodels
makro = pd.read_csv('D:/Тимчасове/Projects/QI/Dane-makro.csv',sep=";", index_col=0, parse_dates=True)
makro=makro['2017-01-01':'2012-01-01']

for z in range(len(makro.columns)):
    danedograngera=pd.concat([spo_portfoliomret, makro.iloc[:,z]], axis=1)
    danedograngera.dropna(inplace=True)
    fig, ax1 = plt.subplots()
    x = danedograngera.index
    y1 = danedograngera.iloc[:,0]
    y2 = danedograngera.iloc[:,1]
    ax2 = ax1.twinx()
    ax1.title.set_text('Miesięczne stopy zwrotu portfela SPO a czynnnik makroekonomiczny')
    ax1.plot(x, y1, 'g-')
    ax2.plot(x, y2, 'b-')
    plt.legend(loc='upper left')
    statsmodels.tsa.stattools.grangercausalitytests(danedograngera,maxlag=12,verbose=False)
    z=z+1
    granger_sym=statsmodels.tsa.stattools.grangercausalitytests(danedograngera,maxlag=12,verbose=False)
plt.show()


granger_test_result =statsmodels.tsa.stattools. grangercausalitytests(danedograngera, maxlag=12,verbose=True)

for key in granger_test_result.keys():
    _F_test_ = granger_test_result[key][0]['params_ftest'][0]
    if _F_test_ > F_test:
        F_test = _F_test_
        optimal_lag = key

print(optimal_lag)