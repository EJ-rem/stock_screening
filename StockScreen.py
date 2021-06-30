def getPriceDF(tickers, start, end):
    
    # create price df from tickers
    price_df = pd.DataFrame({
        ticker: web.get_data_yahoo(ticker, start, end)['Adj Close']
        for ticker in tickers
    }).dropna()
    
    return price_df

# define function to load tickers and create df with prices
def EfficientFrontierMCS(price_df, iterations):
    
    # visualize efficient frontier
    # number of iterations for MCS
    iters = iterations
    
    # empty arrays to store values
    pred_ret = np.zeros(len(range(iters)))
    pred_vol = np.zeros(len(range(iters)))

    # Monte Carlo Simulation to generate sample portfolios
    for i in range(iterations):
        weights = np.random.random(len(price_df.columns))
        weights /= np.sum(weights)
        pred_ret[i] = (np.sum(price_df.pct_change().dropna().mean() * weights) * 252)
        pred_vol[i] = (np.sqrt(np.dot(weights.T, np.dot(price_df.pct_change().dropna().cov() * 252, weights))))

    # visualize
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    ax.scatter(pred_vol, pred_ret, c = pred_ret/pred_vol, marker='o')

    fig.tight_layout()

    plt.show()
    
def runOptimzer(price_df, gamma_MSR=0, gamma_GMV=0):

    # find mean of price data set
    mu = mean_historical_return(price_df)
    
    # shrink the covariance
    S = CovarianceShrinkage(price_df).ledoit_wolf()
    
    # max sharpe ratio portfolio (MSR)
    ef_MSR = EfficientFrontier(mu, S)
    ef_MSR.add_objective(L2_reg, gamma=gamma_MSR)
    raw_MSR = ef_MSR.max_sharpe() # optimal weights; raw data
    clean_MSR = ef_MSR.clean_weights() # clean weights for MSR
    
    #repeat for global minimum volatility portfolio (GMV)
    ef_GMV = EfficientFrontier(mu, S)
    ef_GMV.add_objective(L2_reg, gamma=gamma_GMV)
    raw_GMV = ef_GMV.min_volatility()
    clean_GMV = ef_GMV.clean_weights()
    
    # create dataframes for output
    MSR_df = pd.DataFrame.from_dict(clean_MSR, orient='index')   
    MSR_df = MSR_df.reset_index()
    MSR_df.columns = ['Ticker', 'MSR Weight']
    
    GMV_df = pd.DataFrame.from_dict(clean_GMV, orient='index')
    GMV_df = GMV_df.reset_index()
    GMV_df.columns = ['Ticker', 'GMV Weight']
    
    # portfolio performance summary
    print('MSR Portfolio')
    MSR_summary = ef_MSR.portfolio_performance(verbose=True)
    print('')
    print('GMV Portfolio')
    GMV_summary = ef_GMV.portfolio_performance(verbose=True)
    
    weights_df = MSR_df.merge(GMV_df, on='Ticker')
    
    return weights_df

def plotScreeningWeights(df):
    
    fig, ax = plt.subplots(2, figsize=(12,8))

    ax[0].set_title('Maximum Sharpe Ration Portfolio')
    ax[0].set_ylabel('Weight')
    ax[0].bar(df['Ticker'], df['MSR Weight'])

    ax[1].set_title('Global Minimum Volatility Portfolio')
    ax[1].set_ylabel('Weight')
    ax[1].bar(df['Ticker'], df['GMV Weight'], color='green')

    fig.tight_layout()
    plt.show()
    
# SML functions

def screenSML(tickers, start, end, rfr=0.02, index_comparison='^GSPC', visualize=False):

    # add index to list of tickers; will go to last position in list
    tickers = tickers + [index_comparison]
    
    # generate price list
    price_df = pd.DataFrame({
        ticker: web.get_data_yahoo(ticker, start, end)['Adj Close']
        for ticker in tickers
        }).dropna()
    
    # generate daily returns (dr_df)
    dr_df = price_df.pct_change().dropna()
    
    # X values for regression
    x_benchmark = dr_df.iloc[:, -1].values.reshape(-1,1)
    
    # generate market risk premium for this period
    MRP = (np.mean(x_benchmark) * 252) - rfr
    
    # create empty dictionary to stock beta values for each stock
    beta_dict = {}
    
    for tick in tickers[:-1]:
        y_stock = dr_df.loc[:, tick].values.reshape(-1,1)
        linear_model = LinearRegression().fit(x_benchmark, y_stock)
        beta_dict[tick] = linear_model.coef_.reshape(-1)
    
    # create dataframe to store beta and other data
    df_SML = pd.DataFrame.from_dict(beta_dict, orient='index')
    df_SML.columns = ['Beta']
    df_SML['Expected Return'] = rfr + (df_SML['Beta'] * MRP)
    df_SML['Observed Return'] = (dr_df.iloc[:, :-1].mean() * 252).values.reshape(-1,1)
    
    if visualize==True:
    
        # visualization component
        # lower and upper bounds of SML
        beta_min = 1.2 * min(df_SML['Beta'])
        beta_max = 1.2 * max(df_SML['Beta'])

        # generate SML
        SML_x = np.linspace(beta_min, beta_max, 10)
        SML_y = rfr + (SML_x * MRP)
        
        #visualize
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_title('Security Market Line ESPO Holdings')
        ax.set_xlabel('Beta')
        ax.set_ylabel('Return')
        ax.plot(SML_x, SML_y, color='blue', label='Securiity Market Line') # SML
        ax.scatter(df_SML['Beta'], df_SML['Observed Return'], color='green', label='Observed Returns') #Observed returns of each stock
        
        # labels
        
        for label, x, y in zip(df_SML.index, df_SML.loc[:, 'Beta'], df_SML.loc[:, 'Observed Return']):
            ax.annotate(label, (x,y), textcoords='offset points', xytext=(0,10))
        
        ax.legend()
        fig.tight_layout()
        plt.show()
        
    elif visualize==False:
        None
    
    else:
        None
    
    return df_SML