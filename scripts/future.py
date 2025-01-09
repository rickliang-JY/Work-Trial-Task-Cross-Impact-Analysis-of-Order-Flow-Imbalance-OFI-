#%% import library
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from ofi import OFI_implement
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

#%% Function
def FPI(data, object, forecast_horizon, lags):
    # object = OFI1 or ofiI
    # Rolling window setup
    train_window = 30  # Length of the training set: 30 minutes
    test_window = forecast_horizon  # Test set length equals the future horizon
    step = 1  # Rolling step size: 1 minute

    # Initialize storage for results
    time_ranges = []
    actuals = []
    predictions = []
    r2_out_values = []

    # Rolling window loop
    for start in range(0, len(data) - train_window - test_window, step):
        # Define the range for training and testing
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = train_end + test_window

        # Split into training and testing sets
        train = data.iloc[train_start:train_end]
        test = data.iloc[test_start:test_end]

        # Define features and target variables
        X_train = train[[f'{object}_lag_{lag}' for lag in lags]]
        y_train = train[f'freturns_{forecast_horizon}']
        X_test = test[[f'{object}_lag_{lag}' for lag in lags]]
        y_test = test[f'freturns_{forecast_horizon}']

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate out-of-sample R^2
        historical_mean = np.mean(train[f'freturns_{forecast_horizon}'])  # Baseline model: historical mean
        r2_out = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - historical_mean) ** 2)

        # Store results
        time_ranges.append({
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1]
        })
        actuals.append(list(y_test.values))  # Store as a list for each test window
        predictions.append(list(y_pred))  # Store as a list for each test window
        r2_out_values.append(r2_out)

    # Flatten actuals and predictions for evaluation
    actuals_flat = [val for sublist in actuals for val in sublist]
    predictions_flat = [val for sublist in predictions for val in sublist]

    # Convert results into a DataFrame
    results = pd.DataFrame({
        'Train_Start': [time_range['train_start'] for time_range in time_ranges],
        'Train_End': [time_range['train_end'] for time_range in time_ranges],
        'Test_Start': [time_range['test_start'] for time_range in time_ranges],
        'Test_End': [time_range['test_end'] for time_range in time_ranges],
        'Out_of_Sample_R2': r2_out_values
    })

    # Add actuals and predictions (as lists) for each test window
    results['Actual'] = actuals
    results['Predicted'] = predictions

    # Print the first 10 rows of results
    return results




def FCI(inputdata, stock_list, object, train_window = 30, test_window = 5, step_window = 1, target_stock = 'AAPL'): 
    # object = OFI1 or ofiI
    data = inputdata
    # Set MultiIndex on 'ts_event' and 'symbol', and sort it
    data = data.set_index('symbol', append = True).sort_index()
    #data = data.set_index(['ts_event', 'symbol']).sort_index()
    # print(data.head(10))
    lags = [1,2,3,5,10,20,30]
    # for lag in lags:
    #     data[f'OFI1_lag_{lag}'] = data.groupby('symbol')['OFI1'].shift(lag)
    # data.fillna(0, inplace=True)
    #data.dropna(inplace=True)
    #lags = [1,2,3,5]
    # Rolling window setup

    train_window = train_window  # Training set size (number of minutes)
    test_window = test_window  # Test set size (number of minutes)
    step = step_window  # Rolling step size (1 minute)

    # Initialize storage for results
    time_ranges = []
    actuals = []
    predictions = []
    r2_out_values = []
    models = []
    # Rolling window loop
    for start in range(0, len(data.index.levels[0]) - train_window - test_window, step):
        # Define the training and testing time ranges
        train_start = data.index.levels[0][start]
        train_end = data.index.levels[0][start + train_window]
        test_start = data.index.levels[0][start + train_window]
        test_end = data.index.levels[0][start + train_window + test_window]

        # Select training and testing sets
        train = data.loc[train_start:train_end]
        test = data.loc[test_start:test_end]

        # Define the target stock (e.g., 'AAPL')
        target_stock = 'AAPL'

        # Check if the target stock is in both the training and testing sets
        if target_stock not in train.index.get_level_values('symbol') or target_stock not in test.index.get_level_values('symbol'):
            print(f"Target stock '{target_stock}' not found in train or test set. Skipping this window.")
            continue

        # Dependent variable: Future returns of the target stock
        # y_train = train.xs(target_stock, level='symbol')['Future_Returns']
        # y_test = test.xs(target_stock, level='symbol')['Future_Returns']
        y_train = train.xs(target_stock, level='symbol')[f'freturns_{test_window}']
        y_test = test.xs(target_stock, level='symbol')[f'freturns_{test_window}']

        # Independent variables: Lagged OFI values of all stocks
        # Include both AAPL and MSFT lagged features explicitly
        X_train = train[[f'{object}_lag_{lag}' for lag in lags]].xs(target_stock, level='symbol')
        X_test = test[[f'{object}_lag_{lag}' for lag in lags]].xs(target_stock, level='symbol')

        # Add cross-stock lags (e.g., MSFT lags for predicting AAPL)
        # other_stock = 'MSFT' if target_stock == 'AAPL' else 'AAPL'
        # for lag in lags:
        #     X_train[f'{other_stock}_OFI_lag_{lag}'] = train[[f'OFI_lag_{lag}']].xs(other_stock, level='symbol')
        #     X_test[f'{other_stock}_OFI_lag_{lag}'] = test[[f'OFI_lag_{lag}']].xs(other_stock, level='symbol')


         # 获取其他股票列表（除去 target_stock）
        other_stocks = [stock for stock in stock_list if stock != target_stock]

        # 添加其他股票的滞后特征
        for other_stock in other_stocks:
            if other_stock in train.index.get_level_values('symbol'):
                for lag in lags:
                    X_train[f'{other_stock}_{object}_lag_{lag}'] = train[[f'{object}_lag_{lag}']].xs(other_stock, level='symbol')
            if other_stock in test.index.get_level_values('symbol'):
                for lag in lags:
                    X_test[f'{other_stock}_{object}_lag_{lag}'] = test[[f'{object}_lag_{lag}']].xs(other_stock, level='symbol')
        


        # Handle missing values
        X_train = X_train.dropna()  # Drop rows with NaN in X_train
        y_train = y_train.loc[X_train.index]  # Align y_train with X_train

        X_test = X_test.dropna()  # Drop rows with NaN in X_test
        y_test = y_test.loc[X_test.index]  # Align y_test with X_test

        # X_train = sm.add_constant(X_train)
        # X_test = sm.add_constant(X_test)


        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        
        # Train the model
        if len(X_train) == 0 or len(X_test) == 0:  # Ensure no empty datasets
            print(f"Empty train or test set after handling NaNs. Skipping this window.")
            continue
        #print(y_train)

        model = LinearRegression()
        model.fit(X_train, y_train)
        # model = sm.OLS(y_train, X_train).fit()
        #print(model.summary())
        # Predict future returns on the test set
        if X_train.shape[1] != X_test.shape[1]: break
        y_pred = model.predict(X_test)

        # Out-of-sample R^2 calculation
        historical_mean = np.mean(y_train)  # Baseline: Historical mean
        r2_out = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - historical_mean) ** 2)

        # Store time ranges
        time_ranges.append({
            "train_start": train.index.get_level_values('ts_event')[0],
            "train_end": train.index.get_level_values('ts_event')[-1],
            "test_start": test.index.get_level_values('ts_event')[0],
            "test_end": test.index.get_level_values('ts_event')[-1]
        })

        # Store actual and predicted values
        actuals.append(list(y_test.values))
        predictions.append(list(y_pred))
        r2_out_values.append(r2_out)
        models.append(model)
    # Convert results to a DataFrame
    results = pd.DataFrame({
        'Train_Start': [time_range['train_start'] for time_range in time_ranges],
        'Train_End': [time_range['train_end'] for time_range in time_ranges],
        'Test_Start': [time_range['test_start'] for time_range in time_ranges],
        'Test_End': [time_range['test_end'] for time_range in time_ranges],
        # 'model_summary': [model.summary() for model in models],
        'Out_of_Sample_R2': r2_out_values,
        'Target_Stock': target_stock
    })

    # Add actuals and predictions (as lists)
    results['Actual'] = actuals
    results['Predicted'] = predictions

    # Display results
    return results



#%% Data Preparation
time_range = ['20241230', '20241231', '20250102', '20250103', '20250106', '20250107']
stock_list = ['JPM', 'TSLA', 'XOM', 'AMGN', 'AAPL']


alldata = {}

for stock in stock_list:
    combined_data = pd.DataFrame()  
    for date in time_range:
        
        file_name = f"../data/{stock}{date}.csv"
        try:
           
            chunks = pd.read_csv(file_name, chunksize=100000)
            for chunk in chunks:
                
                chunk = chunk.drop(columns=['ts_recv', 'rtype', 'publisher_id', 'instrument_id', 'flags'], errors='ignore')
                
                chunk = chunk.drop(columns=[col for col in chunk.columns 
                                             if any(char.isdigit() for char in col) 
                                             and (int(''.join(filter(str.isdigit, col))) > 5 
                                                  or int(''.join(filter(str.isdigit, col))) == 0)], errors='ignore')
                
                chunk.fillna(0, inplace=True)
                chunk.set_index('ts_event', inplace=True)  
                print(chunk.head())
                chunk.index = pd.to_datetime(chunk.index) 
                chunk, X_pac = OFI_implement.ofi_implement(chunk, 5, '1min')
                
                
                combined_data = pd.concat([combined_data, chunk])
        except Exception as e:
            print(f"无法读取文件 {file_name}: {e}")
    
    
    alldata[stock] = combined_data
    print(combined_data.head(5))

datapool = alldata.copy()


Lags = [1,2,3,5,10,20,30]
forecast_horizon = 5
for stock_name in stock_list:
    datapool[stock_name][f'freturns_{forecast_horizon}'] = (
    np.log(datapool[stock_name]['midprice'].shift(-forecast_horizon)) - np.log(datapool[stock_name]['midprice'])
)
    # define lag OFI
    L = Lags
    for l in L:
        datapool[stock_name][f'OFI1_lag_{l}'] = datapool[stock_name]['OFI1'].shift(l)  
        datapool[stock_name][f'ofiI_lag_{l}'] = datapool[stock_name]['ofiI'].shift(l) 
    datapool[stock_name] = datapool[stock_name].dropna()

combined_data = []

# 遍历每只股票的数据
for stock_name, data in datapool.items():
    # 添加一个标识符列，用于区分不同股票
    data['symbol'] = stock_name
    # 将数据加入列表
    combined_data.append(data)

# 合并所有数据为一个 DataFrame
combined_data = pd.concat(combined_data, axis=0)
combined_data.head(10)


#%% FPI, and FPII

FPI_result = {}
for stock_name in stock_list:
    result = FPI(datapool[stock_name], 'OFI1',forecast_horizon, lags=L)
    FPI_result[stock_name] = result

FPII_result = {}
for stock_name in stock_list:
    result = FPI(datapool[stock_name], 'ofiI', forecast_horizon, lags=L)
    FPII_result[stock_name] = result
FPII_result['AAPL'].head(10)



#%% FCI, and FCII
FCI_result = {}
for stock_name in stock_list:
    result = FCI(combined_data, stock_list, 'OFI1',train_window = 30, test_window = forecast_horizon, step_window = 1, target_stock = 'AAPL')
    FCI_result[stock_name] = result
FCII_result = {}
for stock_name in stock_list:
    result = FCI(combined_data, stock_list, 'ofiI',train_window = 30, test_window = forecast_horizon, step_window = 1, target_stock = 'AAPL')
    FCII_result[stock_name] = result


