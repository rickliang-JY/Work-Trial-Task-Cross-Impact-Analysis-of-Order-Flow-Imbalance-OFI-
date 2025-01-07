#%%Import libraries
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from ofi import OFI_implement
import warnings
warnings.filterwarnings("ignore") 

#%% Function
## Price impact of best-level OFIs
def PI(data, object, window_length='30min'):
    #object = OFI1/ ofiI
    regression_results = []
    # get the data for each window
    grouped_data = data.resample(window_length)
    # run regression for each window
    for window_start, window_data in grouped_data:
        
        if len(window_data) < 2:
            continue
        
        X = window_data[f'{object}']
        y = window_data['returns']
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        # store the regression results
        regression_results.append({
            'window_start': window_start,
            'window_end': window_start + pd.Timedelta(window_length),
            'params': model.params,            # model parameters
            'r_squared': model.rsquared,       # R^2 value
            'p_values': model.pvalues,         # Pvalues
            'summary': model.summary().as_text()  # regression summary
        })

    results_df = pd.DataFrame(regression_results)
    return results_df


#Price impact of integrated OFIs
# def PII(data, window_length='30min'):
#     regression_results = []
#     # get the data for each window
#     grouped_data = data.resample(window_length)
#     # run regression for each window
#     for window_start, window_data in grouped_data:
        
#         if len(window_data) < 2:
#             continue
        
#         X = window_data['ofiI']
#         y = window_data['returns']
#         X = sm.add_constant(X)

#         model = sm.OLS(y, X).fit()

#         # store the regression results
#         regression_results.append({
#             'window_start': window_start,
#             'window_end': window_start + pd.Timedelta(window_length),
#             'params': model.params,            # model parameters
#             'r_squared': model.rsquared,       # R^2 value
#             'p_values': model.pvalues,         # Pvalues
#             'summary': model.summary().as_text()  # regression summary
#         })

#     results_df = pd.DataFrame(regression_results)
#     return results_df


 ##Cross-impact of best-level OFI

def CI(data, object, stock_list,stock_name, window_length='30min'):
    #object = 'OFI'/ 'ofiI'
    data = data[['returns', 'OFI1','symbol']]
    data = data.set_index('symbol', append = True).unstack(level='symbol')
    #since the number of event in one 
    
    #data.dropna(inplace=True)
    #print(data.isnull().sum())
    data = data.fillna(0)
    regression_results = []
    # get the data for each window
    grouped_data = data.resample(window_length)
    # run regression for each window
    for window_start, window_data in grouped_data:
        
        if len(window_data) < 2:
            continue
        #window_data = window_data.set_index('symbol', append = True).unstack(level='symbol')
        return_col = [f'returns_{i}' for i in stock_list]
        ofi_col = [f'{object}_{i}' for i in stock_list]
        window_data.columns = return_col + ofi_col
        ofi_col_copy = ofi_col.copy()
        ofi_col_copy.remove(f'{object}_{stock_name}')

        y = window_data[f'returns_{stock_name}']
        X = window_data[ofi_col_copy]
        
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        # store the regression results
        regression_results.append({
            'window_start': window_start,
            'window_end': window_start + pd.Timedelta(window_length),
            'params': model.params,            # model parameters
            'r_squared': model.rsquared,       # R^2 value
            'p_values': model.pvalues,         # Pvalues
            'summary': model.summary().as_text()  # regression summary
        })

    results_df = pd.DataFrame(regression_results)
    return results_df

 ##Cross-impact of integrated OFIs

# def CII(data, stock_list,stock_name, window_length='30min'):
    # data = data[['returns', 'OFI1','symbol']]
    # data = data.set_index('symbol', append = True).unstack(level='symbol')
    # #since the number of event in one 
    # #print(data.isnull().sum())
    # data = data.fillna(0)
    # regression_results = []
    # # get the data for each window
    # grouped_data = data.resample(window_length)
    # # run regression for each window
    # for window_start, window_data in grouped_data:
        
    #     if len(window_data) < 2:
    #         continue
    #     #window_data = window_data.set_index('symbol', append = True).unstack(level='symbol')
    #     return_col = [f'returns_{i}' for i in stock_list]
    #     ofiI_col = [f'ofiI_{i}' for i in stock_list]
    #     window_data.columns = return_col + ofiI_col
    #     ofiI_col_copy = ofiI_col.copy()
    #     ofiI_col_copy.remove(f'ofiI_{stock_name}')

    #     y = window_data[f'returns_{stock_name}']
    #     X = window_data[ofiI_col_copy]
        
    #     X = sm.add_constant(X)

    #     model = sm.OLS(y, X).fit()

    #     # store the regression results
    #     regression_results.append({
    #         'window_start': window_start,
    #         'window_end': window_start + pd.Timedelta(window_length),
    #         'params': model.params,            # model parameters
    #         'r_squared': model.rsquared,       # R^2 value
    #         'p_values': model.pvalues,         # Pvalues
    #         'summary': model.summary().as_text()  # regression summary
    #     })

    # results_df = pd.DataFrame(regression_results)
    # return results_df






#%%Load data
stock_list = ["AAPL","MSFT","NVDA","AMGN","GILD","TSLA","PEP","JPM", "V", "XOM"]
datapool = {}

# Loop through each stock's data and store it in the data pool
for stock_name in stock_list:
    file_path = f'../data/{stock_name}20250102_after.csv'  # file path
    data = pd.read_csv(file_path, index_col=0)            
    data.index = pd.to_datetime(data.index)              
    datapool[stock_name] = data

# Get the OFI data for each stock
for stock_name in stock_list:
    data, X_pac = OFI_implement.ofi_implement(datapool[stock_name], 5, '1min')
    datapool[stock_name] = data      


#%% PI and PII
PI_result = {}
PII_result = {}

for stock_name in stock_list:
  result1 = PI(datapool[stock_name], 'OFI1',window_length='30min')
  result2 = PI(datapool[stock_name], 'ofiI',window_length='30min')
  PI_result[stock_name] = result1
  PII_result[stock_name] = result2

#%% CI and CII
combined_data = []
for stock_name, data in datapool.items():
    data['symbol'] = stock_name
    combined_data.append(data)
combined_data = pd.concat(combined_data, axis=0)

CI_result = {}
for stock_name in stock_list:
  result = CI(combined_data, 'OFI', stock_list, stock_name, window_length='30min')
  CI_result[stock_name] = result

CII_result = {}
for stock_name in stock_list:
  result = CI(combined_data, 'ofiI',stock_list, stock_name, window_length='30min')
  CII_result[stock_name] = result
# %%
