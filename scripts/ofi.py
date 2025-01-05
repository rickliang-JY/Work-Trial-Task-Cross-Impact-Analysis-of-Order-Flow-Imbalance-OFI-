import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class OFI_implement:
    def __init__(self, level: int, freq: str):
        self.level = level
        self.freq = freq
    def ofi_implement(data: pd.DataFrame, level: int, freq: str):
        ## compute OFI
        
        ask_sz = [f'ask_sz_{i:02}' for i in range(1, level+1)]
        bid_sz = [f'bid_sz_{i:02}' for i in range(1, level+1)]
        ask_px = [f'ask_px_{i:02}' for i in range(1, level+1)]
        bid_px = [f'bid_px_{i:02}' for i in range(1, level+1)]

        # Create columns to store bOF and aOF
        for i in range(1, level + 1):
            data[f'bOF{i}'] = 0
            data[f'aOF{i}'] = 0

        # Calculate the bid and ask price difference matrix
        bid_px_diff = data[bid_px].diff().values  # bid_price differ value
        ask_px_diff = data[ask_px].diff().values  # ask_price differ value

        bid_sz_matrix = data[bid_sz].values
        ask_sz_matrix = data[ask_sz].values
        bid_sz_matrix_prev = np.roll(bid_sz_matrix, 1, axis=0)  # Previous values of the sliding window
        ask_sz_matrix_prev = np.roll(ask_sz_matrix, 1, axis=0)

        # Initialize bOF and aOF matrices
        bOF_matrix = np.zeros_like(bid_px_diff)
        aOF_matrix = np.zeros_like(ask_px_diff)

        # Calculate bOF (Order Flow Imbalance for buy orders)
        bOF_matrix[bid_px_diff > 0] = bid_sz_matrix[bid_px_diff > 0]  # bid price up
        bOF_matrix[bid_px_diff == 0] = bid_sz_matrix[bid_px_diff == 0] - bid_sz_matrix_prev[bid_px_diff == 0]  # bid pricve no change
        bOF_matrix[bid_px_diff < 0] = -bid_sz_matrix_prev[bid_px_diff < 0]  # bid price down

        # Calculate aOF (Order Flow Imbalance for sell orders)
        aOF_matrix[ask_px_diff > 0] = -ask_sz_matrix_prev[ask_px_diff > 0]  # ask price up
        aOF_matrix[ask_px_diff == 0] = ask_sz_matrix[ask_px_diff == 0] - ask_sz_matrix_prev[ask_px_diff == 0]  # ask price no change
        aOF_matrix[ask_px_diff < 0] = ask_sz_matrix[ask_px_diff < 0]  # ask price down

        # Update the DataFrame with the bOF and aOF matrix results
        for j in range(level):
            data[f'bOF{j + 1}'] = bOF_matrix[:, j]
            data[f'aOF{j + 1}'] = aOF_matrix[:, j]

        ## compute misprice
        data['midprice'] = (data['ask_px_01'] + data['bid_px_01']) / 2

        ## comupte size change in each time, for later agg operation, th sum of last_size before time t sould be equal to the size value at time t
        data['last_size'] = data['size'].diff().values
        data['last_size'][0] = data['size'][0]

        ## Integrated OFI, with PCA
        #freq = freq
        data['event_num'] = 1
        data_th = data.resample(freq).agg(
            {'action': 'last', 
             'side': 'last', 
             'depth': 'last', 
             'price': 'last', 
             'midprice': 'last',
             'last_size': 'sum',
             'aOF1': 'sum', 'aOF2': 'sum', 'aOF3': 'sum', 'aOF4': 'sum', 'aOF5': 'sum',
             'bOF1': 'sum', 'bOF2': 'sum', 'bOF3': 'sum', 'bOF4': 'sum', 'bOF5': 'sum',

            'ask_sz_01': 'sum', 'ask_sz_02': 'sum', 'ask_sz_03': 'sum', 'ask_sz_04': 'sum', 'ask_sz_05': 'sum',
            'bid_sz_01': 'sum', 'bid_sz_02': 'sum', 'bid_sz_03': 'sum', 'bid_sz_04': 'sum', 'bid_sz_05': 'sum',

            'event_num': 'sum', 'symbol': 'last', 'midprice': 'last'
                                    }).rename(columns={'last_size': 'size'})

        # x = x.loc[~(x == 0).all(axis=1)]
        data_th.dropna(inplace=True)
        data_th['returns'] = np.log(data_th['midprice']) - np.log(data_th['midprice'].shift(1))
        data_th['midprice_delta'] = data_th['midprice'] - data_th['midprice'].shift(1)

        ## Best-level OFI
        for i in range(1, level + 1): data_th[f'OFI{i}'] = data_th[f'bOF{i}'] - data_th[f'aOF{i}']

        ## Deeper-level OFI
        M = level
        Q = []
        for i in range(1, level+1): Q.append((data_th[f'ask_sz_{i:02}'] + data_th[f'bid_sz_{i:02}']) / 2 / data_th['event_num'])
        QM = sum(Q) / M
        for i in range(1, level + 1): data_th[f'ofi{i}'] = data_th[f'OFI{i}'] / QM

        ## Integrated OFI
        # data_th = data_th.replace([np.inf, -np.inf], np.nan).dropna()
        # X = data_th[[f'ofi{i}' for i in range(1, level+1)]]
        # pca = PCA(n_components=1)
        # X_pca = pca.fit_transform(X)
        # ofiI = pd.Series(X_pca.flatten(), index=X.index)
        # standard_scaler = StandardScaler()
        # ofiI_standard = standard_scaler.fit_transform(np.array(ofiI).reshape(-1, 1))
        # data_th['ofiI'] = pd.Series(ofiI_standard.flatten(), index=X.index)

        ## Integrated OFI
        data_th = data_th.replace([np.inf, -np.inf], np.nan).dropna()
        X = data_th[[f'ofi{i}' for i in range(1, level + 1)]]
        X_standardized = (X - X.mean()) / X.std() 
        pca = PCA(n_components=1)  
        X_pca = pca.fit_transform(X_standardized)  
        w1 = pca.components_[0]  
        w1_normalized = w1 / np.sum(np.abs(w1))  
        ofiI = np.dot(X, w1_normalized)  
        standard_scaler = StandardScaler()
        ofiI_standard = standard_scaler.fit_transform(np.array(ofiI).reshape(-1, 1))
        data_th['ofiI'] = pd.Series(ofiI_standard.flatten(), index=X.index)

        return data_th, X_pca