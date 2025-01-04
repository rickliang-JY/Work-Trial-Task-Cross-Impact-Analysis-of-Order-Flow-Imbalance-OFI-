import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings
warnings.filterwarnings("ignore")
class OFI_implement:
    def __init__(self, level: int, freq: str):
        self.level = level
        self.freq = freq
    def ofi_implement(data: pd.DataFrame, level: int, freq: str):
        ## compute OFI
        ask_sz = ['ask_sz_01', 'ask_sz_02', 'ask_sz_03', 'ask_sz_04', 'ask_sz_05']
        bid_sz = ['bid_sz_01', 'bid_sz_02', 'bid_sz_03', 'bid_sz_04', 'bid_sz_05']
        ask_px = ['ask_px_01', 'ask_px_02', 'ask_px_03', 'ask_px_04', 'ask_px_05']
        bid_px = ['bid_px_01', 'bid_px_02', 'bid_px_03', 'bid_px_04', 'bid_px_05']

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

        ## comupte last_size
        data['last_size'] = data['size'].diff().values
        data['last_size'][0] = data['size'][0]

        ## Integrated OFI, with PCA
        #freq = freq
        data['event_count'] = 1
        x = data.resample(freq
                            ).agg({ 'action': 'last', 'side': 'last', 'depth': 'last', 
                                    'price': 'last', 'midprice': 'last',
                                    'last_size': 'sum',

                                    'aOF1': 'sum', 'aOF2': 'sum', 'aOF3': 'sum', 'aOF4': 'sum', 'aOF5': 'sum',
                                    'bOF1': 'sum', 'bOF2': 'sum', 'bOF3': 'sum', 'bOF4': 'sum', 'bOF5': 'sum',


                                    'ask_sz_01': 'sum', 'ask_sz_02': 'sum', 'ask_sz_03': 'sum', 'ask_sz_04': 'sum', 'ask_sz_05': 'sum',
                                    'bid_sz_01': 'sum', 'bid_sz_02': 'sum', 'bid_sz_03': 'sum', 'bid_sz_04': 'sum', 'bid_sz_05': 'sum',

                                    'event_count': 'sum'
                                    }).rename(columns={'last_size': 'size'})

        # x = x.loc[~(x == 0).all(axis=1)]
        # x.dropna(inplace=True)

        x['returns'] = np.log(x['midprice']) - np.log(x['midprice'].shift(1))
        x['midprice_delta'] = x['midprice'] - x['midprice'].shift(1)


        x['OFI1'] = x['bOF1'] - x['aOF1']
        x['OFI2'] = x['bOF2'] - x['aOF2']
        x['OFI3'] = x['bOF3'] - x['aOF3']
        x['OFI4'] = x['bOF4'] - x['aOF4']
        x['OFI5'] = x['bOF5'] - x['aOF5']

        M = 5
        Q1 = (x['ask_sz_01'] + x['bid_sz_01']) / 2 / x['event_count']
        Q2 = (x['ask_sz_02'] + x['bid_sz_02']) / 2 / x['event_count']
        Q3 = (x['ask_sz_03'] + x['bid_sz_03']) / 2 / x['event_count']
        Q4 = (x['ask_sz_04'] + x['bid_sz_04']) / 2 / x['event_count']
        Q5 = (x['ask_sz_05'] + x['bid_sz_05']) / 2 / x['event_count']

        Qm = (Q1 + Q2 + Q3 + Q4 + Q5) / M
        x['ofi1'] = x['OFI1'] / Qm
        x['ofi2'] = x['OFI2'] / Qm
        x['ofi3'] = x['OFI3'] / Qm
        x['ofi4'] = x['OFI4'] / Qm
        x['ofi5'] = x['OFI5'] / Qm

        # x.dropna(inplace=True)
        x = x[~x.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        X = x[['ofi1', 'ofi2', 'ofi3', 'ofi4', 'ofi5']]
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X)
        ofiI = pd.Series(X_pca.flatten(), index=X.index)
        standard_scaler = StandardScaler()
        ofiI_standard = standard_scaler.fit_transform(np.array(ofiI).reshape(-1, 1))
        x['ofiI'] = pd.Series(ofiI_standard.flatten(), index=X.index)

        return x, X_pca