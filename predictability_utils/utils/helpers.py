import numpy as np

def compute_anomaly_corrs(out_true, out_pred):
    
    anomaly_corrs = np.zeros(out_pred.shape[1])
    for i in range(anomaly_corrs.size):
        anomaly_corrs[i] = np.corrcoef(out_pred[:,i], out_true[:,i])[0,1]
        
    return anomaly_corrs

def split_train_data(train_months, test_months, train_years, test_years):

    def make_idx(months, years): # based on simple broadcasting
        return np.asarray(months).reshape(-1,1)+(12*np.asarray(years).flatten())

    idx_source_train = make_idx(train_months, train_years)
    idx_target_train = make_idx(test_months, train_years)

    idx_source_test = make_idx(train_months, test_years)
    idx_target_test = make_idx(test_months, test_years)

    return idx_source_train, idx_target_train, idx_source_test, idx_target_test