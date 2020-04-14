import numpy as np

def compute_anomaly_corrs(out_true, out_pred):
    
    anomaly_corrs = np.zeros(out_pred.shape[1])
    for i in range(anomaly_corrs.size):
        anomaly_corrs[i] = np.corrcoef(out_pred[:,i], out_true[:,i])[0,1]
        
    return anomaly_corrs

def split_train_data(ts, y_train, train_months, test_months):
    y_total = len(ts)//12
    m_train = y_train * 12
    
    mm_train = len(ts) - (40*12) #choose the last 40 years

    idx_target_train = np.zeros((len(train_months), y_train), dtype=np.int)
    idx_source_train = np.zeros((len(train_months), y_train), dtype=np.int)
    idx_target_test = np.zeros((len(test_months), y_test), dtype=np.int)
    idx_source_test = np.zeros((len(test_months), y_test), dtype=np.int)
    for i,m in enumerate(train_months):
        idx_source_train[i,:] = np.arange(m, m_train, 12)
        idx_source_test[i,:]  = np.arange(mm_train+m,  len(ts), 12)
    for i,m in enumerate(test_months):
        idx_target_train[i,:] = np.arange(m, m_train, 12)
        idx_target_test[i,:]  = np.arange(mm_train+m,  len(ts), 12)

    return idx_source_train, idx_target_train, idx_source_test, idx_target_test
