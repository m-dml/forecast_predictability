from netCDF4 import Dataset
import numpy as np

from predictability_utils.utils import helpers, io
from predictability_utils.methods.lrlin_method import run_lrlin
from predictability_utils.methods.cca_method import run_cca

import sys

"""
runs CCA-method and LR-rank linear method for given number of latent components n
to predict T2ms from source data X (e.g. T2ms, SSTs, MSLs etc.)
"""

root_data = '../../data/pyrina'
root_results = '../../results/pyrina'


##

# change analysis parameters here
y_train = 51 # training/test data split (y_train=51 for 1900-1951 train, 1951-2010 test)
train_months, test_months = [2,3,4], [5,6,7] # months to predict from, months to predict
n_latents = np.int(sys.argv[1]) # 5 # number of latent components (CCs, rank of linear prediction matrix)
field, region, preprocess = sys.argv[2], sys.argv[3], sys.argv[4] # which source data to use
lr, n_epochs = np.float(sys.argv[5]), 20000 # for gradient descent used in low-rank linear method 

##

m_train = ''.join([str(i) for i in train_months]) # strings for save-file
m_test = ''.join([str(i) for i in test_months])   # identification

# source data to predict T2s from
source_data, _ = io.data_load(field, region, preprocess, root_data, verbose=False)

# Temperature at 2m (EU) ANOMALIES
target_data, _ = io.data_load('t2m', 'EU', 'anomalies', root_data, verbose=False)

# training data time stamps and map shape
nc_fn = root_data + "/t2m_ERA20c_monthly_1900-2010.EU.mv.nc"
ts = Dataset(nc_fn, 'r').variables['time'].__array__().data

idcs = helpers.split_train_data(ts, y_train, train_months, test_months)
idx_source_train, idx_target_train, idx_source_test, idx_target_test = idcs


# CCA analysis
anomaly_corrs, params = run_cca(source_data, target_data, n_latents, idcs, if_plot=False)
sv_fn = f'/{field}_ERA20c_monthly_1900-2010_{region}_{preprocess}__s{m_train}_t{m_test}_split{y_train}__n{n_latents}_CCA'
np.save(root_results + sv_fn, anomaly_corrs)

# LR-linear analysis
anomaly_corrs, params = run_lrlin(source_data, target_data, n_latents, idcs, if_plot=False, lr=lr, n_epochs=n_epochs)
sv_fn = f'/{field}_ERA20c_monthly_1900-2010_{region}_{preprocess}__s{m_train}_t{m_test}_split{y_train}__n{n_latents}_LRL'
np.save(root_results + sv_fn, anomaly_corrs)
