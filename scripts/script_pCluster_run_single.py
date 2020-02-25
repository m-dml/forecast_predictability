from netCDF4 import Dataset
import numpy as np

from predictability_utils.utils import helpers, io
from predictability_utils.methods.lrlin_method import run_lrlin
from predictability_utils.methods.cca_method import run_cca

root_data = '../../data/pyrina'
root_results = '../../results/pyrina'


##

# change analysis parameters here
n_latents = 5
train_months, test_months = [2,3,4], [5,6,7]
field, region, preprocess = 'swvl1', 'EU', 'anomalies'
y_train = 51

##


m_train = ''.join([str(i) for i in train_months])
m_test = ''.join([str(i) for i in test_months])

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
anomaly_corrs, params = run_lrlin(source_data, target_data, n_latents, idcs, if_plot=False)
sv_fn = f'/{field}_ERA20c_monthly_1900-2010_{region}_{preprocess}__s{m_train}_t{m_test}_split{y_train}__n{n_latents}_LRL'
np.save(root_results + sv_fn, anomaly_corrs)
