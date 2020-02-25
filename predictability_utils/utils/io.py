import numpy as np
from netCDF4 import Dataset

def data_load(field, region, preprocess, root_data, verbose=True):
    
    assert field in ['t2m', 'sst', 'swvl1', 'msl']
    assert region in ['NA-EU', 'EU', 'TNA']
    assert preprocess in ['mv', 'anomalies']
    
    fn = f"/{field}_ERA20c_monthly_1900-2010.{region}.{preprocess}.nc"
    tmp =  Dataset(root_data + fn, 'r').variables[field].__array__()

    data, mask = tmp.data, tmp.mask
    if verbose: 
        print(field+'.shape', data.shape)
    
    if isinstance(mask, np.bool_):
        assert not mask
    else:
        mask = np.unique(mask, axis=0)
        assert mask.shape[0] == 1 # make sure mask is time-invariant

    return data, mask