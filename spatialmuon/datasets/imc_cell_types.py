##
import pandas as pd
import os
import numpy as np
import spatialmuon as smu

##
f = '/data/spatialmuon/datasets/imc/raw/Data_publication'
f0 = os.path.join(f, 'BaselTMA/SC_dat.csv')
f1 = os.path.join(f, 'ZurichTMA/PG_zurich.csv')

##
df0 = pd.read_csv(f0)
##
df1 = pd.read_csv(f1)
print('done')
##
df0.shape
df1.shape