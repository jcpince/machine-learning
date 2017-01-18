import os
import math
import numpy as np
import pandas as pd
from scipy.stats import skew

script_name = os.path.realpath(__file__)
base_dir = script_name[:script_name.index('/capstone/')]
input_dir = base_dir + '/capstone/input/'
output_dir = base_dir + '/capstone/output/'


# Load the training and submission datasets
traing_ds = pd.read_csv(input_dir + 'train.csv').set_index('Id')
submit_ds = pd.read_csv(input_dir + 'test.csv').set_index('Id')
submit_ds['SalePrice'] = np.zeros(submit_ds.shape[0],)
max_training_id = traing_ds.index.max()

print('traing_ds.shape', traing_ds.shape)
print('submit_ds.shape', submit_ds.shape)
whole_ds = pd.concat([traing_ds, submit_ds])
print('whole_ds.shape', whole_ds.shape)

# Per feature logical filling
whole_ds.Alley.fillna('No alley access', inplace=True)
na_idx = whole_ds[(whole_ds.MasVnrType.isnull() &
	whole_ds.MasVnrArea.isnull())].index
whole_ds.ix[na_idx, 'MasVnrType'] = 'No veneer masonry'
whole_ds.ix[na_idx, 'MasVnrArea'] = 0
na_idx = whole_ds[(whole_ds.BsmtQual.isnull() &
	whole_ds.BsmtCond.isnull() & whole_ds.BsmtExposure.isnull() &
	whole_ds.BsmtFinType1.isnull())].index
whole_ds.ix[na_idx, 'BsmtQual'] = 'No bsmt'
whole_ds.ix[na_idx, 'BsmtCond'] = 'No bsmt'
whole_ds.ix[na_idx, 'BsmtExposure'] = 'No bsmt'
whole_ds.ix[na_idx, 'BsmtFinType1'] = 'No bsmt'
whole_ds.ix[na_idx, 'BsmtFinSF1'] = 0
whole_ds.ix[na_idx, 'BsmtFinType2'] = 'No bsmt'
whole_ds.ix[na_idx, 'BsmtFinSF2'] = 0
whole_ds.ix[na_idx, 'BsmtUnfSF'] = 0
whole_ds.ix[na_idx, 'BsmtFullBath'] = 0
whole_ds.ix[na_idx, 'BsmtHalfBath'] = 0
whole_ds.ix[na_idx, 'TotalBsmtSF'] = 0
na_idx = whole_ds[(whole_ds.FireplaceQu.isnull() &
	whole_ds.Fireplaces.eq(0))].index
whole_ds.ix[na_idx, 'FireplaceQu'] = 'No fireplace'
whole_ds.ix[na_idx, 'Fireplaces'] = 0
na_idx = whole_ds[(whole_ds.GarageType.isnull() &
	whole_ds.GarageQual.isnull() & whole_ds.GarageCond.isnull())].index
whole_ds.ix[na_idx, 'GarageType'] = 'No garage'
whole_ds.ix[na_idx, 'GarageYrBlt'] = whole_ds.ix[na_idx, 'YearBuilt']
whole_ds.ix[na_idx, 'GarageFinish'] = 'No garage'
whole_ds.ix[na_idx, 'GarageQual'] = 'No garage'
whole_ds.ix[na_idx, 'GarageCond'] = 'No garage'
whole_ds.ix[na_idx, 'GarageCars'] = 0
whole_ds.ix[na_idx, 'GarageArea'] = 0
na_idx = whole_ds[(whole_ds.PoolQC.isnull() & 
	whole_ds.PoolArea.eq(0))].index
whole_ds.ix[na_idx, 'PoolQC'] = 'No pool'
whole_ds.ix[na_idx, 'PoolArea'] = 0
whole_ds.Fence.fillna('No fence', inplace=True)
whole_ds.MiscFeature.fillna('No misc', inplace=True)
# Those columns are likely to be badly set using median values since
# median values will take records where there are no pools or basement
whole_ds.BsmtQual.fillna('TA', inplace=True)
whole_ds.BsmtCond.fillna('TA', inplace=True)
whole_ds.BsmtExposure.fillna('AV', inplace=True)

# Compute medians by Neighborhood for LotFrontage
med_LFG_by_NBHD = whole_ds.groupby('Neighborhood').LotFrontage.median()

# Compute the median of Garage features with the data having a garage
med_GarageCars = whole_ds[whole_ds.GarageCars > 0].GarageCars.median()
med_GarageArea = whole_ds[whole_ds.GarageArea > 0].GarageArea.median()

# Take the most common value for MSZoning, Utilities, Exterior1st,
# Exterior2nd, Electrical, KitchenQual, Functional and SaleType per
# Neighborhood
by_Neighborhood_sorted = whole_ds.groupby('Neighborhood', sort=True)
MSZ_by_NBHD = by_Neighborhood_sorted.MSZoning.unique()
UTI_by_NBHD = by_Neighborhood_sorted.Utilities.unique()
EX1_by_NBHD = by_Neighborhood_sorted.Exterior1st.unique()
EX2_by_NBHD = by_Neighborhood_sorted.Exterior2nd.unique()
ELC_by_NBHD = by_Neighborhood_sorted.Electrical.unique()
KQL_by_NBHD = by_Neighborhood_sorted.KitchenQual.unique()
SLT_by_NBHD = by_Neighborhood_sorted.SaleType.unique()
FUNC_by_NBHD = by_Neighborhood_sorted.Functional.unique()
PQC_by_NBHD = by_Neighborhood_sorted.PoolQC.unique()

# Now, fill the NAs
for idx, record in whole_ds.iterrows():
	if record.isnull().sum() == 0:
		continue
	isnull = record.isnull()
	if isnull.BsmtFinType2:
		whole_ds.loc[idx, 'BsmtFinType2'] = 'ALQ' \
			if record.BsmtFinSF2 > 0 else 'No bsmt'
	if isnull.MasVnrType:
		whole_ds.loc[idx, 'MasVnrType'] = 'BrkCmn'
	if isnull.GarageYrBlt:
		whole_ds.loc[idx, 'GarageYrBlt'] = record.YearBuilt
	if isnull.GarageFinish:
		whole_ds.loc[idx, 'GarageFinish'] = 'RFn'
	if isnull.GarageQual:
		whole_ds.loc[idx, 'GarageQual'] = 'TA'
	if isnull.GarageCond:
		whole_ds.loc[idx, 'GarageCond'] = 'TA'
	# GarageCars and GarageArea should be the median value of the
	# records really having a garage
	if isnull.GarageCars:
		print('set GarageCars for record %d' % idx)
		whole_ds.loc[idx, 'GarageCars'] = med_GarageCars
	if isnull.GarageArea:
		whole_ds.loc[idx, 'GarageArea'] = med_GarageArea
	if isnull.LotFrontage:
		whole_ds.loc[idx, 'LotFrontage'] = \
				med_LFG_by_NBHD[record.Neighborhood]
	if isnull.MSZoning:
		whole_ds.loc[idx, 'MSZoning'] = \
				MSZ_by_NBHD[record.Neighborhood][0]
	if isnull.Utilities:
		whole_ds.loc[idx, 'Utilities'] = \
				UTI_by_NBHD[record.Neighborhood][0]
	if isnull.Exterior1st:
		whole_ds.loc[idx, 'Exterior1st'] = \
				EX1_by_NBHD[record.Neighborhood][0]
	if isnull.Exterior2nd:
		whole_ds.loc[idx, 'Exterior2nd'] = \
				EX2_by_NBHD[record.Neighborhood][0]
	if isnull.Electrical:
		whole_ds.loc[idx, 'Electrical'] = \
				ELC_by_NBHD[record.Neighborhood][0]
	if isnull.KitchenQual:
		whole_ds.loc[idx, 'KitchenQual'] = \
				KQL_by_NBHD[record.Neighborhood][0]
	if isnull.SaleType:
		whole_ds.loc[idx, 'SaleType'] = \
				SLT_by_NBHD[record.Neighborhood][0]
	if isnull.Functional:
		whole_ds.loc[idx, 'Functional'] = \
				FUNC_by_NBHD[record.Neighborhood][0]
	if isnull.PoolQC:
		whole_ds.loc[idx, 'PoolQC'] = \
				PQC_by_NBHD[record.Neighborhood][0]

def show_NAs_columns(ds):
	cols_null = whole_ds.isnull().sum()
	print('Having NAs:\n%s' % str(cols_null[cols_null > 0]))

assert(whole_ds.isnull().sum().sum() == 0)

traing_ds = whole_ds[:max_training_id]
submit_ds = whole_ds[max_training_id:].drop('SalePrice', axis=1)

assert(traing_ds.isnull().sum().sum() == 0)
assert(submit_ds.isnull().sum().sum() == 0)


traing_ds.to_csv(input_dir + 'train_full.csv', sep=',',
	encoding='utf-8', index=True, index_label='Id')
submit_ds.to_csv(input_dir + 'test_full.csv', sep=',',
	encoding='utf-8', index=True, index_label='Id')
