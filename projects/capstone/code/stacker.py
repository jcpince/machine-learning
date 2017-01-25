from threading import Lock
import xgboost as xgb
import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from utils import encode_cats_order, encode_cats_integers, encode_cats_onehot
from utils import add_SoldMonths, scale_ds, log_skewed_features, train_model
from utils import combine_features, rmsdl, get_layer_mean_preds
from utils import get_layer_median_preds, write_results
from ThreadPool import ThreadPool

script_name = os.path.realpath(__file__)
base_dir = script_name[:script_name.index('/code/')]

# Constants
RANDOM_STATE = 0
MAX_THREADS = 4
N_FOLDS = 5
MIN_SCORE = 0.2
INPUT_DIR = base_dir + '/input/'
OUTPUT_DIR = base_dir + '/output/'
BORUTA_FEATURES = ['GrLivArea', 'LotArea', 'Neighborhood', 'OverallQual',
        'YearBuilt', 'YearRemodAdd', 'BsmtQual', 'BsmtFinSF1', 'BedroomAbvGr',
        '1stFlrSF', 'LowQualFinSF', 'KitchenAbvGr', 'KitchenQual',
        'GarageFinish', 'GarageCars']
LASSO_FEATURES = ['MSZoning', 'LotFrontage', 'LotArea', 'LotShape',
        'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'OverallQual',
        'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'Exterior1st',
        'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
        'BsmtUnfSF', 'Heating', 'HeatingQC', 'CentralAir', '1stFlrSF',
        'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'KitchenQual',
        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
        'GarageFinish', 'GarageCars', 'GarageArea', 'GarageCond', 'PavedDrive',
        'PoolArea', 'Fence', 'SaleType', 'SaleCondition', 'SoldMonths']

# Globals
thread_lock = Lock()
pool = ThreadPool(MAX_THREADS)


def load_datasets():
    global INPUT_DIR
    train_ds = pd.read_csv(INPUT_DIR + 'train_full.csv').set_index('Id')
    submit_ds = pd.read_csv(INPUT_DIR + 'test_full.csv').set_index('Id')
    return train_ds, submit_ds


def prepare_datasets(train_ds, submit_ds):
    train_ds = train_ds[train_ds.GrLivArea < 4000]
    train_prices = train_ds['SalePrice']

    train_ds = add_SoldMonths(train_ds)
    train_ds.pop('TotalBsmtSF')
    submit_ds = add_SoldMonths(submit_ds)
    submit_ds.pop('TotalBsmtSF')

    whole_ds = pd.concat([train_ds.drop('SalePrice', axis=1), submit_ds])

    train_sets = dict()
    submit_sets = dict()

    # Encode the categorical features into the different flavors
    whole_cat_ordered = encode_cats_order(whole_ds, train_ds, 'SalePrice')
    whole_cat_ordered_scaled = scale_ds(whole_cat_ordered)
    whole_cat_ordered_skewed = log_skewed_features(whole_cat_ordered)
    whole_cat_int = encode_cats_integers(whole_ds)
    whole_cat_int_scaled = scale_ds(whole_cat_int)
    whole_cat_int_skewed = log_skewed_features(whole_cat_int)
    whole_cat_onehot = encode_cats_onehot(whole_ds)
    whole_cat_onehot_scaled = scale_ds(whole_cat_onehot)
    whole_cat_onehot_skewed = log_skewed_features(whole_cat_onehot)

    train_sets['ordered'] = whole_cat_ordered.ix[train_ds.index]
    train_sets['int'] = whole_cat_int.ix[train_ds.index]
    train_sets['onehot'] = whole_cat_onehot.ix[train_ds.index]
    submit_sets['ordered'] = whole_cat_ordered.ix[submit_ds.index]
    submit_sets['int'] = whole_cat_int.ix[submit_ds.index]
    submit_sets['onehot'] = whole_cat_onehot.ix[submit_ds.index]

    train_sets['ordered_scaled'] = whole_cat_ordered_scaled.ix[train_ds.index]
    train_sets['int_scaled'] = whole_cat_int_scaled.ix[train_ds.index]
    train_sets['onehot_scaled'] = whole_cat_onehot_scaled.ix[train_ds.index]
    submit_sets['ordered_scaled'] = whole_cat_ordered_scaled.ix[submit_ds.index]
    submit_sets['int_scaled'] = whole_cat_int_scaled.ix[submit_ds.index]
    submit_sets['onehot_scaled'] = whole_cat_onehot_scaled.ix[submit_ds.index]

    train_sets['ordered_skewed'] = whole_cat_ordered_skewed.ix[train_ds.index]
    train_sets['int_skewed'] = whole_cat_int_skewed.ix[train_ds.index]
    train_sets['onehot_skewed'] = whole_cat_onehot_skewed.ix[train_ds.index]
    submit_sets['ordered_skewed'] = whole_cat_ordered_skewed.ix[submit_ds.index]
    submit_sets['int_skewed'] = whole_cat_int_skewed.ix[submit_ds.index]
    submit_sets['onehot_skewed'] = whole_cat_onehot_skewed.ix[submit_ds.index]

    return train_sets, submit_sets, train_prices

layer1 = [
    {
        'name': 'XGB_layer1_0',
        'model':xgb.XGBRegressor,
        'train_input_transformation': 'ordered_skewed',
        'train_output_transformation': 'log',
        'n_folds': N_FOLDS,
        'fit_with_val': False,
        'parameters': {
            'max_depth': 3,
            'objective': 'reg:linear',
            'learning_rate': 0.05,
            'min_child_weight': 5,
            'subsample': 0.7,
            'n_estimators': 500
        }
    },
    {
        'name': 'XGB_layer1_1',
        'model':xgb.XGBRegressor,
        'train_input_transformation': 'ordered_skewed',
        'train_output_transformation': 'log',
        'n_folds': N_FOLDS,
        'combine_features': BORUTA_FEATURES,
        'parameters': {
            'max_depth': 7,
            'objective': 'reg:linear',
            'learning_rate': 0.03,
            'min_child_weight': 3,
            'subsample': 0.7,
            'n_estimators': 350
        }
    },
    {
        'name': 'GB_layer1_1',
        'model':GradientBoostingRegressor,
        'train_input_transformation': 'ordered_skewed',
        'train_output_transformation': 'div1000',
        'n_folds': N_FOLDS,
        'fit_with_val': False,
        'combine_features': LASSO_FEATURES,
        'parameters': {
            'random_state': RANDOM_STATE,
            'n_estimators': 500,
            'max_features': 5,
            'max_depth': 4,
            'learning_rate': 0.03,
            'subsample': 0.8,
        }
    },
    {
        'name': 'GB_layer1_0',
        'model':GradientBoostingRegressor,
        'train_input_transformation': 'ordered_scaled',
        'train_output_transformation': None,
        'n_folds': N_FOLDS,
        'fit_with_val': False,
        'parameters': {
            'random_state': RANDOM_STATE,
            'n_estimators': 500,
            'max_features': 15,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
        }
    },
    {
        'name': 'ET_layer1_0',
        'model':ExtraTreesRegressor,
        'train_input_transformation': 'ordered',
        'train_output_transformation': 'div1000',
        'combine_features': LASSO_FEATURES,
        'n_folds': N_FOLDS,
        'fit_with_val': False,
        'parameters': {
            'max_depth': 15,
            'random_state': RANDOM_STATE,
            'criterion': 'mse',
            'n_estimators': 760,
            'max_features': 0.68,
            'min_samples_split': 5
        }
    },
    {
        'name': 'RF_layer1_0',
        'model':RandomForestRegressor,
        'train_input_transformation': 'ordered',
        'train_output_transformation': 'sqrt',
        'combine_features': BORUTA_FEATURES,
        'n_folds': N_FOLDS,
        'fit_with_val': False,
        'parameters': {
            'max_depth': 17,
            'criterion': 'mse',
            'random_state': RANDOM_STATE,
            'n_estimators': 350,
            'max_features': 0.4
        }
    },
    {
        'name': 'Ridge_layer1_0',
        'model':Ridge,
        'train_input_transformation': 'ordered_skewed',
        'train_output_transformation': 'log',
        'n_folds': N_FOLDS,
        'fit_with_val': False,
        'parameters': {
            'alpha':0.005,
            'random_state': RANDOM_STATE,
        }
    },
    {
        'name': 'Ridge_layer1_1',
        'model':Ridge,
        'train_input_transformation': 'ordered_skewed',
        'train_output_transformation': 'sqrt',
        'n_folds': N_FOLDS,
        'combine_features': BORUTA_FEATURES,
        'fit_with_val': False,
        'parameters': {
            'alpha':0.005,
            'random_state': RANDOM_STATE,
        }
    },
    {
        'name': 'Lasso_layer1_0',
        'model':Lasso,
        'train_input_transformation': 'ordered_scaled',
        'train_output_transformation': 'log',
        'n_folds': N_FOLDS,
        'fit_with_val': False,
        'combine_features': BORUTA_FEATURES,
        'parameters': {
            'alpha':0.0006,
            'random_state': RANDOM_STATE,
            'max_iter': 100000
        }
    },
    {
        'name': 'LinReg_layer1_0',
        'model':LinearRegression,
        'train_input_transformation': 'ordered_scaled',
        'train_output_transformation': 'log',
        'n_folds': N_FOLDS,
        'parameters': {
            'fit_intercept': True,
            'normalize': False,
            'copy_X': True,
        }
    },
]

layer2 = [
    #{
        #'name': 'XGB_layer2_0',
        #'model':xgb.XGBRegressor,
        #'train_output_transformation': 'log',
        #'n_folds': N_FOLDS,
        #'parameters': {
            #'max_depth': 4,
            #'objective': 'reg:linear',
            #'learning_rate': 0.005,
            #'min_child_weight': 6,
            #'subsample': 0.7,
            #'n_estimators': 1500,
            #'seed': RANDOM_STATE
        #}
    #},
    #{
        #'name': 'XGB_layer2_1',
        #'model':xgb.XGBRegressor,
        #'train_output_transformation': 'log',
        #'n_folds': N_FOLDS,
        #'parameters': {
            #'max_depth': 2,
            #'objective': 'reg:linear',
            #'learning_rate': 0.005,
            #'min_child_weight': 4,
            #'subsample': 0.7,
            #'n_estimators': 2000,
            #'seed': RANDOM_STATE
        #}
    #},
    {
        'name': 'GB_layer2_0',
        'model':GradientBoostingRegressor,
        'train_output_transformation': 'log',
        'n_folds': N_FOLDS,
        'parameters': {
            'random_state': RANDOM_STATE,
            'n_estimators': 500,
            'max_features': 5,
            'max_depth': 15,
            'learning_rate': 0.01,
            'subsample': 0.9,
        }
    },
    {
        'name': 'RF_layer2_0',
        'model':RandomForestRegressor,
        'train_output_transformation': 'log',
        'n_folds': N_FOLDS,
        'parameters': {
            'max_depth': 17,
            'criterion': 'mse',
            'random_state': RANDOM_STATE,
            'n_estimators': 350,
            'max_features': 0.4
        }
    },
]


def create_parametrized_model(model_description):
    model = model_description['model']()
    parameters = model_description['parameters']
    for parameter in list(parameters.keys()):
        value = parameters[parameter]
        setattr(model, parameter, value)
    return model


def train_layer(layer, train_sets, train_labels, submit_sets):
    if type(train_sets) == dict:
        layer_train_output = pd.DataFrame(index=train_sets['int'].index)
        layer_submit_output = pd.DataFrame(index=submit_sets['int'].index)
    else:
        layer_train_output = pd.DataFrame(index=train_sets.index)
        layer_submit_output = pd.DataFrame(index=submit_sets.index)
    for model_description in layer:
        model = create_parametrized_model(model_description)
        model_name = model_description['name']
        print(('Training %s...' % model_name))
        if type(train_sets) == dict:
            X = train_sets[model_description['train_input_transformation']]
            S = submit_sets[model_description['train_input_transformation']]
        elif type(train_sets) == pd.DataFrame:
            X = train_sets
            S = submit_sets
        if 'combine_features' in list(model_description.keys()):
            X = combine_features(X, model_description['combine_features'])
            S = combine_features(S, model_description['combine_features'])
        Y_transformation = model_description['train_output_transformation']
        n_folds = model_description['n_folds']
        fit_with_val = False
        if 'fit_with_val' in list(model_description.keys()):
            fit_with_val = model_description['fit_with_val']
        train_model(model_name, model, X, train_labels, S,
            layer_train_output, layer_submit_output, n_folds, MIN_SCORE,
            thread_lock, RANDOM_STATE, transformation=Y_transformation,
            fit_with_val=fit_with_val, verbose=True, use_kfold=True)
    return layer_train_output, layer_submit_output

if __name__ == "__main__":
    train_ds, submit_ds = load_datasets()
    train_sets, submit_sets, train_prices = prepare_datasets(
            train_ds, submit_ds)

    if os.path.isfile("cache/layer1_train_output.csv"):
        print('Skip Layer1 training, cache files found')
        layer1_train_output = pd.read_csv(
            'cache/layer1_train_output.csv').set_index('Id')
        layer1_submit_output = pd.read_csv(
            'cache/layer1_submit_output.csv').set_index('Id')
    else:
        print('Layer1 training starting...')
        layer1_train_output, layer1_submit_output = train_layer(layer1,
                train_sets, train_prices, submit_sets)
        print('Layer1 training terminated')
        if not os.path.exists('cache'):
            os.makedirs('cache')
        layer1_train_output.to_csv('cache/layer1_train_output.csv')
        layer1_submit_output.to_csv('cache/layer1_submit_output.csv')

    layer1_train_output = pd.concat([layer1_train_output,
            train_sets['ordered_scaled'][LASSO_FEATURES]], axis=1)
    layer1_submit_output = pd.concat([layer1_submit_output,
            submit_sets['ordered_scaled'][LASSO_FEATURES]], axis=1)

    if os.path.isfile("cache/layer2_train_output.csv"):
        print('Skip Layer2 training, cache files found')
        layer2_train_output = pd.read_csv(
            'cache/layer2_train_output.csv').set_index('Id')
        layer2_submit_output = pd.read_csv(
            'cache/layer2_submit_output.csv').set_index('Id')
    else:
        print('Layer2 training starting...')
        layer2_train_output, layer2_submit_output = train_layer(layer2,
                layer1_train_output, train_prices, layer1_submit_output)
        print('Layer2 training terminated')
        layer2_train_output.to_csv('cache/layer2_train_output.csv')
        layer2_submit_output.to_csv('cache/layer2_submit_output.csv')

    print('==================================================')
    print('==================================================')
    print('Layer 1 scores:')
    layer1_train_output.drop(LASSO_FEATURES, axis=1, inplace=True)
    layer1_submit_output.drop(LASSO_FEATURES, axis=1, inplace=True)
    for algo in layer1_train_output:
        score = rmsdl(train_prices, layer1_train_output[algo])
        print(('\t%s:\t\t %0.6f' % (algo,
            score)))
        write_results(submit_ds.index, layer1_submit_output[algo].as_matrix(),
            score, 'stacker_l1_' + algo, '.')

    print('==================================================')
    print('==================================================')
    print('Layer 2 scores:')
    for algo in layer2_train_output:
        score = rmsdl(train_prices, layer2_train_output[algo])
        print(('\t%s:\t\t %0.6f' % (algo,
            rmsdl(train_prices, layer2_train_output[algo]))))
        write_results(submit_ds.index, layer2_submit_output[algo].as_matrix(),
            score, 'stacker_l2_' + algo, '.')
    l2_mean_score = rmsdl(train_prices,
            get_layer_mean_preds(layer2_train_output))
    l2_median_score = rmsdl(train_prices,
            get_layer_median_preds(layer2_train_output))
    print(('L2 score of the mean predictions:\t %0.6f' % l2_mean_score))
    print(('L2 score of the median predictions:\t %0.6f' % l2_median_score))

    write_results(submit_ds.index,
        get_layer_mean_preds(layer2_submit_output), l2_mean_score,
        'stacker_mean_l2', '.')
    write_results(submit_ds.index,
        get_layer_median_preds(layer2_submit_output), l2_median_score,
        'stacker_median_l2', '.')
