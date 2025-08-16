# -*- coding: utf-8 -*-
"""
This is a code for "A Field-scale Paddy Rice Yield Estimation Framework: Leveraging Spatial Heterogeneity of Crop Growth in Small-sample Scenarios"
by Anqi Li, Tianyu Liu, Kai Tang, Cong Wang*, Xuehong Chen, Chishan Zhang, Lin Qiu, Jin Chen

@author: Anqi Li

Input: the feature data of each sample field
Output: the result of model accuracy
"""

import pandas as pd
import numpy as np
import shap
from scipy.optimize import linprog, minimize, LinearConstraint, NonlinearConstraint
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import random
import os
from sklearn.svm import SVR
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from tabpfn import TabPFNClassifier, TabPFNRegressor
from sklearn.inspection import permutation_importance
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from pykrige.ok import OrdinaryKriging
from openpyxl import Workbook
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU. See section above for instructions.')
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
torch.cuda.empty_cache()

def idw(s_locate,s_y,t_locate):
    sample_num, feature_len = s_locate.shape #Sample's coordinate
    target_num, feature_len = t_locate.shape #Target's coordinate
    weight = np.zeros((target_num, sample_num)) #Neighbor id
    s_y=np.array(s_y).reshape(-1,1)
    for i in range(0,target_num):
        for j in range(0,sample_num):
             t_L=t_locate[i,:]
             s_L=s_locate[j,:]
             w=1/np.sum((t_L-s_L)**2)
             weight[i,j] = w
        weights = np.sum(weight[i,:])
        weight[i,:] = weight[i,:] / weights
    syn_data = np.dot(weight,s_y)
    return syn_data


def metrics(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    r, p = pearsonr(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    RRMSE = np.sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)
    Linear = LinearRegression()
    Linear.fit(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
    l = Linear.coef_[0][0]
    MAE = np.mean(np.abs(y_test - y_pred))
    return r2, RMSE, RRMSE, l, r, MAE


def objective_function(w, X, y):
    return np.sum((X.dot(w) - y) ** 2)


#---The path of feature data
train_data = pd.read_excel(r'../dataset/Sample.xlsx', sheet_name="train")
test_data = pd.read_excel(r'../dataset/Sample.xlsx', sheet_name="test")
#---The path of saving accuracy results
output_dir = '../Results'
#---Set up the sample size
s_size = 50
#---Select sampling method for training dataset
# 'None': use all of the data without sampling
# 'SR': simple random sampling
# 'SS': systematic sampling
# 'ST': stratified sampling
Sampling='None'
#---Auxiliary data are needed to be assigned when using sampling methods of 'SS' and 'ST'
aux='yield'  # Yield
# aux=3        # The index of feature variable

#%%
train_data.set_index(train_data.columns[0], inplace=True)
test_data.set_index(test_data.columns[0], inplace=True)
y_all_list = []
y_s_list = []
metrics_dict = {'r2': {}, 'rmse': {}, 'rrmse': {},'a': {}, 'r': {}, 'mae': {}}
models = ('scm','rf','xgboost','svr','Tab','l_c','rf_c','xgboost_c','svr_c','Tab_c','knn_c','kriging','idw')
for name in models:
    for metric in metrics_dict:
        metrics_dict[metric][name] = []



# ---Iterations of experiments
for i in tqdm(range(1)):

    #---Step 1: Data preparation
    #---Sampling method selection

    #---1. Simple random sampling
    if Sampling == 'SR':
        sample = train_data.sample(s_size,random_state=i)

    #---2. Systematic sampling
    elif Sampling == 'SS':
        sample = np.zeros((s_size, train_data.shape[1]))
        feature_all = train_data.sort_values(by=aux, ascending=True)
        feature_all = feature_all.reset_index(drop=True)
        gn = train_data.shape[0]/s_size
        num = random.randint(0,int(gn))
        for k in range(0,s_size):
            samplek= feature_all.loc[int(k*gn)+num]
            sample[k,:]=samplek
        sample = pd.DataFrame(sample)
        sample.rename(columns={sample.shape[1] - 1: 'y'}, inplace=True)
        sample.rename(columns={sample.shape[1] - 2: 'x'}, inplace=True)

    #---3. Stratified sampling
    elif Sampling == 'ST':
        sample = np.zeros((0, train_data.shape[1]))
        y = train_data['yield'].values
        for k in range(5,12):
            data_choose=train_data.loc[(train_data['yield'] >= k) & (train_data['yield'] < k+1)]
            group=data_choose.shape[0]
        #---Ensure the samples from small strata are included
            if k == 5:
                samplek = np.array(data_choose.sample(1))
            elif k == 11:
                samplek = np.array(data_choose.sample(2))
            else:
                sample_num=int(group/50*s_size)
                samplek= np.array(data_choose.sample(sample_num,random_state=i))
            sample=np.concatenate((sample,samplek),axis=0)
        sample = pd.DataFrame(sample)
        sample.rename(columns={sample.shape[1] - 1: 'y'}, inplace=True)
        sample.rename(columns={sample.shape[1] - 2: 'x'}, inplace=True)

    else:
        sample = train_data

    lons = sample["x"].values
    lats = sample["y"].values
    sample = sample.drop(["x", "y"], axis=1)
    sample.rename(columns={sample.shape[1] - 1: 'yield'}, inplace=True)

    y_s = sample.loc[:, 'yield']
    y_s_list.append(y_s)
    y_s = y_s.to_numpy()
    y_ss = np.repeat(y_s,2).reshape(-1,2) # Duplicate the yield of samples for standardization

    data_t = test_data.sort_index()
    y_t = data_t.loc[:, 'yield']
    t_lon = data_t.loc[:, 'x']
    t_lat = data_t.loc[:, 'y']
    X_test = data_t.drop(['yield','x','y'], axis=1)

# %%
    yield_scm_list = []
    yield_rf_list = []
    yield_xgboost_list = []
    yield_svr_list = []
    yield_Tab_list = []
    y_l_list = []
    y_rf_list = []
    y_xgboost_list = []
    y_svr_list = []
    y_Tab_list = []
    y_knn_list = []


# %%
    for i in X_test.index:
    #---Sample synthesis framework
    #---Step 2: Data standardization
        sample = np.array(sample)
        X_s_o = sample[:, 0:-1]
        X_t_o = X_test.loc[i]
        X_t_o = np.array(X_t_o).reshape(1,-1)
        X_ts = np.concatenate((X_s_o,X_t_o),axis=0)
        X_trans = X_ts
        ss = StandardScaler()
        #ss = MinMaxScaler()
        #ss = RobustScaler()
        X_ts = ss.fit_transform(X_ts).T
        X_s = X_ts[:, 0:X_s_o.shape[0]]
        X_t = X_ts[:, -1]

    #---Step 3: Weight/Regressor generation and yield synthesis
        #---Linear regression with non-negative and sum to 1 constraints
        initial_w = np.ones((X_s_o.shape[0])) / (X_s_o.shape[0])
        constraints = (
            {'type': 'eq', 'fun': lambda w: (np.sum(w) - 1)},
            {'type': 'ineq', 'fun': lambda w: w}
        )
        result = minimize(objective_function, initial_w, args=(X_s, X_t),constraints=constraints)
        optimal_w = result.x
        yield_scm = np.sum(optimal_w * y_s)
        l_x = X_s.dot(optimal_w)
        yield_scm_list.append(yield_scm)

        #---Random Forest Regression
        ss = StandardScaler()
        #ss = MinMaxScaler()
        #ss = RobustScaler()
        y_sss = ss.fit_transform(y_ss)
        rf1 = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt',n_jobs=-1, random_state=42)
        rf1.fit(X_s, X_t)
        yield_rf = rf1.predict(y_sss.T)
        rf_x = rf1.predict(X_s)
        yield_rf_list.append(yield_rf)
        #---Feature importance
        # iprf = rf1.feature_importances_
        # explainer_rf = shap.Explainer(rf1,X_s)
        # shap_values_rf = explainer_rf.shap_values(X_s)
        # shap_rf = np.mean(abs(shap_values_rf),axis=0)


        #---XGBoost
        dtrain = xgb.DMatrix(X_s, label=X_t)
        params = {'gamma': 0.01, 'learning_rate': 0.01, 'max_depth': 5,  "n_jobs":-1}
        model = xgb.train(params, dtrain, num_boost_round=10)
        xg_x = model.predict(xgb.DMatrix(X_s))
        yield_xgb = model.predict(xgb.DMatrix(y_sss.T))
        yield_xgboost_list.append(yield_xgb)
        #---Feature importance
        # ip_xgb = model.get_score(fmap='', importance_type='gain')
        # explainer_xgb = shap.Explainer(model,X_s)
        # shap_values_xgb = explainer_xgb.shap_values(X_s)
        # shap_xgboost=np.mean(abs(shap_values_xgb),axis=0)


        #---Support Vector Regression
        svr = SVR(C=10, gamma='auto', kernel='rbf')
        svr.fit(X_s, X_t)
        yield_svr = svr.predict(y_sss.T)
        svr_x = svr.predict(X_s)
        yield_svr_list.append(yield_svr)
        #---Feature importance
        # ipsvr = permutation_importance(svr, X_s, X_t, n_repeats=10, random_state=42)
        # ip_svr = ipsvr.importances_mean
        # explainer_svr = shap.KernelExplainer(svr.predict,X_s)
        # shap_values_svr = explainer_svr.shap_values(X_s)
        # shap_svr=np.mean(abs(shap_values_svr),axis=0)


        #---TabPFN
        reg1 = TabPFNRegressor(random_state=42,device="cuda")
        #reg1 = AutoTabPFNRegressor(max_time=120, device="cuda")  # 120 seconds tuning time
        reg1.fit(X_s, X_t)
        Tab_x = reg1.predict(X_s)
        yield_Tab = reg1.predict(y_sss.T)
        yield_Tab_list.append(yield_Tab)
        #---Feature importance
        # explainer_Tab = shap.Explainer(reg1.predict,X_s)
        # shap_values_Tab = explainer_Tab(X_s)
        # shap_v_Tab = shap_values_Tab.values
        # shap_Tab = np.mean(abs(shap_v_Tab),axis=0)


    #---Conventional framework
        x_tr=X_s.T
        y_tr=y_s
        x_ts=X_t.reshape(1,-1)

        #---Linear regression
        linear = LinearRegression()
        linear.fit(x_tr, y_tr)
        y_l = linear.predict(x_ts)
        y_l_list.append(y_l)

        #---Random Forest Regression
        rf = RandomForestRegressor(n_estimators=20,max_features='sqrt',max_depth=12,n_jobs=-1, random_state=42)
        rf.fit(x_tr, y_tr)
        y_rf = rf.predict(x_ts)
        y_rf_list.append(y_rf)

        #---XGBoost
        dtrain = xgb.DMatrix(x_tr, label=y_tr)
        params = {'gamma': 0.01, 'learning_rate': 0.01, 'max_depth': 5,  "n_jobs":-1, }
        model = xgb.train(params, dtrain, num_boost_round=10)
        y_xgboost = model.predict(xgb.DMatrix(x_ts))
        y_xgboost_list.append(y_xgboost)

        #---Support Vector Regression
        svr = SVR(C= 1, gamma='auto', kernel='rbf')
        svr.fit(x_tr, y_tr)
        y_svr = svr.predict(x_ts)
        y_svr_list.append(y_svr)

        #---TabPFN
        reg = TabPFNRegressor(device='cuda',random_state=42)
        reg.fit(x_tr, y_tr)
        y_Tab = reg.predict(x_ts)
        y_Tab_list.append(y_Tab)

        #---K-Nearest Neighbor
        knn_regressor = KNeighborsRegressor(algorithm='auto', n_neighbors=9, weights='uniform')
        knn_regressor.fit(x_tr, y_tr)
        y_knn = knn_regressor.predict(x_ts)
        y_knn_list.append(y_knn)

    #---Spatial interpolation
    #---Kriging
    OK = OrdinaryKriging(lons, lats, y_s, variogram_model='spherical')
    yield_kriging, SD = OK.execute('points', t_lon, t_lat)
    yield_kriging = np.array(yield_kriging)
    #---Inverse Distance Weighting interpolation
    s_location= np.concatenate((np.array(lons).reshape(-1,1), np.array(lats).reshape(-1,1)),axis=1)
    t_location = np.concatenate((np.array(t_lon).reshape(-1,1), np.array(t_lat).reshape(-1,1)),axis=1)
    yield_idw = idw(s_location, y_s, t_location).reshape(-1)

#%%
#---Accuracy assessment
    y_t = np.array(y_t)
    models = {
        'scm': np.array(yield_scm_list),
        'rf': np.array(yield_rf_list),
        'xgboost': np.array(yield_xgboost_list),
        'svr': np.array(yield_svr_list),
        'Tab': np.array(yield_Tab_list),

        'l_c': np.array(y_l_list).reshape(-1),
        'rf_c': np.array(y_rf_list).reshape(-1),
        'xgboost_c': np.array(y_xgboost_list).reshape(-1),
        'svr_c': np.array(y_svr_list).reshape(-1),
        'Tab_c': np.array(y_Tab_list).reshape(-1),
        'knn_c': np.array(y_knn_list).reshape(-1),
        'kriging': yield_kriging,
        'idw': yield_idw,
    }
#---De-standardization of yield
    for name in models:
        if name in ('rf','xgboost','svr','Tab'):
            models[name] = ss.inverse_transform(models[name])[:,0]

    #---Results of sample synthesis
    yield_all = np.concatenate([
        y_t.reshape(1, -1),
        models['scm'].reshape(1, -1),
        models['rf'].reshape(1, -1),
        models['xgboost'].reshape(1, -1),
        models['svr'].reshape(1, -1),
        models['Tab'].reshape(1, -1)
    ])
    y_all_list.append(yield_all)

#---Accuracy evaluation metrics calculation
    for name in models:
        y_pred = models[name]
        r2, rmse, rrmse, a, r, mae = metrics(y_t, y_pred)
        metrics_dict['r2'][name].append(r2)
        metrics_dict['rmse'][name].append(rmse)
        metrics_dict['rrmse'][name].append(rrmse)
        metrics_dict['a'][name].append(a)
        metrics_dict['r'][name].append(r)
        metrics_dict['mae'][name].append(mae)

results_data = {}
for metric, model_data in metrics_dict.items():
    for model, values in model_data.items():
        results_data[f'{metric}_{model}'] = values

result_df = pd.DataFrame(results_data)
result_mean = result_df.mean(axis=0)
#---The statistical results of multiple experiments
result_mean = result_df.mean(axis=0).to_frame('mean')
result_std = np.std(result_df,axis=0).to_frame('std')

print("Result：\n", result_mean)

#---Save the results
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
stats_df = pd.concat([result_mean, result_std],axis=1)
stats_df.to_csv(os.path.join(output_dir, 'evaluation_statistics.csv'))

print(f"The accuracy evaluation results have been saved to {output_dir}")

