import xgboost as xgb
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import time
# --------------------------------------------------- prepare data
train_agg = pd.read_csv('./dataset/train/train_agg.csv', sep='\t')
train_flg = pd.read_csv('./dataset/train/train_flg.csv', sep='\t')
train_log = pd.read_csv('./dataset/train/train_log.csv', sep='\t')
test_agg = pd.read_csv('./dataset/test/test_agg.csv', sep='\t')
test_log = pd.read_csv('./dataset/test/test_log.csv', sep='\t')
# 将agg中的类别特征改写成整数
all_aggs = pd.concat([train_agg, test_agg])
Vs = [1+1, 2+1, 3+1, 4+1, 15+1, 21+1, 25+1]
for v in Vs:
    le = LabelEncoder()
    le.fit(all_aggs['V'+str(v)])
    train_agg['V'+str(v)] = le.transform(train_agg['V'+str(v)])
    test_agg['V'+str(v)] = le.transform(test_agg['V'+str(v)])

train = shuffle(pd.merge(train_agg, train_flg, on='USRID'))
logs = train_log.append(test_log, ignore_index=True)

# --------------------------------------------------- feature engineering
# 每个用户日志记录总数
user_logs = logs.groupby(['USRID']).size().reset_index()
user_logs.columns = ['USRID', 'USR_LOGS']
# 每个用户对每一个模块(一二三级模块)的点击数量
logs['EVT1'] = logs.EVT_LBL.map(lambda x: x.split('-')[0]+'_')
evt1_log_num = logs.groupby(['USRID', 'EVT1']).size().reset_index()
evt1_aggs = evt1_log_num.groupby('USRID')[0].agg(['max', 'size', 'mean']).reset_index()
evt1_aggs.columns = ['USRID', 'max_evt1', 'size_evt1', 'mean_evt1']
evt1_logs = evt1_log_num.pivot_table(index='USRID', columns='EVT1', values=0, fill_value=0).reset_index()

logs['EVT2'] = logs.EVT_LBL.map(lambda x: '-'.join(x.split('-')[0:2]))
evt2_log_num = logs.groupby(['USRID', 'EVT2']).size().reset_index()
evt2_aggs = evt2_log_num.groupby('USRID')[0].agg(['max', 'size', 'mean']).reset_index()
evt2_aggs.columns = ['USRID', 'max_evt2', 'size_evt2', 'mean_evt2']
evt2_logs = evt2_log_num.pivot_table(index='USRID', columns='EVT2', values=0, fill_value=0).reset_index()

evt3_log_num = logs.groupby(['USRID', 'EVT_LBL']).size().reset_index()
evt3_aggs = evt3_log_num.groupby('USRID')[0].agg(['max', 'size', 'mean']).reset_index()
evt3_aggs.columns = ['USRID', 'max_evt3', 'size_evt3', 'mean_evt3']
evt3_logs = evt3_log_num.pivot_table(index='USRID', columns='EVT_LBL', values=0, fill_value=0).reset_index()

# 查看的时间特征(以天统计和以小时统计)
logs['DATE'] = logs.OCC_TIM.map(lambda x: x[8:10])
date_aggs_0 = logs.groupby('USRID')['DATE'].agg(['max', 'min']).reset_index()
date_aggs_0.columns = ['USRID', 'date_1', 'date_2']
date_aggs_0['date_1'] = [int(d) for d in date_aggs_0['date_1'].values]
date_aggs_0['date_2'] = [int(d) for d in date_aggs_0['date_1'].values]

date_log_num = logs.groupby(['USRID', 'DATE']).size().reset_index()
date_aggs = date_log_num.groupby('USRID')[0].agg(['max', 'size', 'mean']).reset_index()
date_aggs.columns = ['USRID', 'max_date', 'size_date', 'mean_date']
date_logs = date_log_num.pivot_table(index='USRID', columns='DATE', values=0, fill_value=0).reset_index()

logs['HOUR'] = logs.OCC_TIM.map(lambda x: x[11:13])
hour_log_num = logs.groupby(['USRID', 'HOUR']).size().reset_index()
hour_aggs = hour_log_num.groupby('USRID')[0].agg(['max', 'size', 'mean']).reset_index()
hour_aggs.columns = ['USRID', 'max_hour', 'size_hour', 'mean_hour']
hour_logs = hour_log_num.pivot_table(index='USRID', columns='HOUR', values=0, fill_value=0).reset_index()

# 每一个用户的事件类型数
tch_logs = logs.groupby(['USRID', 'TCH_TYP']).size().reset_index()
tch_logs = tch_logs.pivot_table(index='USRID', columns='TCH_TYP', values=0, fill_value=0).reset_index()

# --------------------------------------------------- combine features
train = pd.merge(train, user_logs, on='USRID', how='left')
train = train.merge(evt1_logs, on='USRID', how='left')
train = train.merge(evt2_logs, on='USRID', how='left')
train = train.merge(evt3_logs, on='USRID', how='left')
train = train.merge(evt1_aggs, on='USRID', how='left')
train = train.merge(evt2_aggs, on='USRID', how='left')
train = train.merge(evt3_aggs, on='USRID', how='left')
train = train.merge(date_logs, on='USRID', how='left')
train = train.merge(hour_logs, on='USRID', how='left')
train = train.merge(date_aggs, on='USRID', how='left')
train = train.merge(date_aggs_0, on='USRID', how='left')
train = train.merge(hour_aggs, on='USRID', how='left')
train = train.merge(tch_logs, on='USRID', how='left')
test = pd.merge(test_agg, user_logs, on='USRID', how='left')
test = test.merge(evt1_logs, on='USRID', how='left')
test = test.merge(evt2_logs, on='USRID', how='left')
test = test.merge(evt3_logs, on='USRID', how='left')
test = test.merge(evt1_aggs, on='USRID', how='left')
test = test.merge(evt2_aggs, on='USRID', how='left')
test = test.merge(evt3_aggs, on='USRID', how='left')
test = test.merge(date_logs, on='USRID', how='left')
test = test.merge(hour_logs, on='USRID', how='left')
test = test.merge(date_aggs, on='USRID', how='left')
test = test.merge(date_aggs_0, on='USRID', how='left')
test = test.merge(hour_aggs, on='USRID', how='left')
test = test.merge(tch_logs, on='USRID', how='left')
train = train.fillna(0)
test = test.fillna(0)
train_Y = train.pop('FLAG').values
train.pop('USRID')
test_usrid = test.pop('USRID')
train_X = train
print('train_X.shape =', train_X.shape)


# --------------------------------------------------- model settings
params = {
    'colsample_bytree': 0.1,
    'learning_rate': 0.02,
    'objective': 'binary:logistic',
    'max_depth': 5,
    'eval_metric': 'auc',
    'silent': 1
}

# --------------------------------------- train model using 10-folds cv
kf = KFold(n_splits=10, shuffle=True, random_state=2018)
for k, (train_index, test_index) in enumerate(kf.split(train_X)):
    start_time = time.time()
    train_data = train_X.iloc[train_index, :]
    test_data = train_X.iloc[test_index, :]
    train_label = [train_Y[ti] for ti in train_index]
    test_label = [train_Y[ti] for ti in test_index]

    dtrain = xgb.DMatrix(train_data, label=train_label)
    dtest = xgb.DMatrix(test_data)
    xgb_model = xgb.train(params, dtrain, 300+100*k)
    test_pred = xgb_model.predict(dtest)
    score = roc_auc_score(test_label, test_pred)
    print('k =', k, 'auc_score =', score)
    print('duration =', time.time() - start_time, 's')
    print('\n')

