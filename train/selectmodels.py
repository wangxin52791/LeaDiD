import pandas as pd


df_rf=pd.read_csv('../results/rf_all/results.csv')
df_xgboost=pd.read_csv('../results/xgboost_all/results.csv')
df_rf['recall_testauc']=df_rf['test_recall']+df_rf['test_auc']
df_xgboost['recall_testauc']=df_xgboost['test_recall']+df_xgboost['test_auc']
df_rf_final=df_rf.nlargest(5,'recall_testauc',keep='all')
df_xgboost_final=df_xgboost.nlargest(5,'recall_testauc',keep='all')
print(df_xgboost_final)
print(df_rf_final.rounds.tolist())
print(df_xgboost_final.rounds.tolist())