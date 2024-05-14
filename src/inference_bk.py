import os
import sys
import random
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings("ignore")
from rdkit.DataStructs import cDataStructs
import numpy as np
import pandas as pd
import  LeaDiD.src.network_case as network_case
from rdkit import Chem
from mordred import Calculator, descriptors
import concurrent.futures

def decide_data(data):
        # 判断是否为SMILES字符串列表
    if all(isinstance(s, str) for s in data):
        print("输入数据为SMILES字符串列表")
    
    # 判断是否为CSV转换后的DataFrame
    elif isinstance(data, pd.DataFrame):
        print("输入数据为CSV转换后的DataFrame")

        # 进一步检查是否包含SMILES字符串列
        if "SMILES" in data.columns:
            print("DataFrame包含SMILES字符串列")
    
    # 判断是否为SDF文件
    else:
        try:
            suppl = Chem.SDMolSupplier(data)
            mol = next(suppl)
            if mol is not None:
                print("输入数据为SDF文件")
        except:
            print("未知数据格式")

def handelSmi(smi):
    try:
        return Chem.MolFromSmiles(smi, sanitize=True)
    except:
        return Chem.MolFromSmiles("C", sanitize=True)
    
    
def calc_descriptors(mols):
    calc = Calculator(descriptors, ignore_3D=True)
    return calc.pandas(mols)

def data_transform(df):
    if type(df) == list:
        print('data是列表')
        df_todf = pd.DataFrame()
        df_todf['P_Ca_SMILES'] = df
        df_case_g = df_todf
    else:
        if set(['PATENT_ID', 'P_Ca_SMILES']).issubset(df.columns):          
            df_case_g = df.loc[:, ['PATENT_ID', 'P_Ca_SMILES']]  
        else:
            df_case_g = df.loc[:, ['PATENT_ID', 'P_Ca_SMILES']]  

    smis = df_case_g.P_Ca_SMILES
    mols = [handelSmi(smi) for smi in smis]
    df_smile = df_case_g
    fp_class = ['ecfp4']
    header = [f'{fp_class[0]}_' + str(i) for i in range(1024)]
    fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
    fp_df = pd.DataFrame(np.array(fp), columns=header)
    fp_df = pd.concat([df_smile, fp_df], axis=1)
    # print('start.......calc')
    
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     # 提交第一个任务并获取Future对象
    #     future1 = executor.submit(network_case.main, fp_df, fp_class)
        
    #     # 提交第二个任务并获取Future对象
    #     future2 = executor.submit(calc_descriptors, mols)
        
    #     # 等待并获取第一个任务的结果
    #     fp_df = future1.result()
        
    #     # 等待并获取第二个任务的结果
    #     descriptors_result = future2.result()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks and get results
        future_fp = executor.submit(network_case.main, fp_df, fp_class)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_desc = executor.submit(calc_descriptors, list(mols))
        
        
        # Get results
    fp_df = future_fp.result()
    descriptors_result = future_desc.result()
    
    
    df_mod = descriptors_result
    
    df_mod_x = df_mod.apply(pd.to_numeric, errors='coerce')
    df_mod_x.fillna(0, inplace=True)
    df_case1 = pd.concat([fp_df, df_mod_x], axis=1)
    df_case1.fillna(0, inplace=True)
    
    return df_case1


def feature_selection_saved(X,file):
    with open(file,'rb') as f:
        process = pickle.load(f)
    X = process.get('fs1').transform(X.values)
    X = process.get('fs2').transform(X)
    df = pd.DataFrame(X,columns=process.get('fs3'))
    columns = process.get('fs4')
    df_final = df[columns]
    X_final = np.array(df_final)
    return X_final,columns


def single_inference(algorithms,dict_rounds_get,X_,df_probability_total):
        df_probability = pd.DataFrame()
        for algorithm in algorithms:
            path = f'../models/{algorithm}_all'
            #path = f'../results/{algorithm}_all'
            for r in dict_rounds_get.get(algorithm):
                file=f'{path}/FeatureSelection_{r}.pkl'
                X,columns = feature_selection_saved(X_,file)
                
                
                with open(f'{path}/{algorithm}_{r}.pkl','rb') as f:
                    clf = pickle.load(f)

                y_proba = clf.predict_proba(X)[:,1]
                df_probability[f'Probability_{algorithm}_{r}'] = y_proba       
            if algorithm == algorithms[0]:
                col_num = df_probability.shape[1]
                df_probability[f'Probability_{algorithm}'] = [np.mean(df_probability.iloc[i,0:].tolist()) for i in range(df_probability.shape[0])]
            elif algorithm == algorithms[1]:
                df_probability[f'Probability_{algorithm}'] = [np.mean(df_probability.iloc[i,col_num+1:].tolist()) for i in range(df_probability.shape[0])]

        df_probability['Probability'] = [np.mean(df_probability.iloc[i,[col_num,-1]].tolist()) for i in range(df_probability.shape[0])]
        # df_probability.sort_values(by=['Probability'],inplace=True,ascending=False,ignore_index=True)
        y_pred_single = np.around(df_probability.Probability.tolist(),0).astype(int)
        df_probability['pred']=y_pred_single 
        df_probability_total = pd.concat([df_probability_total,df_probability])
        return df_probability_total
    
    
def inference(df_test):
    algorithms = ['xgboost','rf']            
    dict_rounds_get = {
        'xgboost': [104, 51, 2, 199, 181],
        'rf'     : [48, 84, 25, 50, 36]
        
    }

    df_probability_total = pd.DataFrame()
    if set(['PATENT_ID','P_Ca_SMILES']).issubset(df_test.columns):        
        for p in df_test.PATENT_ID.unique():
            df_p = df_test[df_test.PATENT_ID == p].reset_index(drop=True)
            X_ = df_p.drop(columns=['PATENT_ID','P_Ca_SMILES'])
            df_probability = pd.DataFrame({'PATENT_ID':df_p.PATENT_ID.tolist(),
                                        'P_Ca_SMILES':df_p.P_Ca_SMILES.tolist(),
                                    })
            for algorithm in algorithms:
                path = f'../models/{algorithm}_all'
                #path = f'../results/{algorithm}_all'
                for r in dict_rounds_get.get(algorithm):
                    file=f'{path}/FeatureSelection_{r}.pkl'
                    X,columns = feature_selection_saved(X_,file)
                    with open(f'{path}/{algorithm}_{r}.pkl','rb') as f:
                        clf = pickle.load(f)

                    y_proba = clf.predict_proba(X)[:,1]
                    df_probability[f'Probability_{algorithm}_{r}'] = y_proba
                if algorithm == algorithms[0]:
                    col_num = df_probability.shape[1]
                    df_probability[f'Probability_{algorithm}'] = [np.mean(df_probability.iloc[i,3:].tolist()) for i in range(df_probability.shape[0])]
                elif algorithm == algorithms[1]:
                    df_probability[f'Probability_{algorithm}'] = [np.mean(df_probability.iloc[i,col_num+1:].tolist()) for i in range(df_probability.shape[0])]

            df_probability['Probability'] = [np.mean(df_probability.iloc[i,[col_num,-1]].tolist()) for i in range(df_probability.shape[0])]
            # df_probability.sort_values(by=['Probability'],inplace=True,ascending=False,ignore_index=True)
            y_pred_single = np.around(df_probability.Probability.tolist(),0).astype(int)
            df_probability['pred']=y_pred_single 
            df_probability_total = pd.concat([df_probability_total,df_probability])
        df_probability_total.to_csv('result_ourmodel.csv')
    else:
        ori = df_test
        X_=df_test.drop(columns=['P_Ca_SMILES'])
        df_probability_total=single_inference(algorithms,dict_rounds_get,X_,df_probability_total)
        df_probability_total["SMILES"]=ori['P_Ca_SMILES']
        df_probability_total.to_csv('result_ourmodel_single.csv')
    return df_probability_total

def main(data):
    inference_data=data_transform(data)
    print(inference(inference_data))

