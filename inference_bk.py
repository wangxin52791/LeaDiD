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
import  network_case
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
            path = f'./models/{algorithm}_all'
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
                path = f'./models/{algorithm}_all'
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

#if __name__ == "__main__":
    #data=['CC1(C)CCc2[nH]c3cc(F)cc4c3c2C1N=NC4=O', 'Cn1c(C2(C)CCCN2C(=O)OCc2ccccc2)cc2c(C(=O)O)cccc21', 'c1ccc2[nH]ccc2c1', 'Cn1c(C2(C)CCCN2)cc2c(C(=O)O)cccc21', 'COC(=O)c1cccc2c1c1c(n2C)CCCC1=O', 'CN1CCCCc2[nH]c3cccc4c3c2C(=CNC4=O)C1', 'CC(C)c1ccc(C2Cc3[nH]c4cc(F)cc5c4c3C(C2)N=NC5=O)cc1', 'CC1CCN2CC(=O)c3c([nH]c4cccc(C(=O)O)c34)C12', 'O=c1ccnnc2ccc3c4ccccc4[nH]c3c12', 'CC(C)(N)C(=O)N1CC2=NNC(O)c3cccc4[nH]c(c2c34)C1', 'O=c1[nH][nH]c2c3c([nH]c4cccc1c43)=CCC2', 'CC(C)(C)C(=O)N1CC2=NNCOCc3cccc4[nH]c(c2c34)C1', 'CN(C)c1ccc(C2Cc3[nH]c4cccc5c4c3C(C2)N=NC5=O)cc1', 'CC(C)(N)C(=O)N1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'CN1CCCCc2[nH]c3cccc4c3c2C(=NNC4=O)C1', 'Cc1ccc2c3c4c([nH]c13)C1CCCN1CC4=NNC2=O', 'CC12CCCN1CC1=CNC(=O)c3cccc4[nH]c2c1c34', 'Cc1c2cc[n+](C)cc2c(C)c2c1[nH]c1ccc(O)cc12', 'CC1(C)CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'COC(=O)c1cccc2[nH]c(C3(C)CCCN3C(=O)OCc3ccccc3)cc12', 'CN1N=C2CC(C)(C)Cc3c2c2c(cccc2n3C)C1=O', 'O=C1NN=C2CCCc3[nH]c4c(F)ccc1c4c32', 'CC(C)(C)C(=O)N1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'CN1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'CC1c2[nH]c3cccc(C(=O)O)c3c2C(=O)CN1CCN(C)C', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CN1CCCC31', 'CC(C)N1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'O=C1NN=C2CCCc3[nH]c4cccc1c4c32', 'CC12CCCN1CC1=NNC(O)c3cc(F)cc4[nH]c2c1c34', 'CN1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'CC1(C)CCc2[nH]c3cc(F)cc4c3c2C1=NNC4=O', 'O=C1NN=C2CC(c3ccccc3)Cc3[nH]c4cc(F)cc1c4c32', 'CC1CCN2CC3=NNC(=O)c4cccc5[nH]c(c3c45)C12', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CNC3', 'COC(=O)c1cc(F)cc2[nH]c3c(c12)C(=O)CN1CCCC31C', 'Cc1c[nH]c2c1C13CC1CN(C(=O)c1cc4cc(NC(=O)c5coc6ccccc56)ccc4[nH]1)C3=CC2=O', 'CC(C)C1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'CC1(C)CCc2[nH]c3cccc4c3c2C1=NNC4=O', 'CC(C)C1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CN(C)C3', 'O=C1NN=C2CC(c3ccccc3)Cc3[nH]c4cccc1c4c32', 'CC12CCCN1CC1=NNC(=O)c3cc(F)cc4[nH]c2c1c34', 'CN(C)c1ccc(C2CC3=NNC(=O)c4cc(F)cc5[nH]c(c3c45)C2)cc1', 'CC(C)N1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'O=C1NN=C2CCCCc3[nH]c4cccc1c4c32', 'O=C1NN=C2CN(C(=O)C3CC3)Cc3[nH]c4cccc1c4c32', 'CC1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'O=C1NN=C2CNCc3[nH]c4cc(F)cc1c4c32', 'CN1CCCCc2[nH]c3cc(F)cc4c3c2C(=NNC4=O)C1', 'CC1(C)CCc2[nH]c3ccc(CC4CC5=NNC(=O)c6cccc7[nH]c(c5c67)C4)c4c3c2C1=NNC4=O', 'CN(C)CCn1c2c3c4c(cc(F)cc41)C(=O)NN=C3CN(C(=O)C1CC1)C2', 'CC(C)(N)C(=O)N1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'CC1(C)CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'O=C1NN=C2CNCc3[nH]c4cccc1c4c32', 'COC(=O)c1cccc2[nH]c(C3(C)CCCN3)cc12', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CN1CCCC31C', 'CN(C)c1ccc(C2CC3=NNC(=O)c4cccc5[nH]c(c3c45)C2)cc1', 'COCOCc1cccc2[nH]c(C3(C)CCCN3CC(=O)OC)cc12', 'COC(=O)c1cc(F)cc2[nH]c(C3(C)CCCN3)cc12', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CN(CCN(C)C)C3', 'CC1(C)CC2=NNC(=O)c3c(F)ccc4[nH]c(c2c34)C1', 'O=C1NN=C2CN(C3CCCCC3)Cc3[nH]c4cccc1c4c32', 'O=C1NN=C2CN(C(=O)OCc3ccccc3)Cc3[nH]c4cccc1c4c32', 'O=C1NN=C2CCCCc3[nH]c4cc(F)cc1c4c32', 'CC1(C)CC2=NN(CCO)C(=O)c3cccc4c3c2c(n4CCO)C1', 'CC1(Cc2cc3c4c5c([nH]c4c2F)CCCC5=NNC3=O)CC2=NNC(=O)c3c(F)ccc4[nH]c(c2c34)C1', 'CC1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'Cn1c2c3c4c(cccc41)C(=O)NN=C3CC(c1ccc3c4c1C(=O)NN=C1CC(C)(C)Cc(c41)n3C)C2', 'O=C1NN=C2CCCc3[nH]c4cc(F)cc1c4c32', 'NC(Cc1ccccc1)C(=O)N1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'CC12CCCN1CC1=NNC(=O)c3cccc4[nH]c2c1c34', 'CN(C)CCn1c2c3c4c(cccc41)C(=O)NN=C3CN(C(=O)C1CC1)C2', 'CC(C)(C)OC(=O)NC(Cc1ccccc1)C(=O)N1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'CCCN1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'CC(N1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1)C(C)(C)N', 'CC1(C)CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1O', 'O=C1NN=C2CN(C(=O)C3CC3)Cc3[nH]c4cc(F)cc1c4c32', 'CC(C)(C)C(=O)N1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'COC(=O)c1cc(F)cc2[nH]c(C3(C)CCCN3C(=O)C(F)(F)F)cc12', 'CN(C)CCn1c2c3c4c(cc(F)cc41)C(=O)NN=C3CC(C)(C)C2', 'Cn1c2c3c4c(cccc41)C(=O)NN=C3CCC2', 'COC(=O)CN1CCCC1(C)c1cc2c(C(=O)OC)cc(F)cc2[nH]1', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CCC3', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CN(C(=O)OCc1ccccc1)C3', 'COC(=O)c1cc(F)cc2[nH]c3c(c12)C(=O)CNC3', 'COC(=O)CN1CCCC1(C)c1cc2c(C(=O)OC)cccc2[nH]1', 'CN(C)CCN1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'O=C1NN=C2CN(C(=O)C3=CC3)Cc3[nH]c4cccc1c4c32', 'COC(=O)c1cc(C)cc2[nH]c3c(c12)C(=O)CNC3', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CC(C)(C)C3', 'Cn1c2c3c4c(cccc41)C(=O)NN=C3CC(C)(C)C2', 'CC1(C)CC2=NNC(=O)c3cccc4c3c2c(n4CCO)C1', 'CC1c2[nH]c3cc(F)cc4c3c2C(=NNC4=O)CC1(C)C', 'CC1(C)CC2=NNC(=O)c3cccc4c3c2c(n4CC2CO2)C1', 'CC1(C)CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1=O', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CN(C(=O)C1CC1)C3', 'COC(=O)CN1CCCC1(C)c1cc2c(C(=O)O)cc(F)cc2[nH]1', 'CC(C)(C)CCN1CC2=NNC(=O)c3cccc4[nH]c(c2c34)C1', 'COC(=O)c1cccc2[nH]c3c(c12)C(=O)CN(C(=O)C1=CC1)C3', 'CN(C)CCn1c2c3c4c(cccc41)C(=O)NN=C3CCC2', 'O=C1NN=C2CCCc3c2c2c1cccc2n3Cc1ccccc1', 'CN(C)CCN1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'CN(C)CCn1c2c3c4c(cccc41)C(=O)NN=C3CC(C)(C)C2', 'CC1(C)CC2=NNC(=O)c3cccc4c3c2c(n4Cc2ccccc2)C1', 'CCCN1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'CCCCN1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'CCN1CC2=NNC(=O)c3cc(F)cc4[nH]c(c2c34)C1', 'CC1(C)CC2=NNC(=O)c3cccc4c3c2c(n4CCN2CCCC2)C1', 'CCN(CC)CCn1c2c3c4c(cccc41)C(=O)NN=C3CCC2', 'CC1(C)CC2=NNC(=O)c3cccc4c3c2c(n4CCN(Cc2ccccc2)Cc2ccccc2)C1', 'O=C1NN=C2CCCc3c2c2c1cccc2n3CCN1CCCC1', 'O=C1NN=C2CCCc3c2c2c1cccc2n3CCN(Cc1ccccc1)Cc1ccccc1', 'O=C1NN=C2CCCc3c2c2c1cccc2n3CCN1CCOCC1', 'O=C1NN=C2CCCc3c2c2c1cccc2n3CCN1CCCCC1', 'CCN(CC)CCn1c2c3c4c(cccc41)C(=O)NN=C3CC(C)(C)C2', 'CC1(C)CC2=NNC(=O)c3cccc4c3c2c(n4CCN2CCCCC2)C1', 'CC1(C)CC2=NNC(=O)c3cccc4c3c2c(n4CCN2CCOCC2)C1']
    
    # filename = input()
    # data_smi = pd.read_csv(filename)
    # data_smi_df = data_smi[data_smi["SMILES"]!=np.nan]
    # smilist = data_smi_df["SMILES"].to_list()
    # main(smilist)
    # filename = '/home/yifan/PatentNetML/data/datacase/WO2017023905_SMILES.csv'
    # data_smi = pd.read_csv(filename)
    # for s in data_smi:
    #     print(type(s),len(data_smi))
    # decide_data(data_smi)
    #filename = '/home/yifan/PatentNetML/data/datacase/WO2022060836_SMILES.csv'
    #data_smi = pd.read_csv(filename)
    #data_smi=data_smi.drop_duplicates()
    #data_smi_df = data_smi[data_smi["SMILES"]!=np.nan]
    #smilist = data_smi_df["SMILES"].to_list()
    #main(smilist)
    #filename = '/home/yifan/PatentNetML/data/datacase/WO2017023905_SMILES.csv'