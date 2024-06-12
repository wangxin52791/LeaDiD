from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
import os
import sys
import random
import numpy as np
import pandas as pd
import network_case
import rf_train_case
import xgbost_train_case
from rdkit import Chem
from mordred import Calculator, descriptors

# df_test = pd.read_csv('../data/mini-test.csv')
# df_train=pd.read_csv('../data/mini-train.csv')
# df_valid=pd.read_csv('../data/mini-valid.csv')

# print(df_test) 
# print(df_valid)

# def data_transform(df):    

#     df_case_g=df.loc[:,['PATENT_ID','P_Ca_SMILES','Target']]  
#     smis=df_case_g.P_Ca_SMILES
#     mols = [Chem.MolFromSmiles(smi) for smi in smis]


#     # create descriptor calculator with all descriptors
#     calc = Calculator(descriptors, ignore_3D=True)
#     # calculate multiple molecule

#     # as pandas
#     df_mod = calc.pandas(mols)
#     df_mod_x = df_mod.apply(pd.to_numeric, errors='coerce')
#     df_mod_x.fillna(0, inplace=True)
#     df_smile=df_case_g
#     fp_class=['ecfp4']
#     header = [f'{fp_class[0]}_' + str(i) for i in range(1024)]
#     fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
#     fp_df=pd.DataFrame(np.array(fp),columns=header)
#     fp_df=pd.concat([df_smile,fp_df],axis=1)
#     fp_df=network_case.main(fp_df,fp_class)
#     df_case1 = pd.concat([fp_df,df_mod_x], axis=1)
#     df_case1.fillna(0, inplace=True)
#     return df_case1

# df_test = data_transform(df_test)
# df_train=data_transform(df_train)
# df_valid=data_transform(df_valid)


# df_test.to_csv('../data/fp/df_test.csv',index=False)
# df_train.to_csv('../data/fp/df_train.csv',index=False)
# df_valid.to_csv('../data/fp/df_valid.csv',index=False)

df_test = pd.read_csv('../data/fp/df_test.csv')
df_train=pd.read_csv('../data/fp/df_train.csv')
df_valid=pd.read_csv('../data/fp/df_valid.csv')
# rf_train_case.main(df_train,df_valid,df_test)

xgbost_train_case.main(df_train,df_valid,df_test)


