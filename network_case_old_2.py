import os
import threading
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
import argparse
from multiprocessing import Pool
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
from pyinstrument import Profiler
import concurrent.futures
profiler = Profiler()
os.environ['NETWORKX_AUTOMATIC_BACKENDS'] = 'cugraph'


def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def process_single_network(t):

    print("process", t)
    sim = pd.read_csv('SimMatrix.csv')
    fp_class = 'ecfp4'
    n= sim.shape[0]

    print("p_class,sim,t, n", fp_class,sim.shape,t, n)

    single_network(fp_class,sim,t, n)

def load_and_process_file(t):
    with open(f'network_matrix_{t}.pkl', 'rb') as file:
        network_matrix = pickle.load(file)
    return network_matrix


def single_network(fp_class,SimMatrix,threshold,n):
    start_time = time.time()

    g = np.array(SimMatrix>=threshold,dtype=int)
    row, col = np.diag_indices_from(g)
    g[row, col] = 0
    G = nx.from_numpy_array(g)

    ########################################################
    ### generate network parameter metrics #################
    ########################################################

    ### Centrality in networks
    #3.7.1 Degree
    try:
        dc = nx.degree_centrality(G, backend='cugraph')
        dc = dict(sorted(dc.items()))
    except:
        dc = nx.degree_centrality(G)
        dc = dict(sorted(dc.items()))

#     #3.7.2 Eigenvector
    try:
        eic = nx.eigenvector_centrality(G,max_iter=100,backend='cugraph')
        eic = dict(sorted(eic.items()))
    except:
        eic = dict(zip(range(n),np.repeat(np.nan,n)))
    #3.7.3 Closeness
    cc = nx.closeness_centrality(G)
    cc = dict(sorted(cc.items()))


    #3.7.5 (Shortest Path) Betweenness
    bc = nx.betweenness_centrality(G,backend='cugraph')
    bc = dict(sorted(bc.items()))

    #3.7.9 Load
    lc = nx.load_centrality(G)
    lc = dict(sorted(lc.items()))

    #3.7.10 Subgraph
    # sc = nx.subgraph_centrality(G)
    # sc = dict(sorted(sc.items()))

    #3.7.11 Harmonic Centrality
    hc = nx.harmonic_centrality(G)
    hc = dict(sorted(hc.items()))

    # k-core
    cn = nx.core_number(G,backend='cugraph')
    cn = dict(sorted(cn.items()))


    ol = nx.onion_layers(G)
    ol = dict(sorted(ol.items()))

    try:
        pr = nx.pagerank(G,backend='cugraph')
        pr = dict(sorted(pr.items()))
    except:
        pr = dict(zip(range(n),np.repeat(np.nan,n)))


    #clustering for each nodes
    clu = nx.clustering(G,backend='cugraph')
    clu = dict(sorted(clu.items()))

    #################################
    #network info
    #desity for the whole network
    density = nx.density(G)
    d_density = dict(zip(range(n),np.repeat(density,n)))

    ########################################################
    ### generate network info matrix #######################
    ########################################################

    network_matrix = pd.DataFrame({f'{fp_class}_{str(int(threshold*100))}_DC':dc,
                                   f'{fp_class}_{str(int(threshold*100))}_EIC':eic,
                                   f'{fp_class}_{str(int(threshold*100))}_CC':cc,
                                   f'{fp_class}_{str(int(threshold*100))}_BC':bc,
                                   f'{fp_class}_{str(int(threshold*100))}_LC':lc,
                                   f'{fp_class}_{str(int(threshold*100))}_HC':hc,
                                   f'{fp_class}_{str(int(threshold*100))}_CN':cn,
                                   f'{fp_class}_{str(int(threshold*100))}_OL':ol,
                                   f'{fp_class}_{str(int(threshold*100))}_PR':pr,
                                   f'{fp_class}_{str(int(threshold*100))}_Clustering':clu,
                                   f'{fp_class}_{str(int(threshold*100))}_Density':d_density,
                                   })

    # write network_matrix into pickle file
    with open(f'network_matrix_{threshold}.pkl', 'wb') as file:
            pickle.dump(network_matrix, file)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"[Sinlge] The code took {execution_time} seconds to run")

    return network_matrix

def single_patent_network_generation(fps,fp_class,tlist):

    ########################################################
    ### generate similarity matrix #########################
    ########################################################
    n = len(fps)
    print(n)
    similarity_matrix = []

    if fp_class == 'mol2vec':
        for i in range(n):
            tsims = [cosine_similarity(fps[i], fps[j]) for j in range(n)]
            similarity_matrix.append(tsims)
    else:
        fpstr = [list(map(lambda x:str(x),np.int64(f))) for f in fps]
        fpstring = [''.join(f) for f in fpstr]
        fps_rdkit = [cDataStructs.CreateFromBitString(f) for f in fpstring]
        for i in range(n):
            #maybe you will have error from here
            tsims = Chem.DataStructs.BulkTanimotoSimilarity(fps_rdkit[i], fps_rdkit)
            similarity_matrix.append(tsims)
    SimMatrix = pd.DataFrame(similarity_matrix)
    print("SimMatrix",SimMatrix.shape)

    SimMatrix.to_csv('SimMatrix.csv', index=False)
    ########################################################
    ### generate graph of network ##########################
    ########################################################

    NetworkMatrix = pd.DataFrame()
    
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        print("Running with multiprocessing...")
        # The map method is used in the same way as with ThreadPoolExecutor
        results = list(executor.map(process_single_network, tlist))

    with concurrent.futures.ProcessPoolExecutor()as executor:

        results = list(executor.map(load_and_process_file, tlist))
  
    NetworkMatrix = pd.concat(results, axis=1)

    end_time = time.time()  
    execution_time = end_time - start_time  
    print(f"The code took {execution_time} seconds to run")  

    network_matrix = pd.DataFrame()
    SimMatrix = pd.DataFrame()
    return NetworkMatrix


def Network_Generation(df,fp_class,tlist,plist):

    df_aimed = df.copy()
    df = pd.DataFrame()
    flag = True
    p_id = None

    df_aimed_network = None
    if set(['PATENT_ID','P_Ca_SMILES']).issubset(df_aimed.columns):
        p_id = df_aimed.PATENT_ID[0]
        if len(plist) == 0:
            pass
        else:
            for i in plist:
                if p_id in i:
                    flag = False
                    break
                else:
                    pass

        print(p_id,df_aimed.shape,flag)

        if flag == True:
            try:
                print(p_id,df_aimed.shape)
                df_aimed_fps = df_aimed.drop(columns=['PATENT_ID','P_Ca_SMILES'])
                fps = np.array(df_aimed_fps)
                patent_id = df_aimed.PATENT_ID
                smis = df_aimed.P_Ca_SMILES
                df_aimed_fps = pd.DataFrame()

                print('start')
                NetworkMatrix = single_patent_network_generation(fps=fps,fp_class=fp_class,tlist=tlist)
                df_aimed_network = NetworkMatrix.copy()
                df_aimed_network.insert(0,'PATENT_ID',patent_id)
                df_aimed_network.insert(1,'P_Ca_SMILES',smis)

                NetworkMatrix = pd.DataFrame()
                print(f'New:{p_id}')
            except Exception as e:
                print(e.args)
                print(str(e))
        else:
            print('Exist')
            pass
    else:
        print(df_aimed.shape)
        df_aimed_fps = df_aimed.drop(columns=['P_Ca_SMILES'])
        fps = np.array(df_aimed_fps)
        smis = df_aimed.P_Ca_SMILES
        df_aimed_fps = pd.DataFrame()

        print('start')
        NetworkMatrix = single_patent_network_generation(fps=fps,fp_class=fp_class,tlist=tlist)
        df_aimed_network = NetworkMatrix.copy()
        df_aimed_network.insert(0,'P_Ca_SMILES',smis)

        NetworkMatrix = pd.DataFrame()
        print(f'New:{p_id}')
        
    return df_aimed_network



def main(df_first,f_class):

    parser = argparse.ArgumentParser(description='network construction')
    parser.add_argument('--p',default=4,type=int,help="number of pools to use")
    parser.add_argument('--fp_class',default='ecfp4',type=str,help="fingerprint type")
    # parser.add_argument('--threshold',default=0.7,type=float,help="threshold for network construction")

    args = parser.parse_args()
    pool = args.p
    # # fp_class = args.fp_class
    # f_class = ['ecfp4']
    print(f_class)
    # path = './'
    for f in f_class:
        plist = []
        df_total = df_first
        ff_df = []
        c = 0
        if set(['PATENT_ID','P_Ca_SMILES']).issubset(df_first.columns):       
            print('df_total.PATENT_ID.unique()')
            print(df_total.PATENT_ID.unique())
            for patent in df_total.PATENT_ID.unique():
                if patent not in plist:
                    df_temp = df_total[df_total.PATENT_ID == patent].reset_index(drop=True)
                    ff_df.append(df_temp)
                    c +=1
        else:
            print('single list input')
            ff_df.append(df_total) 
        df_total = pd.DataFrame()

        tlist = np.linspace(0.4,0.9,11)  # threshold for each graph/network
        print('tlist',tlist)
        results = []
        print("ff_df_len", len(ff_df))
        for f_df in ff_df:
            result=Network_Generation(f_df,f,tlist,plist)
            #r_ = p.apply_async(result)
            results.append(result)
           
        print('start combine')
 
        df_all = pd.DataFrame()
        c = 0
        for l in results:
     
                c +=1
                df_temp = l
                df_temp['PATENT_NUM'] = np.repeat(df_temp.shape[0],df_temp.shape[0])
                df_all = pd.concat([df_all,df_temp]).reset_index(drop=True)

        return df_all

def handelSmi(smi):
    try:
        return Chem.MolFromSmiles(smi, sanitize=True)
    except:
        return Chem.MolFromSmiles("C", sanitize=True)

if __name__ == "__main__":
    
    profiler.start()
    dataset = pd.read_csv('WO2022060836_SMILES.csv')[:300]
    df = dataset["SMILES"].tolist()
    print(df)
    print("network case.py")
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
    print('start.......calc')
    

    main(fp_df,['ecfp4'])
    print("network case.py")
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))