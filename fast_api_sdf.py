from fastapi import FastAPI, File, UploadFile, Body, HTTPException
import pandas as pd
import inference_bk
import numpy as np
from pydantic import BaseModel
from rdkit import Chem
from io import BytesIO
from rdkit.Chem.Crippen import MolLogP
from typing import List
from tempfile import NamedTemporaryFile
# import time
from pyinstrument import Profiler
import json
from google_patent_scraper import scraper_class
import pandas as pd
import json

# profiler = Profiler()
app = FastAPI()

class requestSdf(BaseModel):
    Sdf: List[str]

def process_data(data):
    smilist = data["SMILES"].tolist()
    # print(smilist)
    inference_data = inference_bk.data_transform(smilist)
    result = inference_bk.inference(inference_data)
    return result

def process_csv(file):
    data_smi = pd.read_csv(file.file)
    data_smi = data_smi.drop_duplicates()
    data_smi_df = data_smi[data_smi["SMILES"].notnull()]
    result = process_data(data_smi_df)
    result_file_path = 'result_ourmodel_single.csv'
    result.to_csv(result_file_path, index=False)
    return {"result_file_path": result_file_path, "result_dict": result.to_dict(orient='records')}

def process_sdf(file):
    with NamedTemporaryFile(delete=False, suffix='.sdf') as temp:
        temp.write(file.file.read())
        temp.seek(0)
        suppl = Chem.SDMolSupplier(temp.name)
        smilist = [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]
        sdf_data = pd.DataFrame({"SMILES": smilist})
    result = process_data(sdf_data)
    result_file_path = 'result_ourmodel_single.csv'
    result.to_csv(result_file_path, index=False)
    return {"result_file_path": result_file_path, "result_dict": result.to_dict(orient='records')}



@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # start_time = time.time()  # 记录开始时间
    # profiler.start()
    if file.filename.endswith('.csv'):
        result = process_csv(file)
    elif file.filename.endswith('.sdf'):
        result = process_sdf(file)
    else:
        result = {"error": "Unsupported file format"}

    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
    return result




@app.post("/upload_sdf/")
async def upload_sdf_data(request: requestSdf):
    result_list = []
    smilist=[]
    error=[]
    error_index=[]
    result_dict_with_index=[]
    tag=0
    if not request.Sdf:
        raise HTTPException(
            status_code=404,
            detail="Sdf is empty",
            headers={"X-Error": "There goes user's error"},
        )
    for idx, sdf_data in enumerate(request.Sdf, start=1):
            with NamedTemporaryFile(delete=False, suffix='.sdf') as temp:
                temp.write(sdf_data.encode())
                temp.seek(0) 
                try:
                    suppl = Chem.SDMolSupplier(temp.name)
                    if [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]:
                        smilist.append([Chem.MolToSmiles(mol) for mol in suppl if mol is not None])
                    else:
                        error.append({"index":idx,"error_type": "Sdf file is wrong"})
                        error_index.append(idx)
                        continue
                except Exception as e:
                    error_index.append(idx)
                    error.append({"index":idx,"error_type": "Sdf file is empty " })

    sdf_data_df = pd.DataFrame({"SMILES": smilist})
    try:
        result = process_data(sdf_data_df)
    except :
        raise HTTPException(status_code=404, detail="all of the Sdf fileds are wrong")
    valid_indexes = [i for i in range(1, len(result.to_dict(orient='records'))+len(error_index)+1) if i not in error_index]
    for idx, record in enumerate(result.to_dict(orient='records')):
        if   len(error_index)>0:
            if idx==0:
                while(valid_indexes[idx]>error_index[tag]):
                    result_dict_with_index.append({"index": error_index[tag], "error":[error[tag]]}) 
                    if len(error_index)>tag+1:
                        tag+=1
                    else:
                        break
            if idx>0 and len(error_index)>0:
                while(valid_indexes[idx]>error_index[tag] and error_index[tag] > valid_indexes[idx-1]):
                    result_dict_with_index.append({"index": error_index[tag], "error":[error[tag]]})
                    if len(error_index)>tag+1:
                        tag+=1
                    else:
                        break

            result_dict_with_index.append({"index": valid_indexes[idx], **record})
            if idx==len(valid_indexes)-1:
                while (error_index[tag]>valid_indexes[len(valid_indexes)-1]) :
                    result_dict_with_index.append({"index": error_index[tag], "error":[error[tag]]})
                    if len(error_index)>tag+1:
                        tag+=1
                    else:
                        break
        else: 
            result_dict_with_index.append({"index": valid_indexes[idx], **record})
    result_list.append({"result": result_dict_with_index})
    return result_list


    
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)