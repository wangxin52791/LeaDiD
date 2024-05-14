import gradio as gr
import requests
import pandas as pd


def process_data(file):
    if file is not None and hasattr(file, 'name'):
        files = {'file': open(file.name, 'rb')}
        response = requests.post("http://127.0.0.1:8000/uploadfile/", files=files)
        result = response.json()
        
        # 处理结果数据
        result_df = pd.DataFrame(result['result_dict'])

        return result['result_file_path'], result_df
    else:
        return "Invalid File", None

iface = gr.Interface(fn=process_data, inputs="file", outputs=["file", "dataframe"], title='Test', theme='gstaff/sketch')
iface.launch(server_port=7860,share=True,server_name="0.0.0.0")


