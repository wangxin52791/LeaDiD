FROM python:3.10-slim-buster

# Set the work directory of PandaGradio
RUN mkdir -p /var/leadid_inference_gradio/temp
WORKDIR /var/leadid_inference_gradio

# ADD ./sources.list /etc/apt/
# RUN apt-get update && apt-get install -y poppler-utils

ENV TZ "Asia/Shanghai"
ENV DEBIAN_FRONTEND noninteractive
ENV PIP_NO_CACHE_DIR off

# RUN pip install  --upgrade pip -i http://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
# RUN pip install --upgrade pip

RUN pip install cupy-cuda11x -i http://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
RUN pip install --progress-bar off --no-cache-dir --extra-index-url http://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url http://pypi.nvidia.com nx-cugraph-cu11 --trusted-host pypi.nvidia.com --trusted-host pypi.tuna.tsinghua.edu.cn

#RUN pip install  nx-cugraph-cu11  -i http://pypi.nvidia.com --trusted-host pypi.nvidia.com

# RUN ./test.sh
COPY ./requirements.txt /var/leadid_inference_gradio/temp/requirements.txt
RUN pip install -r /var/leadid_inference_gradio/temp/requirements.txt --no-deps -i http://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
LABEL org.opencontainers.image.authors="wang xin"
ADD ./ /var/leadid_inference_gradio/
CMD ["bash","/var/leadid_inference_gradio/test.sh"]

