FROM nvcr.io/nvidia/pytorch:20.06-py3

RUN mkdir -p /gtt
WORKDIR /gtt

COPY . /gtt

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install jupyter
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
