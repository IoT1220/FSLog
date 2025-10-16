# FSLog

FSLog is a novel federated split learning (FSL) framework. It enables collaborative training while ensuring data privacy and reducing the computational load on the client. It not only significantly lowers local computational load and privacy leakage but also preserves high diagnostic accuracy.

## 🔍 Key Features

- **FL + SL Integration**: Collaborative training while ensuring data privacy and reducing the computational load on the client.
- **Multilayer privacy protection mechanism**: Combining feature filtering and irreversible protection techniques to effectively reduce reconstruction success and safeguard data privacy.
- **A model-splitting (SL-BERT）**: With the server sharing the backbone of the resource-intensive transformer and clients maintaining lightweight embeddings and personalized heads to reduce local computing power load.
- **The personalized local model via FedAvg and EMA (FL-EMA)**: Integrate component-driven log clustering with personalized fine-tuning to enhances model’s generalization and adaptability in heterogeneous environments.
- **Modular & Extensible**: Containerized deployment demonstrates enables rapid fault localization and recovery, effectively protecting enterprise data.

## 📄 Dataset Description
### This study evaluates two datasets:

  - **The publicly available Aliyun dataset:** link at https://tianchi.aliyun.com/competition/entrance/531947/information.  

  - **The Privacy unavailable CMCC datasets:** a proprietary dataset licensed from an industry partner, which cannot be publicly released. Access to the proprietary dataset requires authorization from the provider and a signed data‑use agreement.  

### Data storage and load:
  **dataset is divided into five parts, each representing the log data of an client-server, as shown in the following three files:**

**1.The result of the log sequence after being vectorized by BERT**
```bash
data_{}.npy
```
**2.Semi-supervised labels, where -1 indicates no label**
```bash
 semi_label_{}.npy
```
**3.The label of the original data source**
```bash
 label_{}.npy
```

## 📁 Icore code 

1. **Data Loading Class**
```bash
data_loader.py
```

2. **Data loading, adapting to the docker simulation environment**
```bash
data_loader_docker.py
```

3. **Entry function for the FSLog in the docker environment**
```bash
docker_fed_bert_main_semi-supervised.py
```

4. **Function for serial simulation FSLog entry**
```bash
fed_split_main.py
```

5. **FSLog Model Class**
```bash
transfeomer_fed_bert.py
```

## 📦 Installation

```
conda create --name <env> --file requirements.txt
```

## 🚀 Quick Start

```bash


# docker simulation
docker network create --subnet 192.168.0.0/16 --gateway 192.168.0.1 fednet
docker pull pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

docker run -it --gpus all -v xxx:/workspace --net fednet -p 12000 --name fed0 pytorch /bin/bash
docker run -it --gpus all -v xxx:/workspace --net fednet -p 12000 --name fed0 pytorch /bin/bash
docker run -it --gpus all -v xxx:/workspace --net fednet -p 12000 --name fed1 pytorch /bin/bash
docker run -it --gpus all -v xxx:/workspace --net fednet -p 12000 --name fed2 pytorch /bin/bash
docker run -it --gpus all -v xxx:/workspace --net fednet -p 12000 --name fed3 pytorch /bin/bash
docker run -it --gpus all -v xxx:/workspace --net fednet -p 12000 --name fed4 pytorch /bin/bash
docker run -it --gpus all -v xxx:/workspace --net fednet -p 12000 --name fed5 pytorch /bin/bash


# Serial simulation experiment
cd code
python fed_split_main.py


# Docker simulation experiment
cd code
export MASTER_ADDR=192.168.0.2
export MASTER_PORT=8888
export WORLD_SIZE=3
export RANK=xx
python docker_fed_bert_main_semi-supervised.py
```



## 📁 Project Structure
```
FSLog/
├── code/               # Icore code (SL-Bert, FL-EMA, docker)
├── data/               # Input logs
├── requirements/       # Create an environment
└── README.md           # Project description
```



## 🔗 Links
- [Code](https://github.com/SANER26-Submission-81/FSLog)

