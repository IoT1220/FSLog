# FSLog

FSLog is a synergistic framework for log-based root cause analysis (RCA), combining the semantic reasoning power of large language models (LLMs) with the structure and domain specificity of knowledge graphs (KGs). Designed for modern large-scale distributed systems, it provides an accurate, explainable, and efficient way to identify system faults.

## 🔍 Key Features

- **LLM + KG Integration**: Combine context-aware log understanding with structured fault knowledge.
- **Semantic Entity Aggregation**: Normalize redundant or similar fault indicators via embedding-based clustering.
- **Context-Aware Retrieval**: Dynamically recall relevant fault entities from the KG based on summarized logs.
- **Prompt-Driven RCA**: Construct powerful prompts to guide LLMs in accurate fault reasoning.
- **Modular & Extensible**: Fully script-based pipeline with CLI tools for preprocessing, KG construction, RCA, and evaluation.

## 📄 Dataset Description
**This study evaluates two datasets:**
**The publicly available Aliyun dataset:** link at https://tianchi.aliyun.com/competition/entrance/531947/information.
**The Privacy unavailable CMCC datasets:** a proprietary dataset licensed from an industry partner, which cannot be publicly released. Access to the proprietary dataset requires authorization from the provider and a signed data‑use agreement.
**Data storage and load:** dataset is divided into five parts, each representing the log data of an client-server.
  
1. **the result of the log sequence after being vectorized by BERT**
```bash
data_{}.npy
```
2. **semi-supervised labels, where -1 indicates no label**
```bash
 semi_label_{}.npy
```
3. **the label of the original data source**
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

```bash
pip install .
```

## 🚀 Quick Start



1. **Preprocess Logs**
```bash
aetherlog-preprocess --input data/raw_logs.json --output data/summary.json
```

2. **Data Loading Class**
    data_loader.py
```bash
aetherlog-buildkg
```


2. **Build split bert**
```bash
aetherlog-buildkg
```

3. **Recall Entities**
```bash
aetherlog-recall --log data/summary.json --entity data/kg.json --output data/recalled.json
```

4. **Construct RCA Prompt**
```bash
aetherlog-prompt --summary data/summary.json --entity data/recalled.json --output data/prompt.json
```

5. **Run RCA Analysis**
```bash
aetherlog-rca --log data/summary.json --kg data/kg.json --out data/result.json
```

6. **Evaluate Performance**
```bash
aetherlog-eval --pred data/result.json --gold data/groundtruth.json
```

## 📁 Project Structure
```
FSLog/
├── scripts/            # Main RCA pipeline scripts
├── src/                # Core modules (LLM interface, KG, model)
├── data/               # Input logs, KG and results
├── configs/            # YAML configuration files
├── setup.py            # Install and entry points
└── README.md           # Project description
```



## 🔗 Links
- [Code](https://github.com/SANER26-Submission-81)
