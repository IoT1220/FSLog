# FSLog
This study evaluates two datasets: (1) the publicly available Aliyun dataset (link in the code repository); and (2) a proprietary dataset licensed from an industry partner, which cannot be publicly released. Access to the proprietary dataset requires authorization from the provider and a signed data‑use agreement.

#Dataset Description
is divided into five parts, each representing the log data of an edge server 
data_{}.npy represents the result of the log sequence after being vectorized by BERT
semi_label_{}.npy represents semi-supervised labels, where -1 indicates no label
label_{}.npy represents the label of the original data source

Introduction to code
data_loader.py 
Data Loading Class
data_loader_docker.py 
Data loading, adapting to the docker simulation environment
docker_fed_bert_main_semi-supervised.py 
Entry function for the FSLog in the docker environment
fed_split_main.py
Function for serial simulation FSLog entry
transfeomer_fed_bert.py 
FSLog Model Clas
