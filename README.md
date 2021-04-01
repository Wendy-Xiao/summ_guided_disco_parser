# Predicting Discourse Trees from Summarizers
The official code for the NAACL paper: Predicting Discourse Trees from pre-trained  Transformer-based Neural Summarizers

## Get Started
python==3.6.7

transformers==4.3.2

torch==1.6.0

## Usage
### 1. Dowload the Pre-trained Summarizer 
The link to the models we used in this paper: [link]()
* You can also train your own summarizer by simply use the code [here]() by using self-attention as attention type.
### 2. Download the datasets
As all the three datasets are not open source, you can find the datasets here:
a. RST-DT: 

b: Instructional: 

c: GUM: 

### 3. Pre-process the datasets
Before you can use the summarization model to parse the documents, you need first pre-process the datasets as the format shown in the example.
### 4. Generate and save Ground-truth trees
Use build_gt_tree.py to generate the ground-truth constituency tree and dependency trees
### 5. Parsing
#### Constituency Parsing
Simply specify the number of layers and heads, as well as the model name and gpu to use.
```
python3 main_const.py -n_layers 6 -n_head 8 -device 0 -model_name cnndm-6-8
```
#### Dependency Parsing
Simply specify the number of layers and heads, as well as the model name and gpu to use.
```
python3 main_dep.py -n_layers 6 -n_head 8 -device 0 -model_name cnndm-6-8
```
#### Baseline
```
python3 main_bsl.py
```

## Cite


