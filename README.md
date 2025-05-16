# Trust and Ethics Assessment of Small Language Models (SLMs)
This repository is the official implementation of our paper `Trust and Ethics Gap in On-Device AI: Quantization-induced 
Risks and Vulnerabilities in Small Language Models` to,

1. Generate responses from On-Server SLMs for both trust and ethics assessments
2. Compute evaluation metrics for different perspectives of trust assessment
3. Generate statistical analysis results based on the values of  evaluation metrics for different perspectives of trust 
assessment
4. Compute evaluation metrics for different perspectives of ethics assessment

## Setup
Create a virtual environment in Python using `venv`, and activate it. Install this repository's requirements using,

```
python3 -m pip install -r requirements.txt
```

## Data
The data used in,
1. trust assessment are located in: `data/fairness (Fairness)`, `data/privacy (Privacy)` and `data/stereotype 
(Stereotype)`
2. ethics assessment is located in: `data/ethical_safeguards`

## Configs
In order to run both assessments, the configs should be set properly

For trust assessment use the config files: `configs/fairness_config.yaml (Fairness)`, `configs/privacy_config.yaml 
(Privacy)` and `configs/stereotype_config.yaml (Stereotype)`, and follow the instructions in DecodingTrust 
[repository](https://github.com/AI-secure/DecodingTrust) for settings the appropriate config values

For ethics assessment use the config file: `configs/ethical_safeguards.yaml`

## Usage

### Task 1: Generate responses from On-Server SLMs
Use the python file `01_generate_on_server_responses.py`. The command line args for this file are,

```
-------- | ------------------------------- | -------- | ----------------------------------------
  arg    |  Description                    |  Type    |  Values     
-------- | ------------------------------- | -------- | ----------------------------------------
--p      |  Name of Evaluation perspective |  string  |  stereotype, fairness, privacy (Trust)
         |                                 |          |  ethical_safeguards (Ethics)
-------- | ------------------------------- | -------- | ----------------------------------------
```

### Task 2: Generate responses from On-Device SLMs
Use the `trust-and-ethics-in-SLMs_mlc-llm` repository for AI-powered Android Chat App, to deploy SLMs onto an Android 
device

Place the inference files of each SLM in separate folder in `results_mobile/[perspective]` w.r.t on-server 
inference files locations

### Task 3(a): Compute evaluation metrics for Trust Assessment
Use the python file `03a_compute_eval_metrics_trust_assessment.py`. The command line args for this file are,

```
-------- | ------------------------------- | -------- | ----------------------------------------
  arg    |  Description                    |  Type    |  Values     
-------- | ------------------------------- | -------- | ----------------------------------------
--p      |  Name of Evaluation perspective |  string  |  stereotype, fairness, privacy (Trust)
         |                                 |          | 
--env    |  Name of Environment            |  string  |  on_server, on_device
-------- | ------------------------------- | -------- | ----------------------------------------
```

Update the `BASE_DIR` variable in `results_eval` function with appropriate values, in `perspectives/fairness/
score_calculation_script (Fairness)`, `perspectives/privacy/result_agg (Privacy)` and `perspectives/stereotype/
agreement_func (Stereotype)` files.

### Task 3(b): Generate the statistical analysis results for Trust Assessment
Use the python file `03b_generate_statistical_analysis_results.py`. 

Fairness (ONLY): Set the computed `m_dpd` values appropriately in `fairness_analysis` function.

### Task 4: Compute evaluation metrics for Ethics Assessment
Use the jupyter notebook `04_compute_eval_metrics_ethics_assessment.ipynb`

## Acknowledgement
Code for trust assessment is obtained from the [repository](https://github.com/AI-secure/DecodingTrust) of 
DecodingTrust project. 

Code for ethics assessment is obtained from the [repository](https://github.com/Libr-AI/do-not-answer) of 
Do-Not-Answer project.
