# MST-VAE
### Multi-Scale Temporal Variational Autoencoder for Anomaly Detection in Multivariate Time Series

MST-VAE is an unsupervised learning approach for anomaly detection in multivariate time series. Inspired by InterFusion paper, we propose a simple yet effective multi-scale convolution kernels applied in Variational Autoencoder. 

Main techniques in this paper:
- Multi-scale module: short-scale and long-scale module
- We adopted Beta-VAE for training the model
- MCMC is applied to achieve better latent representations while detecting anomalies

## How to use the repository
### Clone the repository
```bash
git clone https://github.com/tuananhphamds/MST-VAE.git
cd MST-VAE
```

### Prepare experiment environment (GPU is needed)
1. Install Anaconda version 4.11.0
2. Create an environment with Python 3.6, Tensorflow-gpu=1.12.0 and dependencies
```
conda create -n mstvae python=3.6 tensorflow-gpu=1.12.0
conda activate mstvae
pip install -r requirements.txt
```

### Prepare data
In this study, we use five public datasets: ASD (Application Server Dataset), SMD (Server Machine Dataset), PSM (Pooled Server Metrics), SWaT (Secure Water Treatment), and WADI (Water Distribution).

ASD, SMD, PSM can be refered in ``MST-VAE/data/processed`` folder.

SWaT, WADI should be requested from https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

Each dataset must contain three files to be compatible with the code:
- <Name_of_dataset>_train.pkl: train data
- <Name_of_dataset>_test.pkl: test data
- <Name_of_dataset>_test_label.pkl: test label

For all detailed information of datasets, refer to ``MST-VAE/data_info.txt``
___
After requesting dataset SWaT & WADI, an email will be replied from iTrust website.

**For SWaT dataset**: 
download two files ``SWaT_Dataset_Attack_v0.csv`` and ``SWaT_Dataset_Normal_v0.csv`` in folder ``SWaT.A1 & A2_Dec 2015/Physical``

**For WADI dataset**:
download three files ``WADI_attackdata.csv``, ``WADI_14days.csv``, and ``table_WADI.pdf`` in folder ``WADI.A1_9 Oct 2017``.

```bash
Move all downloaded files to folder MST-VAE/explib
```
Make sure current working directory is MST-VAE/explib, run:
```bash
python raw_data_converter.py
```

These following files will be generated: 
- SWaT_train.pkl
- SWaT_test.pkl
- SWaT_test_label.pkl
- WADI_train.pkl
- WADI_test.pkl
- WADI_test_label.pkl (not available): should be created using the description file ``table_WADI.pdf``

Move all the aboved files to ``MST-VAE/data/processed``
### Run the code
Choose the dataset that you want to run and add it to the ``datasets`` list on ``line 22, file algorithm/run_experiment.py``
```bash
datasets = [ <dataset_name>,]

Ex: 
- one dataset: datasets = ['PSM']
- multiple datasets: datasets = ['omi-1', 'omi-2', 'machine-1-1', 'PSM']
```
You can modify ``REPEATED_NUMS`` for number of repeated experiments on ``line 21, file algorithm/run_experiment.py``


Move to folder MST-VAE, run the following command:
```bash
python algorithm/run_experiment.py
```

The training config can be modified at ``algorithm/train_config.json``

### Error
The following error might occur in Windows: 

``File "C:\Users\<username>\anaconda3\envs\mstvae\lib\site-packages\tfsnippet\utils\random.py", line 16, in generate_random_seed
    return np.random.randint(0xffffffff) ValueError: high is out of bounds for int32`` 

Open random.py file as shown in the error

Change ``np.random.randint(0xffffffff)`` -> ``np.random.randint(0x0000ffff)`` at ``line 16`` and save it.

### Run your own data
1. Put your own data (three files) in ``data/processed``: ``<Name_of_dataset>_train.pkl, <Name_of_dataset>_test.pkl, <Name_of_dataset>_test_label.pkl``
2. Add your data dimension (number of metrics) in function ``get_data_dim on line 75, file algorithm/utils.py``
```bash
elif dataset == '<Name_of_dataset>':
    return <dim>
```
3. Add train config for running your data, you can copy and modify an example in ``algorithm/train_config.json`` 
4. Do the steps in **Run the code**

### Check experiment results
Evaluation results are stored as JSON files in ``MST-VAE/experiment_results``
