# 12jynoent-2
## Installation
Use the following command to install pyKT:
Create conda envirment.

```
conda create -n pykt python=3.6
source activate pykt
```

```
cd 12jynoent-2
pip install -e .
```

## Download Datasets & Preprocess
### Download
You can download datasets we used from [pyKT](https://pykt-toolkit.readthedocs.io/en/latest/datasets.html)

### Preprocess
```
cd examples
python data_preprocess.py --dataset_name=assist2015
```

## Train & Evaluate
### Train
```
python wandb_bakt_train.py --d_ff=256 --d_model=256 --dataset_name=assist2015 --dropout=0.3 --emb_type=qid_sparseattn --final_fc_dim=64 --final_fc_dim2=64 --fold=0 --k_index=8 --learning_rate=0.0001 --model_name=bakt --n_blocks=4 --num_attn_heads=8 --seed=3407 
```

```
python wandb_bakt_train.py --d_ff=256 --d_model=64 --dataset_name=assist2015 --dropout=0.3 --emb_type=qid_accumulative_attn --final_fc_dim=256 --final_fc_dim2=256 --fold=0 --learning_rate=0.001 --model_name=bakt --n_blocks=2 --num_attn_heads=4 --save_dir=models/bakt_tiaocan_assist2015 --seed=42
```

### Evaluate
```
python wandb_predict.py --use_wandb=0 --save_dir="/path/of/the/trained/model"
```





