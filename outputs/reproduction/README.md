## Reproduction

### Environments
+ Hardware
```bash
CPU: Intel® Xeon(R) W-2295 CPU (64G)
GPU: NVIDIA GeForce RTX 3080 (10G)
```
+ Software
```bash
Python 3.8.13
Numpy 1.23.0
Pytorch 1.8.1(GPU Version)
```

### Configurations

#### Beauty
```shell script
python main.py \
--load_model reproduction/DLFS-Rec-Beauty \
--do_eval
```
#### Sports_and_Outdoors
```shell script
python main.py \
--data_dir ./data/Sports_and_Outdoors/ \
--data_name Sports_and_Outdoors \
--num_hidden_layers 3 \
--hidden_dropout_prob 0.3 \
--load_model reproduction/DLFS-Rec-Sports_and_Outdoors \
--do_eval
```

#### Toys_and_Games
```shell script
python main.py \
--data_dir ./data/Toys_and_Games/ \
--data_name Toys_and_Games \
--num_hidden_layers 3 \
--hidden_dropout_prob 0.3 \
--load_model reproduction/DLFS-Rec-Toys_and_Games \
--do_eval
```

#### Home_and_Kitchen
```shell script
python main.py \
--data_dir ./data/Home_and_Kitchen/ \
--data_name Home_and_Kitchen \
--load_model reproduction/DLFS-Rec-Home_and_Kitchen \
--do_eval
```


