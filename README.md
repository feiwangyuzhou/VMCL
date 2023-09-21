# VMCL

This repository contains the data and code for the paper (Findings of ACL2023):
```Contrastive Learning with Generated Representations for Inductive Knowledge Graph Embedding```

## Dataset

For inductive link prediction tasks, we use datasets: F1-F4, N1-N4 in the ```data``` folder.

Each group of inductive KGs include a source KG and a target KG, e.g., F1 is the source KG and F1_ind is the target KG. 



## Code
#### Pretrain
```bash
bash script/metatrain.sh
```
#### Finetune

```bash
bash script/finetune.sh
```
You can change ```dataset='N'``` and ```version=1``` to choose a dataset, 
and change ```kge='TransE' ``` to choose a KGE model.



## Citation
Please cite our paper if you use this code in your work.

        @inproceedings{li2023contrastive, 
        title={Contrastive Learning with Generated Representations for Inductive Knowledge Graph Embedding}, 
        author={Li, Qian and Joty, Shafiq and Wang, Daling and Feng, Shi and Zhang, Yifei and Qin, Chengwei}, 
        booktitle={Findings of the Association for Computational Linguistics: ACL 2023}, 
        pages={14273--14287}, 
        year={2023} 
        }


For any clarification, comments, or suggestions please create an issue or contact feiwangyuzhou@foxmail.com
