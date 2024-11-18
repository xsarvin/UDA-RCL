# UDA-RCL
Artifacts accompanying the paper UDA-RCL

# Data
Our source data is available at https://www.aiops.cn/%e5%a4%9a%e6%a8%a1%e6%80%81%e6%95%b0%e6%8d%ae/

we also release our pre-processed data in AIops2021 and AIops2022 folders, respectively.

# Dependencies
```
pip install -r requirements.txt
```



# Run
###### For supervised learning scenario: 

```
python main.py --method uda_rca --s_dataset AIops2021 --t_dataset AIops2021 --batch_size 10 --embed_dim 5 --hidden_dim 5 --beta 1
```

###### For transfer learning scenario:

```
python main_trans.py --method uda_rca --s_dataset AIops2022 --t_dataset AIops2021 --batch_size 45 --embed_dim 5 --hidden_dim 5 --beta 1
```

# Architecture
![image](https://github.com/user-attachments/assets/7f41f145-f7d4-4707-b73e-9d3658ba083a)



# Contact us
Any questions can leave messages in "Issues"!
