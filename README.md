# allmycurrentcode


To run experiments on SBM -

```Python
cd SBM/
### for running the baseline node classification ###
python RunSBMClass.py --path 'Dataset/50Nodes/SBM_75_0.3_0.1_1e-20.pt' --csvout SBMResults2.csv
```

```Python
cd SBM/
### for pruning edges to maximize the spectral gap + node classification ###
python RunSBMMaximize.py --path 'Dataset/50Nodes/SBM_75_0.3_0.1_1e-20.pt' --csvout SBMResultsMax.csv
```

```Python
cd SBM/
### for pruning edges to minimize the spectral gap + node classification ###
python RunSBMMinimize.py --path 'Dataset/50Nodes/SBM_50_0.3_0.02_1e-10.pt' --csvout SBMMin.csv
```




