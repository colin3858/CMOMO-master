# coinstrained multi-objective molecule optimization (CMOMO)

Implementation of the method proposed in the paper "Balancing property optimization and constraint
satisfaction for constrained multi-property molecular
optimization" by Xin Xia, Yajie Zhang, Xiangxiang Zeng, Xingyi Zhang, Chunhou Zheng, Yansen Su.<sup>1</sup>

### Dependencies
- [cddd](https://github.com/jrwnter/cddd)
Notice: You need download the pre-trained encoder-decoder CDDD model to mapping molecules between SMILES and continuous vectors. It can be load by the bash script:
```
./download_default_model.sh

### Installing
The packages need to install: 
- python=3.6
  - rdkit
  - pytorch=1.4.0
  - cudatoolkit=10.0
  - tensorboardX
  - PyTDC
  - Guacamol
  - pip>=19.1,<20.3
  - pip:
    - molsets
    - cddd
```
### The installed environment can be downloaded from the cloud drive
-cmomo-env()


### Getting Started
For Task 1, please run python CMOMO_task1.py
For Task 2, please run python CMOMO_task2.py
For Task 3, please run python CMOMO_task3.py
For Task 4, please run python CMOMO_task3.py

The output results of molecules are summarized in smi_pro_tuple, and further save in .csv file.



### Writing your own Objective Function
The fitness function can wrap any function that has following properties:
- Takes a RDKit mol object as input and returns a number as score.


## Citation

