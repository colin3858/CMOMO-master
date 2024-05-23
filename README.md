# coinstrained multi-objective molecule optimization (CMOMO)

Implementation of the method proposed in the paper "A pareto-based deep evolutionary algorithm for constrained 
multi-property molecular optimization" by Xin Xia, Yajie Zhang, Xiangxiang Zeng, Xingyi Zhang, Chunhou Zheng, Yansen Su.<sup>1</sup>

### Dependencies
- [cddd](https://github.com/jrwnter/cddd)
Download the pre-trained CDDD model using the bash script:
```
./download_default_model.sh
- python=3.6
  - rdkit
  - pytorch=1.4.0
  - cudatoolkit=10.0
  - tensorboardX
  - pip>=19.1,<20.3
  - pip:
    - molsets
    - cddd
```

### Installing
```
cd CMOMO
pip install .
```
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

