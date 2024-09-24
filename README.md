# coinstrained multi-objective molecule optimization (CMOMO)

Implementation of the method proposed in the paper "Balancing property optimization and constraint
satisfaction for constrained multi-property molecular
optimization" by Xin Xia, Yajie Zhang, Xiangxiang Zeng, Xingyi Zhang, Chunhou Zheng, Yansen Su.<sup>1</sup>

### Dependencies
- [cddd](https://github.com/jrwnter/cddd)
Notice: You need download the pre-trained encoder-decoder CDDD model to mapping molecules between SMILES and continuous vectors. It can be load by the bash script:
```
./download_default_model.sh
```
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
- The installed environment can be downloaded from the cloud drive
  -qmocddd()

### Data Description
- data/qedplogp_test: dataset on Task1.
- data/Guacamol_sample_800: dataset on Task2.
- data/docking_test: dataset on Task3.
- data/gsk3_test: dataset on Task4.
- data/archive/: the bank library for each lead molecule on four tasks.

### File Description
- sub_code/fitness.py: The script to calculate the objectives of optimization tasks.
- sub_code/property.py: The script to calculate the molecular properties.
- sub_code/generation_rule.py: The script to generate offspring molecules.
- sub_code/selection_rule.py: The script to compare and select molecules.
- sub_code/models.py: The encoder and decoder process.
- sub_code/calc_no.py: The script to calculate the docking scores.
- sub_code/mechanism.py: Guacamol tasks.
- sub_code/nonDominationSort.py: the non-dominated relationships between molecules.

- download_default_model.sh: download the pre-trained encoder-decoder.
- environment.yml: install the environment.
- CMOMO_task1.py: optimization Task1. 
- CMOMO_task2.py: optimization Task2. 
- CMOMO_task3.py: optimization Task3. 
- CMOMO_task4.py: optimization Task4. 

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

