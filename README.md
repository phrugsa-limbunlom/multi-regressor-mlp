﻿# Multi Regressor MLP
### Command to Run the Python Script
```
python src/dl_training_model_no_batch.py [file path of data set] [log file name] [input neurons size] [hidden neurons size] [output neurons size]
```
Example
```
python src/dl_training_model_no_batch.py data/ce889_dataCollection_15k.csv output/activity_hidden4.log 2 4 2
```
### Parameters:
- **file path of data set**: The path to the dataset for training.
- **log file name**: Filename where logs will be saved.
- **input neurons size**: The number of input neurons for the model.
- **hidden neurons size**: The number of hidden neurons in the model.
- **output neurons size**: The number of output neurons for the model.

### Output
![output/Loss of 15K batch.png](output/Loss of 15K batch.png)

### Resource
Presentation: [[2311569][Phrugsa]CE899-Individual Project.pptx](https://github.com/phrugsa-limbunlom/CE889_Individual_Project_MLP_from_scratch/blob/main/%5B2311569%5D%5BPhrugsa%5DCE899-Individual%20Project.pptx)
