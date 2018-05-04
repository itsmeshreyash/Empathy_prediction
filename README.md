# Empathy_prediction

## Requirements
   - Python 3
   - Numpy
   - Sklearn
   - Pandas
   - Statistics
   - Matplotlib
   - Seaborn
   
## Dataset
  - link:https://www.kaggle.com/miroslavsabo/young-people-survey/data
  - 1010 rows * 150 columns
  - 139 integer columns
  - 9 categorical columns
  
  
## Steps to run the program
  - Download the .ipynb notebook into your project folder
  - Download the dataset from the link provided into the same folder as your notebook
  - make sure all the above mentioned packages are installed.
  
  
## About the Task
**TASK:** You are working for a non-profit that is recruiting student volunteers to help with Alzheimer’s
patients. You have been tasked with predicting how suitable a person is for this task by predicting how
empathetic he or she is. Using the Young People Survey dataset, predict a person’s “empathy” on a scale
from 1 to 5. You can use any of the other attributes in the dataset to make this prediction.

## Procedure followed
### Preprocessing
  - Deal with the missing values in the dataset
  - Use Label Encoding to convert categorical columns to integer columns
  - Find correlations between all the columns with respect to the Target column.
  
### Models
#### Baseline Model
  - SVM classifier with One vs Rest method for multi-class classification
  - Accuracy = 41% 
  - Fscore = 46%
#### Proposed model
  - Use correlation values to choose the best features(top 10)
  - Use standard scaling
  - Use gridsearchCV for tuning parameters
  - Accuracy = 51%
  - Fscore = 54%
