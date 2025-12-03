import pandas as pd
import data_module as dm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = dm.data_upload('./data/', 'processed_data.csv')
print(data.head(10))

print('Nulls values per Column')
print(data.isnull().sum())