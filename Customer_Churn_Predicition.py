#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


#loading of excel file

input_file_path = "customer_churn_large_dataset.xlsx"

if os.path.exists(input_file_path):
    df = pd.read_excel(input_file_path) 
else:
    print("Input file not found:", input_file_path)


# In[3]:


df


# In[4]:


#Checking size of dataset, data types and basic stats of dataset

num_rows, num_cols = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")

data_types = df.dtypes
print("\nData types of columns:")
print(data_types)

numeric_stats = df.describe()
print("\nBasic statistics for numeric columns:")
print(numeric_stats)


# In[5]:


#checking for missing values to impute

missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

#But there is no null values


# In[6]:


#visualization of data distribution

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

columns_to_plot = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB', 'Churn']

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)  # 2 rows, 3 columns of subplots
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot for {column}')

plt.tight_layout()  # Ensures that subplots don't overlap
plt.show()



# In[7]:


#checking for outliers using z-score

from scipy import stats

columns_to_check = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']

# Define the Z-score threshold (e.g., 3 standard deviations)
threshold = 1.5

# Create an empty DataFrame to store outlier information
outliers_df = pd.DataFrame()

# Iterate through each column and find outliers
for column in columns_to_check:
    z_scores = stats.zscore(df[column])
    column_outliers = df[abs(z_scores) > threshold]
    column_outliers.reset_index(drop=True, inplace=True)
    column_outliers['Column_Name'] = column
    
    # Append the column's outliers to the main outlier DataFrame using .loc[]
    outliers_df = pd.concat([outliers_df, column_outliers], ignore_index=True)

print(outliers_df)


# In[8]:


outliers_df.describe()


# In[9]:


#There is no significant reasons to detect some data as outliers
#The stats of original dtatframe and 1.5 threshold outlier datframe are almost equal.
#So it is better to use the whole dataframe for the prediction of customer churn.


# In[10]:


df1 = pd.get_dummies(df, columns=['Gender', 'Location'], drop_first=True)
df1


# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df1.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[12]:


#Checking important fetaures for churn prediction

one_feature = df1['Churn']
multiple_features = df1[['Age', 'Subscription_Length_Months','Monthly_Bill', 'Total_Usage_GB', 'Gender_Male','Location_Houston', 'Location_Los Angeles', 'Location_Miami', 'Location_New York']]

# Calculate the correlation matrix between the one feature and the multiple features
correlation_matrix = multiple_features.corrwith(one_feature)

print(correlation_matrix)


# In[13]:


#According to context of customer churn prediction, Name, Customer ID features are not going to be in training and testing data.
#other features are also not linearly correlated with churn determining but it might play major by giving proper weights.


# In[14]:


#Selecting only the required columns for training and testing 
df2 = df1[['Age', 'Subscription_Length_Months', 'Gender_Male','Monthly_Bill', 'Total_Usage_GB','Location_Houston', 'Location_Los Angeles', 'Location_Miami', 'Location_New York','Churn']] #
df2


# In[15]:


#Normalization of features

scale_cols = ['Age', 'Subscription_Length_Months','Monthly_Bill', 'Total_Usage_GB'] 
# now we scling all the data 
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
df2[scale_cols] = scale.fit_transform(df2[scale_cols])


# In[16]:


df2


# In[17]:


# Assuming 'X' is your feature matrix, and 'y' is your target variable
# Split the data into training (70%), temporary (30%), and testing (15%) sets 
from sklearn.model_selection import train_test_split

X = df2[['Age', 'Subscription_Length_Months','Monthly_Bill', 'Total_Usage_GB', 'Gender_Male','Location_Houston', 'Location_Los Angeles', 'Location_Miami', 'Location_New York']]
y = df2['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



# In[18]:


import tensorflow as tf


# In[19]:


X_train.shape[1]


# In[20]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(48, activation='leaky_relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification, use sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')


# In[21]:


model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification, use sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

From this the accuracy is around 50% eventhough the tuning of hyperparameter and changing the models take place.
# In[22]:


from tensorflow.keras.models import load_model

# Assuming 'model' is your trained Keras model
model.save('churn_prediction_model.h5')


# In[ ]:


from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('churn_prediction_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from POST request
        data = request.json  # Assuming JSON input
        # Preprocess your input data as needed
        input_data = np.array(data['input'])  # Adjust as per your preprocessing
        
        # Make predictions
        predictions = model.predict(input_data)

        # You can post-process the predictions as needed
        # For binary classification (churn or not churn), you might return class labels (0 or 1).
        predicted_labels = (predictions > 0.5).astype(int)

        return jsonify({'predictions': predicted_labels.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




