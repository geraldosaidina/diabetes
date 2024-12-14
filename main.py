import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# reading the data from the CSV file and displaying it
diabetes = pd.read_csv('C:/Users/Geraldo Junior/Documents/Python/ML/diabetes/diabetes.csv')

print(diabetes.head())

# cleaning the data by normalizing the columns so that the values can be between 0 and 1 to avoid skewing the results with a lambda function
diabetes.columns

cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# 'explaining' to TensorFlow what type of data it's dealing with, numeric in all of these cases
num_preg = tf.feature_column.numeric_column('number of pregnancies')
glucose = tf.feature_column.numeric_column('glucose level')
blood_pressure = tf.feature_column.numeric_column('blood pressure')
skin_thickness = tf.feature_column.numeric_column('skin thickness')
insulin = tf.feature_column.numeric_column('insulin level')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

#plotting a histogram with matplotlib, pandas automatically looks for it
diabetes['Age'].hist(bins=20)

#bucketizing the age atrribute
age_buckets = tf.feature_column.bucketized_column(age, boundaries = [20, 30, 40, 50, 60, 70])
feat_cols = [num_preg, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

#making the split between training and testing data
x_data = diabetes.drop('Outcome', axis=1)
labels = diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)

#creating and training the model using the inout function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func, steps=1000)

#making the predictions using test data
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
list(predictions)

#evaluating the model's performance
eval_input_func = tf.estimator.input.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)
print(results)