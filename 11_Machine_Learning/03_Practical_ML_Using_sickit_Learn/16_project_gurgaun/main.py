import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

# 1. Load dataset
housing = pd.read_csv('housing.csv')

# 2. Create a stratified test set
housing['income_cat'] = pd.cut(housing['median_income'],
                                 bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                 labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis=1)

# we will work on copy of training set
housing = strat_train_set.copy()

# 3. separate features and labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis=1)

# print(housing, housing_labels)

# 4. List the numerical and categorical columns
num_attributes = housing.drop('ocean_proximity', axis=1).columns.tolist()
cat_attributes = ['ocean_proximity']

# 5 lets make a pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

# for categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown='ignore'))
])

# Construct full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
])

# 6. transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)


# 7.Train the model

#LinearRegression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
print(f"the root mean squared error for Linear Regression is: {lin_rmse}")

# DecisionTreeRegressor model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
print(f"the root mean squared error for Decision Tree Regressor is: {dec_rmse}")

# RandomForestRegressor model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_preds)
print(f"the root mean squared error for Random Forest Regressor is: {random_forest_rmse}")