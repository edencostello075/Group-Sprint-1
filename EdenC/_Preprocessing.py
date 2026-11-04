#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install ucimlrepo')


# In[7]:


from ucimlrepo import fetch_ucirepo 

# fetch dataset 
myocardial_infarction_complications = fetch_ucirepo(id=579) 

# data (as pandas dataframes) 
X = myocardial_infarction_complications.data.features 
y=myocardial_infarction_complications.data.targets



# In[8]:


from IPython.display import display

# Preview the first five rows
display(X.head())
display(y.head())

# Show dataset dimensions: (number of rows, number of columns)
print("Features shape:", X.shape)
print("Targets shape:", y.shape)

# Show variable descriptions
display(myocardial_infarction_complications.variables)


# In[9]:


import pandas as pd
from IPython.display import display

missing_summary = pd.DataFrame({
    "missing_count": X.isna().sum(),
    "missing_percent": X.isna().mean() * 100
}).sort_values("missing_percent", ascending=False)

display(missing_summary.loc[missing_summary["missing_count"] > 0])

print("Total patients:", len(X))
print("Patients with no missing features:", X.notna().all(axis=1).sum())
print("Completely missing features:", X.columns[X.isna().all()].tolist())


# In[10]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Use the binary mortality target
if isinstance(y, pd.DataFrame):
    y = (y["LET_IS"] != 0).astype(int)

# 60% training, 40% temporary
train_idx, remaining_idx = train_test_split(
    np.arange(len(X)), test_size=0.40,
    random_state=42, stratify=y
)

# Divide the remaining 40% equally
val_idx, test_idx = train_test_split(
    remaining_idx, test_size=0.50,
    random_state=42, stratify=y.iloc[remaining_idx]
)

X_train = X.iloc[train_idx].copy()
X_val = X.iloc[val_idx].copy()
X_test = X.iloc[test_idx].copy()

# Apply the missingness cutoff using training data only
keep = X_train.columns[X_train.isna().mean() <= 0.50]
keep = keep.drop("ID", errors="ignore")

X_train, X_val, X_test = [
    data.loc[:, keep] for data in (X_train, X_val, X_test)
]

for name, features, indices in [
    ("training", X_train, train_idx),
    ("validation", X_val, val_idx),
    ("test", X_test, test_idx),
]:
    data = features.copy()
    data["mortality"] = y.iloc[indices].to_numpy()
    data.to_csv(f"{name}_data.csv", index=False)


# In[11]:


y.head()


# In[ ]:





# In[ ]:




