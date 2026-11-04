#!/usr/bin/env python
# coding: utf-8

# ## 4 Conclusion

# Here we compare the best models from all 3 sections, and look at their performance across our chosen performance metrics, by plotting a table.

# In[12]:


import pandas as pd
from IPython.display import display

files = {
    "Logistic Regression": "model_01_metrics.csv",
    "Decision Tree": "decision_tree_test_results.csv",
    "Random Forest": "RF Results Formatted.csv",
}

tables = []

for model_name, file_path in files.items():
    df = pd.read_csv(file_path)

    # Remove accidental spaces
    df.columns = df.columns.str.strip()

    df["Model"] = model_name

    tables.append(df)

# Combine the three files.
combined = pd.concat(tables, ignore_index=True)

# Combine F1 columns
if "F1-score" in combined.columns:
    if "F1" in combined.columns:
        combined["F1"] = combined["F1"].fillna(combined["F1-score"])
    else:
        combined["F1"] = combined["F1-score"]


combined = combined.drop(
    columns=["F1-score", "Metric"],
    errors="ignore",
)
combined = combined.loc[
    :, ~combined.columns.str.startswith("Unnamed:")
]


column_order = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "Balanced Accuracy",
    "ROC-AUC",
]

table = combined[
    [column for column in column_order if column in combined.columns]
].copy()

table.columns.name = None


display(
    table.style
    .format(precision=3, na_rep="—")
    .hide(axis="index")
)


table.to_csv("model_comparison.csv", index=False)


# We settled on recall as our final performance metric, given that we felt false negatives were the most important error to avoid for the context, having each achieved a high recall after selecting our models based on balanced accuracy and F1.
# We conclude that the logistic regression model is the best for this data, as it achieves a much higher recall while also achieving higher F1 and MCC, indicating a strong ability to discriminate between classes. It does perform worse on balanced accuracy, precision, and accuracy, as it is a very low threshold model, but for medical risk we believe this is worth the trade off.
# We had considered using AUC as our most important metric, which would lead us to favour the Random Forest model, but we chose not to be the ROC curve for this model (in section 4) indicated that it performs best for high-threshold tests, and we chose a low threshold test.
# The decision tree achieved the highest accuracy, but only marginally and we felt the sacrifice in recall and lower MCC were not worth it: threshold changing probably wouldn't have improved the trade off by much since the AUC is lower than the other 2 models.

# Overall, we fitted 3 successful models and gave a considered analysis of performance across different metrics, and achieved an 87% recall on a 70% accurate model. We have explored 3 different approached to the problem, focusing on data preprocessing, making good use of cross validation, and threshold changing. The models have been trained, validated and tested on the same data to achieve a fair comparison.
# 
# We feel that we were limited by not having done exploratory data analysis beforehand. For example, we were limited by not choosing to cluster the data, inspite of it having highly clustered structure, as covered in [1], but we were still able to achieve good classifiers. An extension would specify a model for each cluster, predict a cluster for each new observation, and then feed the observation into the appropriate model. We also could have been aware of the fact that the data was unbalanced, and looked further into techniques for working with unbalanced data such as [2], although we did address the problem in each section by favouring low threshold models, or balacing the data in preprocessing.

# 1. Singh M, Cariou A, Varenne O, Picard F, Pham V. Phenotypic clustering of myocardial infarction complicated by out-of-hospital cardiac arrest using unsupervised machine learning. Resuscitation. 2026 Jun;223:111037. doi: 10.1016/j.resuscitation.2026.111037. Epub 2026 Mar 5. PMID: 41794115.
# 2. https://pmc.ncbi.nlm.nih.gov/articles/PMC12074545/
