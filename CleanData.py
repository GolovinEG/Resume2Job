import pandas as pd

df = pd.read_csv("Resume.csv")
#Removes words from resume text that are also used as one of the labels
categories = df["Category"].unique()
#Also removes words that share a word root
categories_ext = ["public", "media", "financial", "accounting", "relations", "teaching", "digital", "design", "bank", "technology"]

def remove_categories(x: str):
    x = x.lower()
    for category in categories:
        x = x.replace(category.lower(), "")
    for category in categories_ext:
        x = x.replace(category, "")
    return x

df["Resume_str"] = df["Resume_str"].apply(remove_categories)
df.to_csv("Resume_clean.csv")