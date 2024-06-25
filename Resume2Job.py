import pandas as pd
import pickle
from FTModel import build_model, output_graphs_generic, output_graphs_binary_selection
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csc_matrix

def get_top_probs(target: float, classifier: SGDClassifier, input: csc_matrix):
    '''Returns a list of labels with highest probability from given classifier
    with give input until their combined probability reaches given target
    
    Target must be within (0; 1)'''
    df_probs = pd.DataFrame(classifier.classes_, columns = ["Label"])
    df_probs["Probs"] = classifier.predict_proba(input)[0]
    df_probs.sort_values("Probs", ascending = False, inplace = True)
    top_probs = list()
    total = 0
    i = 0
    while total < target:
        top_probs.append([df_probs["Label"].iloc[i], df_probs["Probs"].iloc[i]])
        total += df_probs["Probs"].iloc[i]
        i += 1
    return top_probs

#Loads the "Resume.csv" file that has been run theough CleanData
df = pd.read_csv("Resume_clean.csv")
#Sets parameters
v_params = {"sublinear_tf": True, "max_df": 0.5, "min_df": 5, "stop_words": "english"}
c_params = {"early_stopping": True, "C": 0.3, "loss": "squared_hinge"}
#Retrives data and model
X_train, X_test, y_train, y_test = build_model(df, "Resume_str", "Category", "TfidfVectorizer", "PassiveAggressiveClassifier", v_params, c_params, verbose = True)
with open("model.pickle", "rb") as file: vectorizer, classifier, labels, feature_names = pickle.load(file)
output_graphs_generic(classifier, X_train, X_test, y_test, labels, feature_names)

#The accuracy of above model is fairly low. The following code allows us to retrive several most likely labels for a give input. Since matching resumes to jobs is a low-risk task, retrival matters to us more than precision.
c_params = {"early_stopping": True, "loss": "log_loss"}
X_train, X_test, y_train, y_test = build_model(df, "Resume_str", "Category", "TfidfVectorizer", "SGDClassifier", v_params, c_params, file_name = "SDG.pickle", verbose = True)
with open("SDG.pickle", "rb") as file: vectorizer, classifier, labels, feature_names = pickle.load(file)
top_probs = get_top_probs(0.7, classifier, X_test[0])
print(f"{len(top_probs)} most likely labels for first test input:")
for i in range(len(top_probs)): print(f"{i})  {top_probs[i][0]}:{top_probs[i][1]:.3f}")

#According to the confusion matrix, there are several pair of labels that often get confused for eachother because of their common terminology. If we get an input that returns a result in one of such group, we can double-check it by limiting the model to the labels of that group
confusion_groups = [["ACCOUNTANT", "FINANCE"],["ARTS", "TEACHER"],["CONSULTANT", "INFORMATION-TECHNOLOGY"],["ADVOCATE", "HEALTHCARE"]]
confusion_labels = list()
#For demonstration purposes, we get the first input that returns a label in one of the groups above
for label in classifier.predict(X_test):
    for group in confusion_groups:
        if label in group:
            confusion_labels = group.copy()
            break
    if len(confusion_labels) > 0: break
#Filter DataFrame down to the records with the labels from the group we've found
mask = list()
for label in df["Category"]: mask.append(label in confusion_labels)
#Build the more focused model and output the graphs for it
X_train, X_test, y_train, y_test = build_model(df[mask], "Resume_str", "Category", "TfidfVectorizer", "PassiveAggressiveClassifier", v_params, c_params, file_name = "secondary.pickle", verbose = True)
with open("secondary.pickle", "rb") as file: vectorizer, classifier, labels, feature_names = pickle.load(file)
output_graphs_generic(classifier, X_train, X_test, y_test, labels, feature_names)
if len(labels) == 2: output_graphs_binary_selection(classifier, X_test, y_test, labels)