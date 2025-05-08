import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def get_path_from_root(path_param):
    current_dir = os.getcwd()
    
    while True:
        
        if os.path.basename(current_dir) == "DataScience":
            return os.path.join(current_dir, path_param)
        
        parent_dir = os.path.dirname(current_dir)
        
        if parent_dir == current_dir:
            raise FileNotFoundError("A \"DataScience\" mappa nem található a mappa-hierarchiában.")
        
        current_dir = parent_dir


def read_paysim(get_original_data: bool):
    DATA = pd.read_csv( get_path_from_root("data/synthetic_financial_dataset.csv") )
    
    DATA = DATA.rename(columns = {
        'type': 'transaction_type',
        'nameOrig': 'sender',
        'oldbalanceOrg': 'sender_old_balance',
        'newbalanceOrig': 'sender_new_balance',
        'nameDest': 'receiver',
        'oldbalanceDest': 'receiver_old_balance',
        'newbalanceDest': 'receiver_new_balance',
        'isFraud': 'isfraud'
    })
    DATA = DATA.drop(columns=['isFlaggedFraud'])

    COPIED_DATA = DATA.copy()
    COPIED_DATA['sender_receiver_type'] = pd.Series(dtype="object")
    COPIED_DATA.loc[DATA["sender"].str.contains('C') & DATA["receiver"].str.contains('C'), 'sender_receiver_type'] = 'CC'
    COPIED_DATA.loc[DATA["sender"].str.contains('C') & DATA["receiver"].str.contains('M'), 'sender_receiver_type'] = 'CM'
    COPIED_DATA.loc[DATA["sender"].str.contains('M') & DATA["receiver"].str.contains('C'), 'sender_receiver_type'] = 'MC'
    COPIED_DATA.loc[DATA["sender"].str.contains('M') & DATA["receiver"].str.contains('M'), 'sender_receiver_type'] = 'MM'

    columns_new_order = ['step','transaction_type','sender_receiver_type','amount','sender','sender_old_balance',
                         'sender_new_balance','receiver','receiver_old_balance','receiver_new_balance','isfraud']
    COPIED_DATA = COPIED_DATA[columns_new_order]
    COPIED_DATA = COPIED_DATA.drop(columns=["sender", "receiver"], axis='columns')
        
    COPIED_DATA["sender_balance_diff"] = COPIED_DATA["sender_old_balance"] - COPIED_DATA["sender_new_balance"]
    COPIED_DATA["receiver_balance_diff"] = COPIED_DATA["receiver_new_balance"] - COPIED_DATA["receiver_old_balance"]
    
    if get_original_data == True:
        return DATA, COPIED_DATA
    else:
        return COPIED_DATA
    

def plot_history(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label="loss")
    ax.plot(history.history['val_loss'], label="val_loss")
    ax.set_title('Model Loss')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()
    
    return fig
    
    
def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(fpr, tpr, linewidth=3, label=f"AUC = {round(roc_auc, 3)}")
    ax.plot([0,1], [0,1], linewidth=3, ls="--")
    ax.set_xlim(left=-0.02, right=1)
    ax.set_ylim(bottom=0, top=1.02)
    ax.set_xlabel("FPR (False Positive Rate)")
    ax.set_ylabel("TPR (True Positive Rate)")
    ax.set_title("ROC curve")
    ax.legend()
    
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, cmap="coolwarm", annot=True, linewidths=0.5, fmt="d")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Original")
    
    return fig


def plot_f1_score(threshold, best_threshold, f1_score, max_f1_score):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(threshold, f1_score, label="F1 score", linewidth=3, color="green")
    ax.scatter(best_threshold, max_f1_score, label=f"Max F1 score = {round(max_f1_score, 3)}", s=50, color="red")
    ax.axvline(best_threshold, color="black", ls="--", label=f"Threshold = {best_threshold}")
    ax.axhline(max_f1_score, color="black", ls="-")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 score különböző threshold értékeknél")
    ax.legend(loc="upper right")
    
    return fig


def plot_pr_curve(precision, recall, average_precision, best_precision, best_recall, max_f1_score):
    
    fig, ax = plt.subplots(figsize=(12,6))
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.001, 1)
        y = f_score * x / (2*x-f_score)
        plt.plot(x[y>=0], y[y>=0], color="gray", alpha=0.2)
        plt.annotate('F1 = {0:0.2f}'.format(f_score), xy = (0.95, y[45]+0.02))
        
    plt.plot(recall[1:], precision[1:], label=f"Area = {round(average_precision, 3)}", linewidth=3)
    plt.scatter(best_recall, best_precision, label=f"F1 score = {round(max_f1_score, 3)}", s = 50, color = "red")
    plt.axvline(best_recall, color="black", ls="--", label=f"Recall = {round(best_recall, 3)}")
    plt.axhline(best_precision, color="black", ls="-", label=f"Precision = {round(best_precision, 3)}")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve")
    ax.legend(loc = "upper right")
    
    return fig        


def calc_tree_depth(node, depth=0):
    
    # Ha nincs gyermeke
    if "left_child" not in node and "right_child" not in node:
        return depth
    
    left_depth, right_depth = 0, 0
    
    if "right_child" in node:
        right_depth = calc_tree_depth(node["right_child"], depth+1)
    if "left_child" in node:
        left_depth = calc_tree_depth(node["left_child"], depth+1)
        
    return max(right_depth, left_depth)


def lgbm_get_max_tree_depth(tree_info):
    max_depth = 0
    
    for tree in tree_info:
        tree_depth = calc_tree_depth(tree["tree_structure"])
        max_depth = max(max_depth, tree_depth)
        
    return max_depth

