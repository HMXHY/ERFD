from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import logging
import torch

def log_config(name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/4090/'+name+'.log'),
            logging.StreamHandler()
        ]
    )

def log_confusion_matrix(cm, label):
    """将混淆矩阵格式化输出到日志"""
    logging.info(f"\n{label} Confusion Matrix:")
    for i, row in enumerate(cm):
        row_str = " ".join(f"{val:5d}" for val in row)
        logging.info(f"Class {i}: {row_str}")
    logging.info(f"Total samples: {np.sum(cm)}")

def calculate_macro_metrics(y_true, y_pred):
    """计算macro平均指标（返回百分比）"""
    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    accuracy = accuracy_score(y_true, y_pred) * 100
    return accuracy, precision, recall, f1

def output_metrics_metrics(y_true, y_pred , test_type):
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    log_confusion_matrix(cm, test_type)
    acc, prec, rec, f1 = calculate_macro_metrics(y_true, y_pred)
    logging.info(f"{test_type} - Macro Acc: {acc:.2f}%, Prec: {prec:.2f}%, Rec: {rec:.2f}%, F1: {f1:.2f}%")
    return acc, prec, rec, f1

