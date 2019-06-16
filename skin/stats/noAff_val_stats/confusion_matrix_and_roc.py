import pandas as pd
from ast import literal_eval
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp


stats = pd.read_csv('../noAff_aff_stats/noAff/val_results', delimiter=',')
class_label_matcher = pd.read_csv('../../trainer/label_class_matcher', delimiter=',').set_index('label').to_dict()['class']
stats['probabilities'] = stats['probabilities'].map(literal_eval)
classes = list(class_label_matcher.values())
nr_classes = len(classes)

confusion_matrix_raw = confusion_matrix(stats['true'], stats['predicted'])
confusion_matrix_scaled = confusion_matrix_raw.astype('float') / confusion_matrix_raw.sum(axis=1)[:, np.newaxis]

f = plt.figure()
sns.heatmap(confusion_matrix_scaled, xticklabels=classes, yticklabels=classes, annot=True)
plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidziana klasa')
f.savefig('./pdfs/confusionMatrixVal.pdf', bbox_inches='tight')
plt.show()

f = plt.figure()
fpr = dict()
tpr = dict()
roc_auc = dict()
for label, _class in class_label_matcher.items():
    fpr[label], tpr[label], _ = roc_curve(stats['true'], stats['probabilities'].map(lambda x: x[label]), pos_label=label)
    roc_auc[label] = auc(fpr[label], tpr[label])
    plt.plot(fpr[label], tpr[label], label='ROC dla klasy {0} (AUC = {1:0.2f})'.format(_class, roc_auc[label]), lw=0.75)

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
fpr_macro = np.unique(np.concatenate([fpr[i] for i in range(nr_classes)]))
tpr_mean = np.zeros_like(fpr_macro)
for i in range(nr_classes):
    tpr_mean += interp(fpr_macro, fpr[i], tpr[i])

tpr_macro = tpr_mean / nr_classes

macro_roc_auc = auc(fpr_macro, tpr_macro)
plt.plot(fpr_macro, tpr_macro, label='ROC macro-averaged (AUC macro-averaged = {0:0.2f})'.format(macro_roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywe ROC oraz wartości AUC')
plt.legend(loc="lower right")
f.savefig('./pdfs/rocVal.pdf', bbox_inches='tight')
plt.show()
