from data_helpers import *
from loader import seq_tokens

from sklearn import linear_model
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def get_feats(X):
    feats = np.zeros(shape=[len(X), len(seq_tokens)])
    for i, seq in enumerate(X):
        for j,token in enumerate(seq_tokens):
            feats[i,j] = seq.count(token)
    return feats

X,Y = zip(*train_data)
X_dev,Y_dev = zip(*dev_data)
X_test,Y_test = zip(*test_data)

X_feat = get_feats(X)
X_dev_feat = get_feats(X_dev)
X_test_feat = get_feats(X_test)

logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X_feat,Y)


Y_pred = logreg.predict(X_feat)
Y_dev_pred = logreg.predict(X_dev_feat)
Y_test_pred = logreg.predict(X_test_feat)

acc = np.mean(np.equal(Y_pred, Y, dtype=float))
acc_dev = np.mean(np.equal(Y_dev_pred, Y_dev, dtype=float))
acc_test = np.mean(np.equal(Y_test_pred, Y_test, dtype=float))

print('Train acc: ',acc)
print('Dev acc: ',acc_dev)
print('Test acc: ',acc_test)
# print(X_feat[0])

probs_test = logreg.predict_proba(X_test_feat)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(np.equal(Y_test,i), probs_test[:, i], drop_intermediate=False)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
lw = 2
for c in range(num_classes):
    plt.plot(fpr[c], tpr[c],
         lw=lw, label='{:} (area = {:0.2f})'.format( seq_classes_rev[c], roc_auc[c]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")


plt.show()
