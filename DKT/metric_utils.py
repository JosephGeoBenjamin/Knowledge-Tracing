from sklearn import metrics
import numpy as np

class AccuracyTeller():
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []

    def register_result(self, true, pred):
        true = true.cpu().numpy(); pred = pred.cpu().numpy()
        true_ = true[true>-1].astype(np.int)
        pred_ = ( pred[true>-1] > 0.65 ).astype(np.int)

        self.true_labels.extend(true_)
        self.pred_labels.extend(pred_)

    def area_under_curve(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.true_labels, self.pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    def accuracy_score(self):
        acc = metrics.accuracy_score(self.true_labels, self.pred_labels,)
        return acc