from sklearn import metrics
import numpy as np

class AccuracyTeller():
    """ TODO: Unintentional OOPs inspiration, should be made Pure
    """
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []

    def register_result(self, true, pred):
        true = true.cpu().numpy(); pred = pred.cpu().numpy()
        true_ = true[true>-1].astype(np.int)
        pred_ = pred[true>-1]

        self.true_labels.extend(true_)
        self.pred_labels.extend(pred_)

    def reset(self):
        self.true_labels = []
        self.pred_labels = []

    def area_under_curve(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.true_labels, self.pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    def accuracy_score(self):
        pred_ = ( np.asarray(self.pred_labels) > 0.65 ).astype(np.int)
        acc = metrics.accuracy_score(self.true_labels, pred_)
        return acc