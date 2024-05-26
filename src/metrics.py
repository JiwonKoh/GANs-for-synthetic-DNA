from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import math

class metric():
    def __init__(self, preds, labels):
        tp, tn, fn, fp = 0, 0, 0, 0

        for (_, p), l in zip(preds, labels):
            if p >= 0.5 and l[1] == 1:
                tp += 1
            elif p < 0.5 and l[1] == 1:
                fn += 1
            elif p < 0.5 and l[1] == 0:
                tn += 1
            elif p >= 0.5 and l[1] == 0:
                fp += 1
        
        self.tp = tp
        self.tn = tn
        self.fn = fn
        self.fp = fp

    def recall(self):
        self.rec = self.tp / (self.tp + self.fn)
        return self.tp / (self.tp + self.fn)


    def precision(self):
        self.pre = self.tp / (self.tp + self.fp)
        return self.tp / (self.tp + self.fp)


    def f1score(self):
        r = self.rec
        p = self.pre

        return 2 * (r * p) / (r + p)

    def mcc(self):
        nom = self.tp * self.tn - self.fp * self.fn 
        denom = math.sqrt((self.tp + self.fp)*(self.tp + self.fn)*(self.tn + self.fp)*(self.tn + self.fn))
        return nom / denom