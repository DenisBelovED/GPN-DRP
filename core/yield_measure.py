from numpy import zeros, int32
from tabulate import tabulate


class YieldMeasure:
    def __init__(self, class_dict):
        self.classes_count = len(class_dict)
        self.classes_support = class_dict
        self.confusion_matrix = zeros((self.classes_count, self.classes_count), int32)

    def append(self, true_list, predict_list):
        for i_t in zip(true_list, predict_list):
            self.confusion_matrix[i_t] += 1

    def get_estimate(self):
        if not (self.confusion_matrix.sum(1).prod() * self.confusion_matrix.sum(0).prod()):
            raise ValueError(f"Confusion matrix has zeros line\n{self.confusion_matrix}")
        precision = []
        recall = []
        true_predicts = 0
        f1_score = []
        for i in range(self.classes_count):
            precision.append(self.confusion_matrix[i, i] / self.confusion_matrix[:, i].sum())
            recall.append(self.confusion_matrix[i, i] / self.confusion_matrix[i, :].sum())
            f1_score.append((2 * precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))
            true_predicts += self.confusion_matrix[i, i]
        accuracy = true_predicts / self.confusion_matrix.sum()
        return precision, recall, f1_score, accuracy

    def show_rating(self):
        precision, recall, f1_score, accuracy = self.get_estimate()
        l_f = lambda x: f"{x * 100:.2f}%"
        precision = list(map(l_f, precision))
        recall = list(map(l_f, recall))
        f1_score = list(map(l_f, f1_score))
        accuracy = l_f(accuracy)
        labels = list(self.classes_support.keys())
        supports = list(self.classes_support.values())

        table = [[f"true {e}" for e in labels] + ['precision', 'recall', 'F1', 'support']]
        for i in range(len(labels)):
            table.append(
                [f"pred {labels[i]}", *self.confusion_matrix[:, i], precision[i], recall[i], f1_score[i], supports[i]]
            )
        print(tabulate(table, headers="firstrow"))
        print(f"Accuracy: {accuracy}")
