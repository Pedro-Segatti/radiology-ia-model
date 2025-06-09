import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from cnn.data.pre_proccess_data import PreProccessData


class ValidateTrain:
    def __init__(self, model, data_path, input_shape=(256, 256, 1), output_dir="output"):
        self.model = model
        self.data_path = data_path  # <- caminho raiz, contendo train/ e validation/
        self.input_shape = input_shape
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def validate_model(self):
        # Usa apenas o val_gen agora, pois você já separou os dados
        _, val_gen = PreProccessData(self.data_path, target_size=self.input_shape[:2]).preprocess_data()

        # Avaliação simples
        loss, accuracy = self.model.evaluate(val_gen, verbose=1)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        # Predições
        y_prob = self.model.predict(val_gen, verbose=1)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = val_gen.classes
        class_labels = list(val_gen.class_indices.keys())

        # Geração dos gráficos e relatórios
        self._generate_classification_report(y_true, y_pred, class_labels)
        self._generate_confusion_matrix(y_true, y_pred, class_labels)
        self._generate_roc_curve(y_true, y_prob, class_labels)
        self._generate_class_distribution_plot(y_true, y_pred, class_labels)

        return loss, accuracy

    def _generate_classification_report(self, y_true, y_pred, class_labels, filename="static/classification_report.txt"):
        report = classification_report(y_true, y_pred, target_names=class_labels)
        print(report)
        with open(filename, "w") as f:
            f.write(report)

    def _generate_confusion_matrix(self, y_true, y_pred, class_labels, filename="static/confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _generate_roc_curve(self, y_true, y_prob, class_labels, filename="static/roc_curve.png"):
        n_classes = len(class_labels)
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f"Classe {class_labels[i]} (AUC = {roc_auc[i]:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Taxa de Falsos Positivos")
        plt.ylabel("Taxa de Verdadeiros Positivos")
        plt.title("Curvas ROC por classe")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def _generate_class_distribution_plot(self, y_true, y_pred, class_labels, filename="static/class_distribution.png"):
        counts_true = np.bincount(y_true, minlength=len(class_labels))
        counts_pred = np.bincount(y_pred, minlength=len(class_labels))

        x = np.arange(len(class_labels))
        width = 0.35

        plt.figure(figsize=(8, 6))
        plt.bar(x - width/2, counts_true, width, label="Real")
        plt.bar(x + width/2, counts_pred, width, label="Predito")
        plt.xticks(x, class_labels)
        plt.ylabel("Número de imagens")
        plt.title("Distribuição: Real vs Predito")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
