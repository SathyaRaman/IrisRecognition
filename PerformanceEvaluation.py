import numpy as np
import matplotlib.pyplot as plt
from IrisMatching import iris_matching

def CRR_Result(y_test, y_pred, X_train, y_train, X_test_groups):
    """
    Computes and visualizes Correct Recognition Rate (CRR) results.
    Generates Table 3 and Figure 10 from the paper.
    
    Parameters:
    - y_test: true labels for test set
    - y_pred: predicted labels from iris_matching, dict per metric
    - X_train, y_train: training features and labels
    - X_test_groups: list of test features (with rotations)
    """
    #dict to storre CRR per similarity metric
    crr_per_metric = {}
    for metric in y_pred:
        #count number of correct predictions
        num_correct = np.sum(y_pred[metric] == y_test)
        crr_for_metric = (num_correct / len(y_test)) * 100
        crr_per_metric[metric] = crr_for_metric

    #Table 3: Recognition Results Using Different Similarity Measures
    print("Table 3: Recognition Results Using Different Similarity Measures")
    print("(In Reduced Feature Set)")
    print(f"{'Similarity Measure':<20} | {'Correct Recognition Rate (%)':>10}")
    for metric, value in crr_per_metric.items():
        print(f"{metric:<20} | {value:>10.2f}")

    #save as an image as per Edstem
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table_data = [["Similarity Measure", "Correct Recognition Rate (%)"]] + [
        [metric.upper(), f"{value:.2f}"] for metric, value in crr_per_metric.items()
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)
    plt.title("Table 3: Recognition Results Using Different Similarity Measures", fontsize=10)
    plt.tight_layout()
    plt.savefig("Table3_CRR.png", dpi=300)
    plt.close()

    #Figure 10: CRR vs Dimensionality
    dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 220]
    metrics = ("l1", "l2", "cosine")
    crr_history = {m: [] for m in metrics}

    #iterate over feature vector dimensions for CRR evaluation
    for d in dimensions:
        #run iris matching with specified LDA dimensionality
        y_pred_d, *_ = iris_matching(X_train, y_train, X_test_groups, y_test, lda_dim=d)
        yt = np.asarray(y_test)
        for m in metrics:
            yp = np.asarray(y_pred_d[m])
            #compute CRR for this metric and dimension
            crr = (np.sum(yp == yt) / len(yt)) * 100.0
            crr_history[m].append(crr)

    #plot and save as png
    plt.figure(figsize=(7, 4))
    for metric, vals in crr_history.items():
        plt.plot(dimensions, vals, marker='o', label=metric.upper())
    plt.xlabel('Dimensionality of the Feature Vector')
    plt.ylabel('Correct Recognition Rate (%)')
    plt.title('Figure 10: CRR vs Feature Dimensionality')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure10_CRR_vs_Dimensionality.png", dpi=300)
    plt.close()


def ROC_Result(all_distances, y_test, class_labels, metrics=("l1", "l2", "cosine")):
    """
    Computes and visualizes ROC (Verification Mode) results.
    Generates Table 4 and Figure 11 from the paper.
    
    Parameters:
    - all_distances: dictionary of distances per metric (from iris_matching)
    - y_test: true labels for test set
    - class_labels: list of unique classes in training set
    - metrics: list of similarity metrics to evaluate
    """
    print("\nTable 4: ROC (Verification Mode)")
    eer_results = []

    plt.figure(figsize=(7, 5))
    #iterate over each similarity metric
    for metric in metrics:
        D = all_distances[metric]
        genuine = [] #distances for corrrrect matches
        impostor = [] #distances for incorrect matches

        #calculate genuine and impostor distances
        for i, true_label in enumerate(y_test):
            true_idx = np.where(class_labels == true_label)[0][0]
            genuine.append(D[i, true_idx])
            impostor.extend(D[i, np.arange(len(class_labels)) != true_idx])

        genuine = np.array(genuine)
        impostor = np.array(impostor)

        #thresholds for ROC curve
        thresholds = np.unique(np.concatenate([genuine, impostor])) 
        FMR = [(impostor <= t).mean() for t in thresholds] #false match rate
        FNMR = [(genuine > t).mean() for t in thresholds] #false non-match rate

        #computing ERR
        eer_idx = np.argmin(np.abs(np.array(FMR) - np.array(FNMR)))
        eer = 0.5 * (FMR[eer_idx] + FNMR[eer_idx])
        eer_results.append((metric.upper(), eer * 100))

        print(f"{metric.upper():<8} | Equal Error Rate (EER): {eer*100:.2f}%")
        plt.plot(FMR, FNMR, label=f"{metric.upper()} (EER={eer*100:.2f}%)")

    #save Table 4 as image
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table_data = [["Similarity Measure", "Equal Error Rate (EER %)"]] + [
        [m, f"{v:.2f}"] for m, v in eer_results
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)
    plt.title("Table 4: ROC (Verification Mode)", fontsize=10)
    plt.tight_layout()
    plt.savefig("Table4_ROC_EER.png", dpi=300)
    plt.close()

    #Figure 11: ROC Curves
    plt.xlabel("False Acceptance Rate (FAR / FMR)")
    plt.ylabel("False Rejection Rate (FRR / FNMR)")
    plt.title("Figure 11: ROC Curves for Verification Mode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure11_ROC_Curves.png", dpi=300)
    plt.close()
