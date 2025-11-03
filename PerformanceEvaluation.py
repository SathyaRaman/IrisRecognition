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


def ROC_Result(all_distances, y_test, class_labels, metrics):
    print("\nTable 4: False Match and False Non-match Rates with Paper Thresholds")
    paper_thresholds = [0.446, 0.472, 0.502]

    #get thresholds
    table4_rows = [["Metric", "Threshold", "False Match Rate (%)", "False Non-match Rate (%)"]]

    plt.figure(figsize=(6.5, 4.8))

    #we are just using the cosine metric as the paper did
    #can add other metrics to input if that is needed
    for metric in metrics:
        D = all_distances[metric]
        genuine, impostor = [], []

        #get genuine and impostor distances
        for i, true_label in enumerate(y_test):
            idx = np.where(class_labels == true_label)[0][0]
            genuine.append(D[i, idx])
            impostor.extend(D[i, np.arange(len(class_labels)) != idx])

        genuine = np.array(genuine)
        impostor = np.array(impostor)

        #compute ROC curve for visualization
        #calculate the FMR and FNMR using imposter and genuine distances
        #the below is for the continuous curve in figure 11
        thresholds = np.linspace(genuine.min(), impostor.max(), 200)
        FMR = [(impostor <= t).mean() * 100 for t in thresholds]
        FNMR = [(genuine > t).mean() * 100 for t in thresholds]
        plt.plot(FMR, FNMR, label=metric.upper())

        #compute FMR/FNMR at the paper thresholds (0.446, 0.472, 0.502)  
        for t in paper_thresholds:
            fmr = (impostor <= t).mean() * 100
            fnmr = (genuine > t).mean() * 100
            table4_rows.append([
                metric.upper(),
                f"{t:.3f}",
                f"{fmr:.3f}",
                f"{fnmr:.3f}"
            ])

    #print table
    for row in table4_rows:
        print("\t".join(row))

    #save table 4 visualization
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    tbl = ax.table(cellText=table4_rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    ax.set_title("Table 4: FMR / FNMR at Paper Thresholds", fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig("Table4_PaperThresholds.png", dpi=300)
    plt.close(fig)

    #save ROC Curve (Figure 11)
    plt.xlabel("False Match Rate (%)")
    plt.ylabel("False Non-match Rate (%)")
    plt.title("Figure 11: ROC Curves (Verification Mode)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure11_ROC_Curves_PaperThresholds.png", dpi=300)
    plt.close()
    plt.savefig("Figure11_ROC_Curves.png", dpi=300)
    plt.close()
