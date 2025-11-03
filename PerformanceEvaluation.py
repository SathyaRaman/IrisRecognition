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
    """
    Computes and visualizes Correct Recognition Rate (CRR) results.
    Generates Table 4 and Figure 11 from the paper.
    
    Parameters:
    - all_distances: distance from each eye to the closest center per metric
    - y_test: true labels for test set
    - class_labels: distances calculated for test samples
    - metrics: distance measures we want included, currently set to only cosine in main, can include l1 and l2 as well
    """
    print("\nTable 4: False Match and False Non-match Rates")

    #set up table
    table4_rows = [["Metric", "Threshold", "False Match Rate (%)", "False Non-match Rate (%)"]]

    plt.figure(figsize=(6.5, 4.8))

    #we are just using the cosine metric as the paper did
    #can add other metrics to input if that is needed
    for metric in metrics:
        D = all_distances[metric]
        authorized, impostor = [], []

        #get authorized and impostor distances
        for i, true_label in enumerate(y_test):
            idx = np.where(class_labels == true_label)[0][0]
            authorized.append(D[i, idx])
            impostor.extend(D[i, np.arange(len(class_labels)) != idx])

        authorized = np.array(authorized)
        impostor = np.array(impostor)

        #boolean to determine if the metric is similarity-based or distance-based
        #if the authorized mean is higher than imposter mean, then the metric is similarity (cosine), else L1 and L2
        similarity = np.mean(authorized) > np.mean(impostor) 
        
        # automatically select 3 thresholds
        if similarity: #for cosine similarity
            low = np.percentile(impostor, 10) #low threshold is for the 10th percentile of imposters
            mid = np.median((authorized.mean(), impostor.mean())) # the mean of the authorized and imposters
            high = np.percentile(authorized, 90) # high threshold is the 90th percentile of authorized
        else: #for L1 and L2 metrics
            low = np.percentile(authorized, 90)  #low threshold is for 90th percentile of authorized
            mid = (authorized.mean() + impostor.mean()) / 2  #the mean of the authorized and importers
            high = np.percentile(impostor, 10) # high threshold is for the 10th percentile of imposters

        #compute ROC curve for visualization
        #calculate the FMR and FNMR using imposter and authorized distances
        #the below is for the continuous curve in figure 11
        thresholds = sorted([low, mid, high])
        FMR_list, FNMR_list = [], []
        for t in thresholds:
            if similarity: #for cosine similarity 
                fmr = (impostor >= t).mean() * 100 #false matches are for imposters that are above the threshold
                fnmr = (authorized < t).mean() * 100 #false nonmatches are for authorized below the threshold 
            else: #for L1 and L2
                fmr = (impostor <= t).mean() * 100 #false matches are for imposters that are below the threshold 
                fnmr = (authorized > t).mean() * 100 #false nonmatches are for the authorized above the threshold 

            FMR_list.append(fmr)
            FNMR_list.append(fnmr)
            #append the thresholds to the table
            table4_rows.append([
                metric.upper(),
                f"{t:.3f}",
                f"{fmr:.3f}",
                f"{fnmr:.3f}"
            ])
        #plot the ROC curve
        plt.plot(FMR_list, FNMR_list, marker='o', lw=1.8, label=f"{metric.upper()} (ROC)")

    #print table
    print("Metric Threshold FMR(%) FNMR(%)")
    for row in table4_rows[1:]:
        print(*row)


    #save table 4 visualization
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    tbl = ax.table(cellText=table4_rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    ax.set_title("Table 4: FMR / FNMR", fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig("Table4_.png", dpi=300)
    plt.close(fig)

    #save ROC Curve (Figure 11)
    plt.xscale('log') #log scale the x axis, as done in the Ma paper
    plt.xlabel("False Match Rate (%)")
    plt.ylabel("False Non-match Rate (%)")
    plt.title("Figure 11: ROC Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure11_ROC_Curves.png", dpi=300)
    plt.close()
