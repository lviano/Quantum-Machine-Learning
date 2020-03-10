import numpy as np
import matplotlib.pyplot as plt
import debug
import os

def visualize_and_save_figure(figure, path, file_name, file_type = 'pdf', dpi = 300, show = False, close = True):
    # Create dsgtination path as needed
    os.makedirs(os.path.normpath(path), exist_ok=True)

    # Save figure to file
    figure.savefig(os.path.join(path, file_name + '.' + file_type), dpi=dpi, format=file_type)

    # Show figure if specified
    if show:
        plt.show(figure)

    # Close figure if specified
    if close:
        plt.close(figure)

def visualize_pca_2d(x_pca_2d, db, names, rank, args, visualize_noise = False):
    # Extract the data to be visualized
    labels = db.labels_
    unique_labels = set(labels)
    n_cluster = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    db_figure = plt.figure()

    for k in unique_labels:
        if (k != -1):
            class_k_members_mask = (labels == k)
            points_k_class = x_pca_2d[class_k_members_mask]
            plt.plot(points_k_class[:,0], points_k_class[:,1], 'o')
            print("In class k = " + str(k) + "\n" + str(names[class_k_members_mask]) )
        #else:
        #    break

    # Set plot title
    plt.title('2D PCA Scatter Plot of clusters obtained by DBSCAN in ' + str(rank) + " dimensions. " + '\n' + " Parameters. eps = " + str(args.epsilon) + " NumPointsMin = " + str(args.min_samples))

    visualize_and_save_figure(db_figure, args.figure_path, 'db_figure', args.file_type, args.dpi, args.show)

    noise_mask = (labels ==-1)
    noisy_points = x_pca_2d[noise_mask]

    #Plot the noise
    db_noise = plt.figure()
    plt.plot(noisy_points[:,0], noisy_points[:,1], 'o')
    plt.title("Outlier points obtained by DBSCAN in " + str(rank) + " dimensions. " + '\n' + " Parameters. eps = " + str(args.epsilon) + " NumPointsMin = " + str(args.min_samples) )
    visualize_and_save_figure(db_noise, args.figure_path, 'db_noise', args.file_type, args.dpi, args.show)

def visualize(args, names, rank, db, x_pca_2d):
    visualize_pca_2d(x_pca_2d, db, names, rank, args, visualize_noise=False)

    return
