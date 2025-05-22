import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

def analyze_class_pairwise_distances(embeddings, labels, max_samples=5000):
    """
    Analyze pairwise distances between embeddings by class.
    Returns:
    - A dictionary mapping class pairs to their distance distributions
    - Overall same-class and different-class distance arrays
    """
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        sample_emb = embeddings[indices]
        sample_labels = labels[indices]
    else:
        sample_emb = embeddings
        sample_labels = labels
    distances = euclidean_distances(sample_emb)
    class_pair_distances = {}
    same_class_dists = []
    diff_class_dists = []
    unique_labels = np.unique(sample_labels)
    for i in range(len(sample_labels)):
        for j in range(i+1, len(sample_labels)):
            class_i = sample_labels[i]
            class_j = sample_labels[j]
            distance = distances[i, j]
            pair_key = tuple(sorted([class_i, class_j]))
            if pair_key not in class_pair_distances:
                class_pair_distances[pair_key] = []
            class_pair_distances[pair_key].append(distance)
            if class_i == class_j:
                same_class_dists.append(distance)
            else:
                diff_class_dists.append(distance)
    return class_pair_distances, np.array(same_class_dists), np.array(diff_class_dists)

def plot_class_distance_heatmap(class_pair_distances, unique_labels, class_names=None):
    n_classes = len(unique_labels)
    distance_matrix = np.zeros((n_classes, n_classes))
    for (class_i, class_j), distances in class_pair_distances.items():
        i = np.where(unique_labels == class_i)[0][0]
        j = np.where(unique_labels == class_j)[0][0]
        median_dist = np.median(distances)
        distance_matrix[i, j] = median_dist
        distance_matrix[j, i] = median_dist
    fig, ax = plt.subplots(figsize=(12, 10))
    if class_names is None:
        class_names = [f"Class {label}" for label in unique_labels]
    sns.heatmap(distance_matrix, annot=True, fmt=".2f", 
                xticklabels=class_names, yticklabels=class_names,
                cmap="YlGnBu", ax=ax)
    ax.set_title("Median Pairwise Distances Between Classes")
    plt.tight_layout()
    return fig

def identify_problematic_pairs(class_pair_distances, unique_labels, class_names=None):
    if class_names is None:
        class_names = {label: f"Class {label}" for label in unique_labels}
    same_class_stats = {}
    diff_class_stats = {}
    for (class_i, class_j), distances in class_pair_distances.items():
        if class_i == class_j:
            same_class_stats[class_i] = {
                'median': np.median(distances),
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            }
        else:
            pair_name = (class_i, class_j)
            diff_class_stats[pair_name] = {
                'median': np.median(distances),
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            }
    similar_diff_classes = sorted(diff_class_stats.items(), key=lambda x: x[1]['median'])
    most_similar = similar_diff_classes[:5]
    dispersed_classes = sorted(same_class_stats.items(), key=lambda x: x[1]['median'], reverse=True)
    most_dispersed = dispersed_classes[:5]
    return {
        'most_similar_different_classes': most_similar,
        'most_dispersed_same_classes': most_dispersed
    }

def plot_class_specific_distributions(class_pair_distances, class_pairs_of_interest, class_names=None):
    import matplotlib.pyplot as plt
    if class_names is None:
        class_names = lambda x: f"Class {x}"
    fig, axes = plt.subplots(len(class_pairs_of_interest), 1, figsize=(10, 4 * len(class_pairs_of_interest)))
    if len(class_pairs_of_interest) == 1:
        axes = [axes]
    for i, pair in enumerate(class_pairs_of_interest):
        pair_key = tuple(sorted(pair))
        if pair_key in class_pair_distances:
            distances = class_pair_distances[pair_key]
            axes[i].hist(distances, bins=30, alpha=0.7)
            if pair[0] == pair[1]:
                title = f"Same-Class Distances: {class_names(pair[0])}"
            else:
                title = f"Different-Class Distances: {class_names(pair[0])} vs {class_names(pair[1])}"
            axes[i].set_title(title)
            axes[i].set_xlabel("Distance")
            axes[i].set_ylabel("Frequency")
            mean_dist = np.mean(distances)
            median_dist = np.median(distances)
            std_dist = np.std(distances)
            axes[i].axvline(mean_dist, color='r', linestyle='--', label=f'Mean: {mean_dist:.2f}')
            axes[i].axvline(median_dist, color='g', linestyle='--', label=f'Median: {median_dist:.2f}')
            axes[i].legend()
            stats_text = f"Mean: {mean_dist:.2f}\nMedian: {median_dist:.2f}\nStd Dev: {std_dist:.2f}"
            axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.tight_layout()
    return fig

def analyze_pairwise_distances_by_class(df, output_dir="pairwise_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    embeddings = np.vstack(df['embedding'].values)
    labels = df['class'].values
    unique_labels = np.unique(labels)
    class_pair_distances, same_class_dists, diff_class_dists = analyze_class_pairwise_distances(embeddings, labels)
    distances_df = pd.DataFrame({
        'class_pair': [str(pair) for pair in class_pair_distances.keys()],
        'distances': [distances for distances in class_pair_distances.values()]
    })
    distances_df.to_pickle(os.path.join(output_dir, "class_pair_distances.pkl"))
    fig_overall = plt.figure(figsize=(12, 6))
    plt.hist(same_class_dists, bins=50, alpha=0.5, label='Same Class')
    plt.hist(diff_class_dists, bins=50, alpha=0.5, label='Different Classes')
    plt.title('Overall Pairwise Distance Distribution')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    fig_overall.savefig(os.path.join(output_dir, "overall_distance_distribution.png"))
    fig_heatmap = plot_class_distance_heatmap(class_pair_distances, unique_labels)
    fig_heatmap.savefig(os.path.join(output_dir, "class_distance_heatmap.png"))
    problem_pairs = identify_problematic_pairs(class_pair_distances, unique_labels)
    with open(os.path.join(output_dir, "problematic_pairs.txt"), "w") as f:
        f.write("Most similar different classes (potential confusion):\n")
        for (class_i, class_j), stats in problem_pairs['most_similar_different_classes']:
            f.write(f"Class {class_i} and Class {class_j}: Median distance = {stats['median']:.4f}\n")
        f.write("\nMost dispersed same classes (high intra-class variance):\n")
        for class_i, stats in problem_pairs['most_dispersed_same_classes']:
            f.write(f"Class {class_i}: Median internal distance = {stats['median']:.4f}\n")
    similar_pairs = [pair for pair, _ in problem_pairs['most_similar_different_classes'][:3]]
    fig_similar = plot_class_specific_distributions(class_pair_distances, similar_pairs)
    fig_similar.savefig(os.path.join(output_dir, "similar_different_classes.png"))
    dispersed_classes = [(c, c) for c, _ in problem_pairs['most_dispersed_same_classes'][:3]]
    fig_dispersed = plot_class_specific_distributions(class_pair_distances, dispersed_classes)
    fig_dispersed.savefig(os.path.join(output_dir, "dispersed_same_classes.png"))
    return {
        'class_pair_distances': class_pair_distances,
        'problematic_pairs': problem_pairs
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze pairwise distances by class')
    parser.add_argument('--input', type=str, required=True, help='Path to input DataFrame pickle with embeddings')
    parser.add_argument('--output', type=str, default='pairwise_analysis', help='Output directory for results')
    args = parser.parse_args()
    df = pd.read_pickle(args.input)
    analyze_pairwise_distances_by_class(df, output_dir=args.output) 