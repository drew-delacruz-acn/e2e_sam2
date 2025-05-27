import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def load_representatives(experiment_name):
    """Load representatives from a specific experiment"""
    path = f"gitignore_exception/org/{experiment_name}/representatives.pkl"
    
    with open(path, 'rb') as f:
        representatives = pickle.load(f)
    
    return representatives

def analyze_similarity_matrix(representatives, experiment_name):
    """Analyze the similarity matrix of representatives"""
    
    # Handle DataFrame format
    if hasattr(representatives, 'columns') and 'finetuned_embedding' in representatives.columns:
        class_names = representatives['class'].tolist()
        # Convert list of embeddings to numpy array
        embeddings_list = representatives['finetuned_embedding'].tolist()
        rep_matrix = np.array(embeddings_list)
    elif isinstance(representatives, dict):
        class_names = list(representatives.keys())
        rep_matrix = np.array([representatives[cls] for cls in class_names])
    else:
        # Assume it's already a matrix
        rep_matrix = representatives
        class_names = [f"Class_{i}" for i in range(len(rep_matrix))]
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(rep_matrix)
    
    # Remove diagonal (self-similarity)
    np.fill_diagonal(similarity_matrix, 0)
    
    # Calculate metrics
    max_similarity = np.max(similarity_matrix)
    mean_similarity = np.mean(similarity_matrix)
    std_similarity = np.std(similarity_matrix)
    
    # Find most similar pair
    max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    most_similar_pair = (class_names[max_idx[0]], class_names[max_idx[1]])
    
    # Find pairs with high similarity (>0.4)
    high_sim_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            if similarity_matrix[i, j] > 0.4:
                high_sim_pairs.append((
                    class_names[i], 
                    class_names[j], 
                    similarity_matrix[i, j]
                ))
    
    high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return {
        'experiment': experiment_name,
        'similarity_matrix': similarity_matrix,
        'class_names': class_names,
        'max_similarity': max_similarity,
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'most_similar_pair': most_similar_pair,
        'high_similarity_pairs': high_sim_pairs,
        'num_high_sim_pairs': len(high_sim_pairs)
    }

def plot_similarity_heatmaps(analyses):
    """Plot similarity heatmaps for all experiments"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (exp_name, analysis) in enumerate(analyses.items()):
        similarity_matrix = analysis['similarity_matrix']
        class_names = analysis['class_names']
        
        # Truncate class names for better display
        short_names = [name[:15] + "..." if len(name) > 15 else name for name in class_names]
        
        sns.heatmap(
            similarity_matrix, 
            annot=False, 
            cmap='RdYlBu_r', 
            center=0,
            vmin=0, 
            vmax=1,
            xticklabels=short_names,
            yticklabels=short_names,
            ax=axes[i]
        )
        
        axes[i].set_title(f'{exp_name}\nMax: {analysis["max_similarity"]:.3f}, Mean: {analysis["mean_similarity"]:.3f}')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig('similarity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(analyses):
    """Create a summary table comparing all experiments"""
    summary_data = []
    
    for exp_name, analysis in analyses.items():
        summary_data.append({
            'Experiment': exp_name,
            'Max Similarity': f"{analysis['max_similarity']:.4f}",
            'Mean Similarity': f"{analysis['mean_similarity']:.4f}",
            'Std Similarity': f"{analysis['std_similarity']:.4f}",
            'Most Similar Pair': f"{analysis['most_similar_pair'][0]} ↔ {analysis['most_similar_pair'][1]}",
            'High Sim Pairs (>0.4)': analysis['num_high_sim_pairs']
        })
    
    return pd.DataFrame(summary_data)

def main():
    print("🔍 ANALYZING REPRESENTATIVE SIMILARITIES")
    print("=" * 50)
    
    experiments = ['quick_test_agg', 'quick_test_balanced', 'very_agg']
    analyses = {}
    
    # Load and analyze each experiment
    for exp_name in experiments:
        print(f"\nLoading representatives from {exp_name}...")
        try:
            representatives = load_representatives(exp_name)
            analysis = analyze_similarity_matrix(representatives, exp_name)
            analyses[exp_name] = analysis
            print(f"✅ Loaded {len(analysis['class_names'])} classes")
        except Exception as e:
            print(f"❌ Error loading {exp_name}: {e}")
            continue
    
    if not analyses:
        print("❌ No experiments could be loaded!")
        return
    
    # Create summary table
    print("\n📊 SIMILARITY ANALYSIS SUMMARY:")
    print("=" * 80)
    summary_df = create_summary_table(analyses)
    print(summary_df.to_string(index=False))
    
    # Detailed analysis for each experiment
    print("\n🔍 DETAILED ANALYSIS:")
    print("=" * 50)
    
    for exp_name, analysis in analyses.items():
        print(f"\n📋 {exp_name.upper()}:")
        print(f"   Max Similarity: {analysis['max_similarity']:.4f}")
        print(f"   Mean Similarity: {analysis['mean_similarity']:.4f}")
        print(f"   Most Similar: {analysis['most_similar_pair'][0]} ↔ {analysis['most_similar_pair'][1]}")
        
        if analysis['high_similarity_pairs']:
            print(f"   High Similarity Pairs (>0.4): {len(analysis['high_similarity_pairs'])}")
            for pair in analysis['high_similarity_pairs'][:3]:  # Show top 3
                print(f"     - {pair[0]} ↔ {pair[1]}: {pair[2]:.4f}")
        else:
            print("   ✅ No pairs with similarity >0.4!")
    
    # Determine best separation
    print(f"\n🏆 BEST CLASS SEPARATION:")
    print("=" * 30)
    
    # Rank by max similarity (lower is better)
    max_sims = [(name, analysis['max_similarity']) for name, analysis in analyses.items()]
    max_sims.sort(key=lambda x: x[1])
    
    print("Ranking by Maximum Similarity (lower = better separation):")
    for i, (name, max_sim) in enumerate(max_sims, 1):
        print(f"  {i}. {name}: {max_sim:.4f}")
    
    # Rank by mean similarity (lower is better)
    mean_sims = [(name, analysis['mean_similarity']) for name, analysis in analyses.items()]
    mean_sims.sort(key=lambda x: x[1])
    
    print("\nRanking by Mean Similarity (lower = better separation):")
    for i, (name, mean_sim) in enumerate(mean_sims, 1):
        print(f"  {i}. {name}: {mean_sim:.4f}")
    
    # Overall winner
    best_max = max_sims[0][0]
    best_mean = mean_sims[0][0]
    
    if best_max == best_mean:
        print(f"\n🎯 CLEAR WINNER: {best_max}")
        print("   Best in both maximum and mean similarity!")
    else:
        print(f"\n🎯 WINNERS:")
        print(f"   Best Max Similarity: {best_max}")
        print(f"   Best Mean Similarity: {best_mean}")
    
    # Plot heatmaps
    print("\nGenerating similarity heatmaps...")
    plot_similarity_heatmaps(analyses)
    
    return analyses

if __name__ == "__main__":
    analyses = main() 