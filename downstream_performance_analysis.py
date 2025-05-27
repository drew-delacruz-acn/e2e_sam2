import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def analyze_downstream_vs_similarity():
    """Analyze the relationship between similarity metrics and downstream performance"""
    
    print("🔍 DOWNSTREAM PERFORMANCE vs SIMILARITY ANALYSIS")
    print("=" * 60)
    
    # Load results from all experiments
    experiments = ['quick_test_agg', 'quick_test_balanced', 'very_agg']
    analysis = {}
    
    for exp_name in experiments:
        # Load loss results (contains F1 scores)
        with open(f"gitignore_exception/org/{exp_name}/results.json", 'r') as f:
            loss_data = json.load(f)
        
        # Load representatives for similarity analysis
        with open(f"gitignore_exception/org/{exp_name}/representatives.pkl", 'rb') as f:
            representatives = pickle.load(f)
        
        # Calculate similarity metrics
        embeddings_list = representatives['finetuned_embedding'].tolist()
        rep_matrix = np.array(embeddings_list)
        similarity_matrix = cosine_similarity(rep_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        
        analysis[exp_name] = {
            'config': loss_data['config'],
            'final_f1': loss_data['metrics']['final_f1'],
            'baseline_f1': loss_data['metrics']['baseline_f1'],
            'f1_improvement': loss_data['metrics']['improvement'],
            'max_similarity': np.max(similarity_matrix),
            'mean_similarity': np.mean(similarity_matrix),
            'high_sim_pairs': np.sum(similarity_matrix > 0.4),
            'final_loss': loss_data['history']['losses'][-1]
        }
    
    return analysis

def create_performance_comparison(analysis):
    """Create a comparison focusing on downstream performance"""
    
    print("\n📊 PERFORMANCE COMPARISON TABLE:")
    print("=" * 80)
    
    df_data = []
    for exp_name, data in analysis.items():
        df_data.append({
            'Experiment': exp_name,
            'Margin': data['config']['margin'],
            'Lambda': data['config']['lambda_push'],
            'Final F1': f"{data['final_f1']:.4f}",
            'F1 Improvement': f"{data['f1_improvement']:.4f}",
            'Max Similarity': f"{data['max_similarity']:.4f}",
            'Mean Similarity': f"{data['mean_similarity']:.4f}",
            'High Sim Pairs': data['high_sim_pairs'],
            'Final Loss': f"{data['final_loss']:.4f}"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    return df

def analyze_performance_paradox(analysis):
    """Analyze why better similarity doesn't always mean better F1"""
    
    print(f"\n🤔 THE PERFORMANCE PARADOX:")
    print("=" * 40)
    
    # Rank by F1 performance
    f1_ranking = sorted(analysis.items(), key=lambda x: x[1]['final_f1'], reverse=True)
    
    # Rank by similarity (lower is theoretically better)
    sim_ranking = sorted(analysis.items(), key=lambda x: x[1]['max_similarity'])
    
    print("F1 Performance Ranking (higher is better):")
    for i, (name, data) in enumerate(f1_ranking, 1):
        print(f"  {i}. {name}: F1 = {data['final_f1']:.4f}")
    
    print("\nSimilarity Ranking (lower is theoretically better):")
    for i, (name, data) in enumerate(sim_ranking, 1):
        print(f"  {i}. {name}: Max Sim = {data['max_similarity']:.4f}")
    
    # Check if rankings match
    f1_winner = f1_ranking[0][0]
    sim_winner = sim_ranking[0][0]
    
    if f1_winner != sim_winner:
        print(f"\n⚠️  PARADOX DETECTED!")
        print(f"   F1 Winner: {f1_winner}")
        print(f"   Similarity Winner: {sim_winner}")
        print(f"   → Better separation ≠ Better classification!")
    
    return f1_winner, sim_winner

def explain_paradox(analysis, f1_winner, sim_winner):
    """Explain why the paradox occurs"""
    
    print(f"\n💡 WHY {f1_winner.upper()} WON IN DOWNSTREAM TASKS:")
    print("=" * 50)
    
    winner_data = analysis[f1_winner]
    sim_winner_data = analysis[sim_winner]
    
    print("Possible explanations:")
    
    # 1. Over-separation
    if winner_data['max_similarity'] > sim_winner_data['max_similarity']:
        print("\n1. 🎯 OPTIMAL SEPARATION ZONE")
        print(f"   - {f1_winner} found the 'sweet spot' for separation")
        print(f"   - Max similarity: {winner_data['max_similarity']:.4f} (not too low, not too high)")
        print(f"   - {sim_winner} may have over-separated classes")
        print(f"   - Over-separation can hurt classification by losing useful relationships")
    
    # 2. Parameter balance
    print(f"\n2. ⚖️ PARAMETER BALANCE")
    print(f"   - {f1_winner}: margin={winner_data['config']['margin']}, λ={winner_data['config']['lambda_push']}")
    print(f"   - {sim_winner}: margin={sim_winner_data['config']['margin']}, λ={sim_winner_data['config']['lambda_push']}")
    print(f"   - {f1_winner} parameters may better preserve classification-relevant features")
    
    # 3. Loss vs F1 relationship
    print(f"\n3. 📈 LOSS vs F1 RELATIONSHIP")
    print(f"   - Lower contrastive loss ≠ Better classification")
    print(f"   - {f1_winner} optimized for the right objective")
    print(f"   - Similarity metrics are proxies, not the end goal")
    
    # 4. Generalization
    print(f"\n4. 🎯 GENERALIZATION QUALITY")
    print(f"   - {f1_winner} representatives may generalize better to new samples")
    print(f"   - Extreme separation can reduce robustness")
    print(f"   - Moderate separation preserves intra-class variance")

def recommendations_based_on_f1(analysis, f1_winner):
    """Provide recommendations based on F1 performance"""
    
    print(f"\n🚀 UPDATED RECOMMENDATIONS:")
    print("=" * 30)
    
    winner_data = analysis[f1_winner]
    
    print(f"✅ USE {f1_winner.upper()} FOR PRODUCTION")
    print(f"   Configuration:")
    print(f"   - Margin: {winner_data['config']['margin']}")
    print(f"   - Lambda: {winner_data['config']['lambda_push']}")
    print(f"   - Learning Rate: {winner_data['config']['lr']}")
    print(f"   - Epochs: {winner_data['config']['epochs']}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   1. Downstream F1 is the ultimate metric")
    print(f"   2. Similarity metrics are useful but not definitive")
    print(f"   3. {f1_winner} found the optimal balance for your task")
    print(f"   4. Over-optimization on similarity can hurt performance")
    
    print(f"\n📊 MONITORING STRATEGY:")
    print(f"   - Always validate with downstream tasks")
    print(f"   - Use similarity as a guide, not the goal")
    print(f"   - Consider F1 score as the primary metric")
    print(f"   - Test on held-out validation set")

def main():
    # Analyze the data
    analysis = analyze_downstream_vs_similarity()
    
    # Create comparison table
    df = create_performance_comparison(analysis)
    
    # Analyze the paradox
    f1_winner, sim_winner = analyze_performance_paradox(analysis)
    
    # Explain why this happened
    explain_paradox(analysis, f1_winner, sim_winner)
    
    # Updated recommendations
    recommendations_based_on_f1(analysis, f1_winner)
    
    return analysis, f1_winner

if __name__ == "__main__":
    analysis, winner = main() 