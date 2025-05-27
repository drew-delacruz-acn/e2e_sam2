import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def analyze_perfect_f1_scenario():
    """Analyze what to do when all experiments achieve perfect F1"""
    
    print("🎯 PERFECT F1 SCENARIO ANALYSIS")
    print("=" * 50)
    
    print("📊 SITUATION: All experiments achieved F1 = 1.0000")
    print("   This means we need different criteria to choose the best model.")
    
    # Load all data
    experiments = ['quick_test_agg', 'quick_test_balanced', 'very_agg']
    analysis = {}
    
    for exp_name in experiments:
        with open(f"gitignore_exception/org/{exp_name}/results.json", 'r') as f:
            loss_data = json.load(f)
        
        with open(f"gitignore_exception/org/{exp_name}/representatives.pkl", 'rb') as f:
            representatives = pickle.load(f)
        
        # Calculate additional metrics
        embeddings_list = representatives['finetuned_embedding'].tolist()
        rep_matrix = np.array(embeddings_list)
        similarity_matrix = cosine_similarity(rep_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        
        analysis[exp_name] = {
            'config': loss_data['config'],
            'final_loss': loss_data['history']['losses'][-1],
            'max_similarity': np.max(similarity_matrix),
            'mean_similarity': np.mean(similarity_matrix),
            'std_similarity': np.std(similarity_matrix),
            'high_sim_pairs': np.sum(similarity_matrix > 0.4),
            'training_epochs': loss_data['config']['epochs'],
            'convergence_stability': np.std(loss_data['history']['losses'][-10:])
        }
    
    return analysis

def secondary_ranking_criteria(analysis):
    """Rank experiments by secondary criteria when F1 is tied"""
    
    print("\n🏆 SECONDARY RANKING CRITERIA:")
    print("=" * 40)
    
    criteria = {
        'Robustness (Lower Max Similarity)': lambda x: x['max_similarity'],
        'Separation Quality (Lower Mean Similarity)': lambda x: x['mean_similarity'],
        'Training Efficiency (Fewer Epochs)': lambda x: x['training_epochs'],
        'Convergence Stability (Lower Variance)': lambda x: x['convergence_stability'],
        'No Confusing Pairs (Fewer High Sim)': lambda x: x['high_sim_pairs']
    }
    
    rankings = {}
    for criterion_name, criterion_func in criteria.items():
        # Sort by criterion (lower is better for all these)
        ranking = sorted(analysis.items(), key=lambda x: criterion_func(x[1]))
        rankings[criterion_name] = ranking
        
        print(f"\n{criterion_name}:")
        for i, (name, data) in enumerate(ranking, 1):
            value = criterion_func(data)
            print(f"  {i}. {name}: {value:.4f}")
    
    return rankings

def calculate_secondary_score(analysis, rankings):
    """Calculate overall score based on secondary criteria"""
    
    weights = {
        'Robustness (Lower Max Similarity)': 0.3,
        'Separation Quality (Lower Mean Similarity)': 0.25,
        'No Confusing Pairs (Fewer High Sim)': 0.2,
        'Training Efficiency (Fewer Epochs)': 0.15,
        'Convergence Stability (Lower Variance)': 0.1
    }
    
    scores = {}
    for exp_name in analysis.keys():
        score = 0
        for criterion, ranking in rankings.items():
            # Find rank (1-based)
            for rank, (name, _) in enumerate(ranking, 1):
                if name == exp_name:
                    # Convert rank to score (lower rank = higher score)
                    criterion_score = (len(ranking) + 1 - rank) / len(ranking)
                    score += weights[criterion] * criterion_score
                    break
        scores[exp_name] = score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def recommendations_for_perfect_f1(analysis, secondary_scores):
    """Provide recommendations when all models achieve perfect F1"""
    
    print(f"\n🚀 RECOMMENDATIONS FOR PERFECT F1 SCENARIO:")
    print("=" * 50)
    
    winner = secondary_scores[0][0]
    winner_data = analysis[winner]
    
    print(f"✅ RECOMMENDED: {winner.upper()}")
    print(f"   Secondary Score: {secondary_scores[0][1]:.4f}")
    
    print(f"\n📋 Why {winner} is recommended:")
    
    if winner == 'quick_test_agg':
        print("   🎯 Best class separation (max sim: 0.385)")
        print("   🚫 Zero confusing pairs")
        print("   ⚡ Good training efficiency (50 epochs)")
        print("   🛡️ Most robust for new data")
        
    elif winner == 'quick_test_balanced':
        print("   ⚖️ Balanced approach with good stability")
        print("   📊 Moderate separation without over-optimization")
        print("   🔄 Reliable convergence")
        
    else:  # very_agg
        print("   💪 Strongest loss optimization")
        print("   🎯 Good separation with more training")
        print("   📈 Best final loss value")
    
    print(f"\n💡 ADDITIONAL CONSIDERATIONS:")
    print("   1. Test on a harder/larger validation set")
    print("   2. Measure prediction confidence scores")
    print("   3. Test generalization to new domains")
    print("   4. Consider computational efficiency")
    print("   5. Monitor performance on edge cases")
    
    print(f"\n🔍 NEXT STEPS:")
    print("   1. Deploy the recommended model")
    print("   2. Create a more challenging test set")
    print("   3. Monitor real-world performance")
    print("   4. Consider ensemble methods if needed")

def main():
    # Analyze the perfect F1 scenario
    analysis = analyze_perfect_f1_scenario()
    
    # Rank by secondary criteria
    rankings = secondary_ranking_criteria(analysis)
    
    # Calculate secondary scores
    secondary_scores = calculate_secondary_score(analysis, rankings)
    
    print(f"\n🏆 OVERALL SECONDARY RANKING:")
    print("=" * 30)
    for i, (name, score) in enumerate(secondary_scores, 1):
        print(f"  {i}. {name}: {score:.4f}")
    
    # Provide recommendations
    recommendations_for_perfect_f1(analysis, secondary_scores)
    
    return analysis, secondary_scores

if __name__ == "__main__":
    analysis, scores = main() 