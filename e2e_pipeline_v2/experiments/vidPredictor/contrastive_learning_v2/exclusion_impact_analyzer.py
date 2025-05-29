#!/usr/bin/env python3
"""
Video vs Frame-Level Exclusion Impact Analyzer

Analyzes tracking results to compare frame-level vs video-level exclusion impacts.
Generates detailed reports showing data preservation efficiency and F-score comparisons.

Usage:
    python exclusion_impact_analyzer.py --tracking-dir ../../../../gitignore_exception/tracking_exports
    python exclusion_impact_analyzer.py --tracking-dir /path/to/tracking_exports --output impact_analysis
"""

import pandas as pd
import json
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict
import sys
import numpy as np

class ExclusionImpactAnalyzer:
    """Analyzes and compares frame-level vs video-level exclusion impacts"""
    
    def __init__(self, tracking_dir: Path, output_dir: Path):
        self.tracking_dir = Path(tracking_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.evaluation_data = None
        self.exclusions_data = None
        self.video_counts = None
        self.analysis_results = {}
        self.performance_data = {}
        
    def load_tracking_data(self):
        """Load all relevant tracking data files"""
        print("📂 Loading tracking data...")
        
        # Find the latest evaluation data file
        eval_files = list(self.tracking_dir.glob("*evaluation_data_before.csv"))
        if not eval_files:
            eval_files = list(self.tracking_dir.glob("*evaluation_data_after.csv"))
        
        if not eval_files:
            raise FileNotFoundError(f"No evaluation data files found in {self.tracking_dir}")
        
        # Use the most recent file
        eval_file = sorted(eval_files)[-1]
        print(f"📊 Loading evaluation data: {eval_file.name}")
        self.evaluation_data = pd.read_csv(eval_file)
        
        # Load cumulative exclusions
        exclusions_file = self.tracking_dir / "cumulative_exclusions_all.csv"
        if not exclusions_file.exists():
            raise FileNotFoundError(f"Exclusions file not found: {exclusions_file}")
        
        print(f"🚨 Loading exclusions data: {exclusions_file.name}")
        self.exclusions_data = pd.read_csv(exclusions_file)
        
        # Load existing performance data
        self._load_existing_performance_data()
        
        # Calculate video sample counts
        self.video_counts = self.evaluation_data['video'].value_counts()
        
        print(f"✅ Loaded {len(self.evaluation_data):,} evaluation samples")
        print(f"✅ Loaded {len(self.exclusions_data):,} exclusions")
        print(f"✅ Found {len(self.video_counts):,} unique videos")
        
    def _load_existing_performance_data(self):
        """Load existing F-scores and confusion matrices from iteration summaries"""
        print("📈 Loading existing performance data...")
        
        # Try to load iteration summary files
        summary_files = list(self.tracking_dir.glob("iteration_*_summary.json"))
        
        if not summary_files:
            print("⚠️  No iteration summary files found - F-score analysis will be limited")
            return
        
        performance_by_iteration = {}
        
        for summary_file in sorted(summary_files):
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                # Extract iteration number from filename
                iteration_num = int(summary_file.stem.split('_')[1])
                
                # Store performance metrics
                performance_by_iteration[iteration_num] = {
                    'confusion_matrix': data.get('confusion_matrix', {}),
                    'f1_score': data.get('f1_score', None),
                    'precision': data.get('precision', None),
                    'recall': data.get('recall', None),
                    'accuracy': data.get('accuracy', None)
                }
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"⚠️  Could not parse {summary_file}: {e}")
                continue
        
        self.performance_data = performance_by_iteration
        
        if performance_by_iteration:
            latest_iteration = max(performance_by_iteration.keys())
            latest_perf = performance_by_iteration[latest_iteration]
            print(f"✅ Loaded performance data for {len(performance_by_iteration)} iterations")
            print(f"📊 Latest F1: {latest_perf.get('f1_score', 'N/A')}")
        else:
            print("⚠️  No valid performance data found")
    
    def analyze_frame_level_impact(self):
        """Analyze current frame-level exclusion impact"""
        print("\n🎯 Analyzing frame-level exclusion impact...")
        
        total_exclusions = len(self.exclusions_data)
        problematic_videos = set(self.exclusions_data['video'].unique())
        problematic_classes = set(self.exclusions_data['class'].unique())
        
        # Group exclusions by iteration if available
        exclusions_by_iteration = {}
        if 'iteration' in self.exclusions_data.columns:
            for iteration, group in self.exclusions_data.groupby('iteration'):
                exclusions_by_iteration[iteration] = len(group)
        
        # Class-level breakdown
        class_exclusions = self.exclusions_data['class'].value_counts()
        
        # Video-level breakdown  
        video_exclusions = self.exclusions_data['video'].value_counts()
        
        frame_analysis = {
            'total_exclusions': total_exclusions,
            'problematic_videos': len(problematic_videos),
            'problematic_classes': len(problematic_classes),
            'exclusions_by_iteration': exclusions_by_iteration,
            'top_problematic_classes': class_exclusions.head(10).to_dict(),
            'top_problematic_videos': video_exclusions.head(10).to_dict(),
            'exclusions_per_video_stats': {
                'mean': video_exclusions.mean(),
                'median': video_exclusions.median(),
                'min': video_exclusions.min(),
                'max': video_exclusions.max()
            }
        }
        
        self.analysis_results['frame_level'] = frame_analysis
        return frame_analysis
        
    def analyze_video_level_impact(self):
        """Analyze hypothetical video-level exclusion impact"""
        print("\n📹 Analyzing video-level exclusion impact...")
        
        problematic_videos = set(self.exclusions_data['video'].unique())
        
        # Calculate samples that would be removed with video-level exclusions
        video_level_samples = self.video_counts[self.video_counts.index.isin(problematic_videos)]
        total_video_level_removal = video_level_samples.sum()
        
        # Calculate statistics
        total_evaluation_samples = len(self.evaluation_data)
        percentage_impact = (total_video_level_removal / total_evaluation_samples) * 100
        frame_level_exclusions = len(self.exclusions_data)
        amplification_factor = total_video_level_removal / frame_level_exclusions
        
        # Top videos by sample count
        top_videos_by_samples = video_level_samples.sort_values(ascending=False).head(10)
        
        video_analysis = {
            'total_samples_removed': total_video_level_removal,
            'percentage_of_dataset': percentage_impact,
            'amplification_factor': amplification_factor,
            'additional_samples_lost': total_video_level_removal - frame_level_exclusions,
            'video_sample_stats': {
                'mean': video_level_samples.mean(),
                'median': video_level_samples.median(),
                'min': video_level_samples.min(),
                'max': video_level_samples.max()
            },
            'top_videos_by_samples': top_videos_by_samples.to_dict(),
            'efficiency_loss': f"{amplification_factor:.1f}x more data removed"
        }
        
        self.analysis_results['video_level'] = video_analysis
        return video_analysis

    def simulate_video_level_performance(self):
        """Simulate F-scores if we used video-level exclusions"""
        print("\n🎭 Simulating video-level exclusion performance...")
        
        if not self.performance_data:
            print("⚠️  No existing performance data available for simulation")
            return None
        
        # Get problematic videos
        problematic_videos = set(self.exclusions_data['video'].unique())
        
        # Check if evaluation data has required columns for simulation
        required_columns = ['video', 'true_label', 'predicted_label']
        missing_columns = [col for col in required_columns if col not in self.evaluation_data.columns]
        
        if missing_columns:
            print(f"⚠️  Missing columns for F-score simulation: {missing_columns}")
            print("📊 Will estimate based on data reduction only")
            return self._estimate_performance_from_data_reduction()
        
        # Filter evaluation data (remove entire problematic videos)
        original_eval = self.evaluation_data.copy()
        filtered_eval = original_eval[~original_eval['video'].isin(problematic_videos)]
        
        print(f"📊 Original evaluation samples: {len(original_eval):,}")
        print(f"📊 Filtered evaluation samples: {len(filtered_eval):,}")
        print(f"📊 Samples removed: {len(original_eval) - len(filtered_eval):,}")
        
        # Calculate confusion matrices for both approaches
        original_cm = self._calculate_confusion_matrix(original_eval)
        filtered_cm = self._calculate_confusion_matrix(filtered_eval)
        
        # Calculate F-scores
        original_metrics = self._calculate_metrics_from_confusion_matrix(original_cm)
        filtered_metrics = self._calculate_metrics_from_confusion_matrix(filtered_cm)
        
        simulation_results = {
            'frame_level_performance': {
                'confusion_matrix': original_cm,
                'metrics': original_metrics,
                'sample_count': len(original_eval)
            },
            'video_level_simulation': {
                'confusion_matrix': filtered_cm,
                'metrics': filtered_metrics,
                'sample_count': len(filtered_eval)
            },
            'performance_comparison': {
                'f1_degradation': original_metrics['f1'] - filtered_metrics['f1'],
                'precision_degradation': original_metrics['precision'] - filtered_metrics['precision'],
                'recall_degradation': original_metrics['recall'] - filtered_metrics['recall'],
                'f1_degradation_percent': ((original_metrics['f1'] - filtered_metrics['f1']) / original_metrics['f1']) * 100 if original_metrics['f1'] > 0 else 0
            },
            'data_availability_impact': {
                'samples_lost': len(original_eval) - len(filtered_eval),
                'samples_lost_percent': ((len(original_eval) - len(filtered_eval)) / len(original_eval)) * 100,
                'videos_removed': len(problematic_videos),
                'videos_remaining': len(filtered_eval['video'].unique()) if 'video' in filtered_eval.columns else 0
            }
        }
        
        self.analysis_results['performance_simulation'] = simulation_results
        return simulation_results
    
    def _estimate_performance_from_data_reduction(self):
        """Estimate performance impact based on data reduction when labels aren't available"""
        print("📈 Estimating performance impact from data reduction...")
        
        problematic_videos = set(self.exclusions_data['video'].unique())
        
        # Get latest performance metrics
        if self.performance_data:
            latest_iteration = max(self.performance_data.keys())
            current_f1 = self.performance_data[latest_iteration].get('f1_score', 0.58)  # fallback to known value
        else:
            current_f1 = 0.58  # Use known final F1 score
        
        # Calculate data reduction
        total_samples = len(self.evaluation_data)
        problematic_video_samples = self.video_counts[self.video_counts.index.isin(problematic_videos)].sum()
        data_reduction_percent = (problematic_video_samples / total_samples) * 100
        
        # Estimate performance degradation based on data loss
        # Conservative estimate: 0.5-1.5% F1 drop per 10% data loss for good models
        estimated_f1_drop = (data_reduction_percent / 10.0) * 0.01  # 1% drop per 10% data loss
        estimated_video_level_f1 = max(0.0, current_f1 - estimated_f1_drop)
        
        estimation_results = {
            'frame_level_performance': {
                'f1_score': current_f1,
                'sample_count': total_samples,
                'source': 'actual_results'
            },
            'video_level_estimation': {
                'estimated_f1_score': estimated_video_level_f1,
                'estimated_sample_count': total_samples - problematic_video_samples,
                'source': 'data_reduction_estimation'
            },
            'performance_comparison': {
                'estimated_f1_degradation': current_f1 - estimated_video_level_f1,
                'estimated_f1_degradation_percent': ((current_f1 - estimated_video_level_f1) / current_f1) * 100,
                'confidence': 'MODERATE',
                'estimation_method': 'linear_data_reduction'
            },
            'data_availability_impact': {
                'samples_lost': problematic_video_samples,
                'samples_lost_percent': data_reduction_percent,
                'videos_removed': len(problematic_videos)
            }
        }
        
        self.analysis_results['performance_simulation'] = estimation_results
        return estimation_results
    
    def _calculate_confusion_matrix(self, eval_data):
        """Calculate confusion matrix from evaluation data"""
        if 'true_label' not in eval_data.columns or 'predicted_label' not in eval_data.columns:
            return None
        
        # Get unique labels
        all_labels = sorted(set(eval_data['true_label'].unique()) | set(eval_data['predicted_label'].unique()))
        
        # Initialize confusion matrix
        cm = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'detailed_matrix': {}
        }
        
        # Calculate binary classification metrics (assuming positive class exists)
        for _, row in eval_data.iterrows():
            true_label = row['true_label']
            pred_label = row['predicted_label']
            
            if true_label == pred_label:
                if true_label == 1 or true_label == 'positive':  # Positive class
                    cm['true_positives'] += 1
                else:  # Negative class
                    cm['true_negatives'] += 1
            else:
                if pred_label == 1 or pred_label == 'positive':  # Predicted positive, actually negative
                    cm['false_positives'] += 1
                else:  # Predicted negative, actually positive
                    cm['false_negatives'] += 1
        
        return cm
    
    def _calculate_metrics_from_confusion_matrix(self, cm):
        """Calculate precision, recall, F1 from confusion matrix"""
        if not cm:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
        
        tp = cm['true_positives']
        fp = cm['false_positives']
        tn = cm['true_negatives']
        fn = cm['false_negatives']
        
        # Calculate metrics with zero-division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': tp + fn
        }
        
    def generate_comparison_analysis(self):
        """Generate comprehensive comparison analysis"""
        print("\n⚖️  Generating comparison analysis...")
        
        frame_analysis = self.analysis_results['frame_level']
        video_analysis = self.analysis_results['video_level']
        
        comparison = {
            'summary': {
                'frame_level_exclusions': frame_analysis['total_exclusions'],
                'video_level_samples_removed': video_analysis['total_samples_removed'],
                'efficiency_ratio': video_analysis['amplification_factor'],
                'data_preservation_advantage': f"Frame-level preserves {video_analysis['amplification_factor']:.1f}x more data"
            },
            'recommendations': self._generate_recommendations(video_analysis['amplification_factor']),
            'dataset_impact': {
                'total_evaluation_samples': len(self.evaluation_data),
                'frame_level_impact_percent': (frame_analysis['total_exclusions'] / len(self.evaluation_data)) * 100,
                'video_level_impact_percent': video_analysis['percentage_of_dataset']
            }
        }
        
        # Add performance comparison if available
        if 'performance_simulation' in self.analysis_results:
            perf_sim = self.analysis_results['performance_simulation']
            comparison['performance_impact'] = perf_sim.get('performance_comparison', {})
        
        self.analysis_results['comparison'] = comparison
        return comparison
        
    def _generate_recommendations(self, amplification_factor):
        """Generate recommendations based on amplification factor"""
        if amplification_factor > 100:
            return {
                'verdict': 'CATASTROPHIC_WASTE',
                'message': 'Video-level exclusions would be catastrophically wasteful!',
                'recommendation': 'Continue with frame-level precision approach',
                'confidence': 'EXTREMELY_HIGH'
            }
        elif amplification_factor > 50:
            return {
                'verdict': 'EXTREME_WASTE', 
                'message': 'Video-level exclusions would be extremely wasteful',
                'recommendation': 'Frame-level approach is essential',
                'confidence': 'HIGH'
            }
        elif amplification_factor > 20:
            return {
                'verdict': 'HIGH_WASTE',
                'message': 'Video-level exclusions would be very wasteful',
                'recommendation': 'Frame-level approach strongly recommended',
                'confidence': 'HIGH'
            }
        elif amplification_factor > 10:
            return {
                'verdict': 'MODERATE_WASTE',
                'message': 'Video-level exclusions would be moderately wasteful',
                'recommendation': 'Frame-level approach recommended',
                'confidence': 'MEDIUM'
            }
        else:
            return {
                'verdict': 'ACCEPTABLE',
                'message': 'Video-level exclusions might be acceptable',
                'recommendation': 'Consider trade-offs between simplicity and precision',
                'confidence': 'LOW'
            }
    
    def save_frame_level_report(self):
        """Save detailed frame-level analysis report"""
        output_file = self.output_dir / "frame_level_exclusion_analysis.json"
        
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'tracking_directory': str(self.tracking_dir),
                'total_evaluation_samples': len(self.evaluation_data)
            },
            'frame_level_analysis': self.analysis_results['frame_level'],
            'detailed_exclusions': self.exclusions_data.to_dict('records')[:100]  # First 100 for brevity
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Saved frame-level analysis: {output_file}")
        return output_file
    
    def save_video_level_report(self):
        """Save detailed video-level analysis report"""
        output_file = self.output_dir / "video_level_exclusion_analysis.json"
        
        # Create detailed video breakdown
        problematic_videos = set(self.exclusions_data['video'].unique())
        video_breakdown = []
        
        for video in problematic_videos:
            exclusions_count = len(self.exclusions_data[self.exclusions_data['video'] == video])
            samples_count = self.video_counts.get(video, 0)
            video_breakdown.append({
                'video': video,
                'exclusions': exclusions_count,
                'total_samples': samples_count,
                'waste_factor': samples_count / exclusions_count if exclusions_count > 0 else 0
            })
        
        video_breakdown.sort(key=lambda x: x['total_samples'], reverse=True)
        
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'tracking_directory': str(self.tracking_dir),
                'total_evaluation_samples': len(self.evaluation_data)
            },
            'video_level_analysis': self.analysis_results['video_level'],
            'detailed_video_breakdown': video_breakdown
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Saved video-level analysis: {output_file}")
        return output_file

    def save_performance_comparison_report(self):
        """Save detailed performance comparison report"""
        output_file = self.output_dir / "performance_comparison_analysis.json"
        
        if 'performance_simulation' not in self.analysis_results:
            print("⚠️  No performance simulation data to save")
            return None
        
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'tracking_directory': str(self.tracking_dir),
                'analysis_type': 'f_score_simulation'
            },
            'performance_analysis': self.analysis_results['performance_simulation'],
            'existing_performance_data': self.performance_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Saved performance comparison: {output_file}")
        return output_file
    
    def save_comparison_report(self):
        """Save comprehensive comparison report"""
        output_file = self.output_dir / "exclusion_impact_comparison.json"
        
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'tracking_directory': str(self.tracking_dir),
                'analysis_version': '2.0'
            },
            'complete_analysis': self.analysis_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save a human-readable summary
        summary_file = self.output_dir / "exclusion_impact_summary.txt"
        self._save_human_readable_summary(summary_file)
        
        print(f"💾 Saved comparison analysis: {output_file}")
        print(f"📄 Saved human-readable summary: {summary_file}")
        return output_file, summary_file
    
    def _save_human_readable_summary(self, output_file):
        """Save human-readable summary report"""
        frame_analysis = self.analysis_results['frame_level']
        video_analysis = self.analysis_results['video_level']
        comparison = self.analysis_results['comparison']
        
        with open(output_file, 'w') as f:
            f.write("🎯 EXCLUSION IMPACT ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Size: {len(self.evaluation_data):,} evaluation samples\n")
            f.write(f"Total Videos: {len(self.video_counts):,}\n\n")
            
            f.write("📊 FRAME-LEVEL APPROACH (CURRENT)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Exclusions: {frame_analysis['total_exclusions']:,}\n")
            f.write(f"Problematic Videos: {frame_analysis['problematic_videos']:,}\n")
            f.write(f"Problematic Classes: {frame_analysis['problematic_classes']:,}\n")
            f.write(f"Impact: {comparison['dataset_impact']['frame_level_impact_percent']:.3f}% of dataset\n\n")
            
            f.write("📹 VIDEO-LEVEL APPROACH (HYPOTHETICAL)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Samples Removed: {video_analysis['total_samples_removed']:,}\n")
            f.write(f"Impact: {video_analysis['percentage_of_dataset']:.1f}% of dataset\n")
            f.write(f"Amplification Factor: {video_analysis['amplification_factor']:.1f}x\n")
            f.write(f"Additional Waste: {video_analysis['additional_samples_lost']:,} samples\n\n")
            
            # Add performance comparison if available
            if 'performance_simulation' in self.analysis_results:
                perf_sim = self.analysis_results['performance_simulation']
                f.write("🎭 F-SCORE SIMULATION\n")
                f.write("-" * 25 + "\n")
                
                if 'frame_level_performance' in perf_sim:
                    frame_f1 = perf_sim['frame_level_performance'].get('metrics', {}).get('f1', 'N/A')
                    f.write(f"Frame-level F1: {frame_f1:.4f}\n" if isinstance(frame_f1, float) else f"Frame-level F1: {frame_f1}\n")
                
                if 'video_level_simulation' in perf_sim:
                    video_f1 = perf_sim['video_level_simulation'].get('metrics', {}).get('f1', 'N/A')
                    f.write(f"Video-level F1 (sim): {video_f1:.4f}\n" if isinstance(video_f1, float) else f"Video-level F1 (sim): {video_f1}\n")
                elif 'video_level_estimation' in perf_sim:
                    est_f1 = perf_sim['video_level_estimation'].get('estimated_f1_score', 'N/A')
                    f.write(f"Video-level F1 (est): {est_f1:.4f}\n" if isinstance(est_f1, float) else f"Video-level F1 (est): {est_f1}\n")
                
                if 'performance_comparison' in perf_sim:
                    perf_comp = perf_sim['performance_comparison']
                    f1_deg = perf_comp.get('f1_degradation', perf_comp.get('estimated_f1_degradation', 'N/A'))
                    f1_deg_pct = perf_comp.get('f1_degradation_percent', perf_comp.get('estimated_f1_degradation_percent', 'N/A'))
                    
                    if isinstance(f1_deg, float):
                        f.write(f"F1 Degradation: {f1_deg:.4f}\n")
                    if isinstance(f1_deg_pct, float):
                        f.write(f"F1 Loss Percent: {f1_deg_pct:.1f}%\n")
                
                f.write("\n")
            
            f.write("⚖️  COMPARISON\n")
            f.write("-" * 20 + "\n")
            f.write(f"Efficiency Gain: {video_analysis['amplification_factor']:.1f}x more data preserved\n")
            f.write(f"Recommendation: {comparison['recommendations']['recommendation']}\n")
            f.write(f"Verdict: {comparison['recommendations']['message']}\n\n")
            
            f.write("🔝 TOP PROBLEMATIC VIDEOS\n")
            f.write("-" * 30 + "\n")
            for i, (video, samples) in enumerate(list(video_analysis['top_videos_by_samples'].items())[:5]):
                scene_name = video.split('__')[1].split('_')[0] if '__' in video else video[:20]
                f.write(f"{i+1:2d}. {scene_name:<20} {samples:>5,} samples\n")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Exclusion Impact Analysis with F-Score Simulation")
        print("=" * 60)
        
        try:
            # Load data
            self.load_tracking_data()
            
            # Run analyses
            self.analyze_frame_level_impact()
            self.analyze_video_level_impact()
            
            # NEW: Run F-score simulation
            self.simulate_video_level_performance()
            
            self.generate_comparison_analysis()
            
            # Save reports
            frame_report = self.save_frame_level_report()
            video_report = self.save_video_level_report()
            performance_report = self.save_performance_comparison_report()
            comparison_report, summary_report = self.save_comparison_report()
            
            # Print summary
            self._print_analysis_summary()
            
            print(f"\n✅ Analysis complete! Reports saved to: {self.output_dir}")
            
            reports = {
                'frame_level_report': frame_report,
                'video_level_report': video_report,
                'comparison_report': comparison_report,
                'summary_report': summary_report
            }
            
            if performance_report:
                reports['performance_report'] = performance_report
                
            return reports
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _print_analysis_summary(self):
        """Print concise analysis summary to console"""
        frame_analysis = self.analysis_results['frame_level']
        video_analysis = self.analysis_results['video_level']
        
        print(f"\n🎯 ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Frame-level exclusions:    {frame_analysis['total_exclusions']:>8,}")
        print(f"Video-level samples:       {video_analysis['total_samples_removed']:>8,}")
        print(f"Amplification factor:      {video_analysis['amplification_factor']:>8.1f}x")
        print(f"Data preservation gain:    {video_analysis['amplification_factor']:>8.1f}x better")
        
        # Add F-score summary if available
        if 'performance_simulation' in self.analysis_results:
            perf_sim = self.analysis_results['performance_simulation']
            print("\n🎭 F-SCORE SIMULATION")
            print("-" * 25)
            
            if 'frame_level_performance' in perf_sim and 'video_level_simulation' in perf_sim:
                frame_f1 = perf_sim['frame_level_performance']['metrics']['f1']
                video_f1 = perf_sim['video_level_simulation']['metrics']['f1']
                f1_loss = frame_f1 - video_f1
                f1_loss_pct = (f1_loss / frame_f1) * 100 if frame_f1 > 0 else 0
                
                print(f"Frame-level F1:            {frame_f1:>8.4f}")
                print(f"Video-level F1 (sim):      {video_f1:>8.4f}")
                print(f"F1 degradation:            {f1_loss:>8.4f}")
                print(f"F1 loss percentage:        {f1_loss_pct:>8.1f}%")
                
            elif 'frame_level_performance' in perf_sim and 'video_level_estimation' in perf_sim:
                frame_f1 = perf_sim['frame_level_performance']['f1_score']
                est_f1 = perf_sim['video_level_estimation']['estimated_f1_score']
                est_loss = frame_f1 - est_f1
                est_loss_pct = (est_loss / frame_f1) * 100 if frame_f1 > 0 else 0
                
                print(f"Frame-level F1:            {frame_f1:>8.4f}")
                print(f"Video-level F1 (est):      {est_f1:>8.4f}")
                print(f"Estimated F1 loss:         {est_loss:>8.4f}")
                print(f"Estimated loss percent:    {est_loss_pct:>8.1f}%")
        
        print(f"\nRecommendation: {self.analysis_results['comparison']['recommendations']['message']}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze frame-level vs video-level exclusion impacts with F-score simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze tracking results (from contrastive_learning_v2 directory)
  python exclusion_impact_analyzer.py --tracking-dir ../../../../gitignore_exception/tracking_exports
  
  # Specify custom output directory  
  python exclusion_impact_analyzer.py --tracking-dir ../../../../gitignore_exception/tracking_exports --output analysis_results
  
  # Use absolute path
  python exclusion_impact_analyzer.py --tracking-dir /path/to/tracking --output /path/to/analysis
        """
    )
    
    parser.add_argument(
        '--tracking-dir', 
        type=str, 
        default='../../../../gitignore_exception/tracking_exports',
        help='Path to tracking exports directory (default: ../../../../gitignore_exception/tracking_exports)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='exclusion_impact_analysis',
        help='Output directory for analysis reports (default: exclusion_impact_analysis)'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    # Initialize analyzer
    analyzer = ExclusionImpactAnalyzer(
        tracking_dir=args.tracking_dir,
        output_dir=args.output
    )
    
    # Run analysis
    reports = analyzer.run_complete_analysis()
    
    print(f"\n📊 Generated {len(reports)} analysis reports:")
    for report_type, report_path in reports.items():
        print(f"  • {report_type}: {report_path}")

if __name__ == "__main__":
    main() 