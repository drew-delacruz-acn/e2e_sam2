#!/usr/bin/env python3
"""
Video vs Frame-Level Exclusion Impact Analyzer

Analyzes tracking results to compare frame-level vs video-level exclusion impacts.
Generates detailed reports showing data preservation efficiency.

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
        
        # Calculate video sample counts
        self.video_counts = self.evaluation_data['video'].value_counts()
        
        print(f"✅ Loaded {len(self.evaluation_data):,} evaluation samples")
        print(f"✅ Loaded {len(self.exclusions_data):,} exclusions")
        print(f"✅ Found {len(self.video_counts):,} unique videos")
        
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
    
    def save_comparison_report(self):
        """Save comprehensive comparison report"""
        output_file = self.output_dir / "exclusion_impact_comparison.json"
        
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'tracking_directory': str(self.tracking_dir),
                'analysis_version': '1.0'
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
        print("🚀 Starting Exclusion Impact Analysis")
        print("=" * 50)
        
        try:
            # Load data
            self.load_tracking_data()
            
            # Run analyses
            self.analyze_frame_level_impact()
            self.analyze_video_level_impact()
            self.generate_comparison_analysis()
            
            # Save reports
            frame_report = self.save_frame_level_report()
            video_report = self.save_video_level_report()
            comparison_report, summary_report = self.save_comparison_report()
            
            # Print summary
            self._print_analysis_summary()
            
            print(f"\n✅ Analysis complete! Reports saved to: {self.output_dir}")
            return {
                'frame_level_report': frame_report,
                'video_level_report': video_report,
                'comparison_report': comparison_report,
                'summary_report': summary_report
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            sys.exit(1)
    
    def _print_analysis_summary(self):
        """Print concise analysis summary to console"""
        frame_analysis = self.analysis_results['frame_level']
        video_analysis = self.analysis_results['video_level']
        
        print(f"\n🎯 ANALYSIS SUMMARY")
        print("=" * 30)
        print(f"Frame-level exclusions:    {frame_analysis['total_exclusions']:>8,}")
        print(f"Video-level samples:       {video_analysis['total_samples_removed']:>8,}")
        print(f"Amplification factor:      {video_analysis['amplification_factor']:>8.1f}x")
        print(f"Data preservation gain:    {video_analysis['amplification_factor']:>8.1f}x better")
        print(f"Recommendation: {self.analysis_results['comparison']['recommendations']['message']}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze frame-level vs video-level exclusion impacts",
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