import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

class IterationTracker:
    """Comprehensive tracking system for iterative pipeline exclusions and data flow."""
    
    def __init__(self, output_base_dir: Path):
        self.output_base_dir = Path(output_base_dir)
        self.tracking_dir = self.output_base_dir / "tracking_exports"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking storage
        self.data_flow_log = []
        self.exclusion_records = []
        self.iteration_summaries = []
        
        print(f"📊 Tracking system initialized. Exports will be saved to: {self.tracking_dir}")
    
    def log_data_flow_step(self, iteration: int, step: str, data_size: int, 
                          details: Optional[Dict] = None):
        """Log a step in the data flow process."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'step': step,
            'data_size': data_size,
            'details': details or {}
        }
        self.data_flow_log.append(record)
        print(f"📊 TRACKING | Iter {iteration} | {step}: {data_size} samples")
    
    def export_evaluation_data_before(self, iteration: int, data: pd.DataFrame):
        """Export evaluation data before filtering."""
        filename = f"iteration_{iteration}_evaluation_data_before.csv"
        filepath = self.tracking_dir / filename
        
        # Create a summary with key columns
        export_columns = ['video', 'frame', 'owl_label', 'finetuned_embedding']
        available_columns = [col for col in export_columns if col in data.columns]
        
        # Add embedding info without exporting the actual embedding
        export_data = data[available_columns].copy() if available_columns else data.copy()
        
        if 'finetuned_embedding' in export_data.columns:
            # Replace embedding with metadata
            export_data['embedding_shape'] = export_data['finetuned_embedding'].apply(
                lambda x: str(x.shape) if isinstance(x, np.ndarray) else str(type(x))
            )
            export_data['embedding_norm'] = export_data['finetuned_embedding'].apply(
                lambda x: float(np.linalg.norm(x)) if isinstance(x, np.ndarray) else 0.0
            )
            export_data = export_data.drop(columns=['finetuned_embedding'])
        
        export_data.to_csv(filepath, index=False)
        print(f"📊 Exported evaluation data before filtering: {filepath}")
        
        self.log_data_flow_step(iteration, "evaluation_data_before", len(data), {
            'columns': list(data.columns),
            'filename': filename
        })
    
    def export_evaluation_data_after(self, iteration: int, data: pd.DataFrame, 
                                   exclusions_applied: int):
        """Export evaluation data after filtering."""
        filename = f"iteration_{iteration}_evaluation_data_after.csv"
        filepath = self.tracking_dir / filename
        
        # Create a summary with key columns
        export_columns = ['video', 'frame', 'owl_label', 'finetuned_embedding']
        available_columns = [col for col in export_columns if col in data.columns]
        
        export_data = data[available_columns].copy() if available_columns else data.copy()
        
        if 'finetuned_embedding' in export_data.columns:
            export_data['embedding_shape'] = export_data['finetuned_embedding'].apply(
                lambda x: str(x.shape) if isinstance(x, np.ndarray) else str(type(x))
            )
            export_data['embedding_norm'] = export_data['finetuned_embedding'].apply(
                lambda x: float(np.linalg.norm(x)) if isinstance(x, np.ndarray) else 0.0
            )
            export_data = export_data.drop(columns=['finetuned_embedding'])
        
        export_data.to_csv(filepath, index=False)
        print(f"📊 Exported evaluation data after filtering: {filepath}")
        
        self.log_data_flow_step(iteration, "evaluation_data_after", len(data), {
            'exclusions_applied': exclusions_applied,
            'columns': list(data.columns),
            'filename': filename
        })
    
    def export_exclusions_added(self, iteration: int, exclusions: List[Dict]):
        """Export new exclusions added in this iteration."""
        if not exclusions:
            print(f"📊 No exclusions to export for iteration {iteration}")
            return
        
        filename = f"iteration_{iteration}_exclusions_added.csv"
        filepath = self.tracking_dir / filename
        
        # Convert exclusions to DataFrame
        exclusions_df = pd.DataFrame(exclusions)
        exclusions_df['iteration'] = iteration
        exclusions_df['timestamp'] = datetime.now().isoformat()
        
        # Reorder columns for better readability
        column_order = ['iteration', 'timestamp', 'video', 'frame', 'class', 
                       'original_wrong_prediction', 'reason']
        available_columns = [col for col in column_order if col in exclusions_df.columns]
        remaining_columns = [col for col in exclusions_df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        exclusions_df = exclusions_df[final_columns]
        exclusions_df.to_csv(filepath, index=False)
        print(f"📊 Exported {len(exclusions)} new exclusions: {filepath}")
        
        # Add to running log
        self.exclusion_records.extend(exclusions)
        
        self.log_data_flow_step(iteration, "exclusions_added", len(exclusions), {
            'filename': filename,
            'classes_excluded': list(exclusions_df['class'].value_counts().to_dict()) if 'class' in exclusions_df.columns else []
        })
    
    def export_false_positives_extracted(self, iteration: int, fp_data: pd.DataFrame, 
                                       evaluation_results: pd.DataFrame):
        """Export detailed false positive analysis."""
        filename = f"iteration_{iteration}_false_positives_extracted.csv"
        filepath = self.tracking_dir / filename
        
        # Get FP cases from evaluation results
        fp_cases = evaluation_results[evaluation_results['classification'] == 'FP'].copy() if 'classification' in evaluation_results.columns else pd.DataFrame()
        
        if fp_cases.empty:
            print(f"📊 No false positives found in iteration {iteration}")
            # Still create empty file for consistency
            empty_df = pd.DataFrame(columns=['iteration', 'video', 'frame', 'class', 'confidence', 'reason'])
            empty_df.to_csv(filepath, index=False)
            return
        
        # Enhance FP cases with additional info
        fp_cases['iteration'] = iteration
        fp_cases['extraction_timestamp'] = datetime.now().isoformat()
        fp_cases['reason'] = 'false_positive_for_training'
        
        # Add embedding info if available in fp_data - FIXED for capping
        if not fp_data.empty and 'class' in fp_data.columns:
            # Create a mapping from fp_data to match against fp_cases
            # fp_data contains negative classes like "not_TemPad", "not_TimeSpear"
            # fp_cases contains original classes like "TemPad", "TimeSpear"
            fp_data_classes = {}
            for _, row in fp_data.iterrows():
                negative_class = row['class']
                if negative_class.startswith('not_'):
                    original_class = negative_class[4:]  # Remove "not_" prefix
                    fp_data_classes[original_class] = negative_class
            
            # Map fp_cases to indicate which got negative classes created
            # This handles the length mismatch: fp_cases (88) vs fp_data (28)
            fp_cases['negative_class_created'] = fp_cases['class'].map(fp_data_classes)
            fp_cases['embedding_available'] = fp_cases['negative_class_created'].notna()
            
            # Debug info for verification
            processed_count = fp_cases['embedding_available'].sum()
            total_count = len(fp_cases)
            print(f"📊 FP Processing Summary: {processed_count}/{total_count} FPs processed for training (rest kept in test set)")
        else:
            fp_cases['negative_class_created'] = None
            fp_cases['embedding_available'] = False
        
        fp_cases.to_csv(filepath, index=False)
        print(f"📊 Exported {len(fp_cases)} false positives: {filepath}")
        
        self.log_data_flow_step(iteration, "false_positives_extracted", len(fp_cases), {
            'filename': filename,
            'classes_with_fps': list(fp_cases['class'].value_counts().to_dict()) if 'class' in fp_cases.columns else []
        })
    
    def export_iteration_summary(self, iteration: int, metrics: Dict, 
                               exclusions_count: int, training_data_size: int):
        """Export summary for this iteration."""
        summary = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'f1_score': metrics.get('f1', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'tp': metrics.get('TP', 0),
            'fp': metrics.get('FP', 0),
            'fn': metrics.get('FN', 0),
            'tn': metrics.get('TN', 0),
            'exclusions_added_this_iteration': exclusions_count,
            'training_data_size': training_data_size,
            'evaluation_samples': metrics.get('TP', 0) + metrics.get('FP', 0) + metrics.get('FN', 0) + metrics.get('TN', 0)
        }
        
        self.iteration_summaries.append(summary)
        
        # Export individual iteration summary
        filename = f"iteration_{iteration}_summary.json"
        filepath = self.tracking_dir / filename
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Exported iteration summary: {filepath}")
    
    def export_cumulative_analysis(self):
        """Export cumulative analysis across all iterations."""
        print(f"\n📊 Generating cumulative analysis...")
        
        # 1. All exclusions across iterations
        if self.exclusion_records:
            exclusions_df = pd.DataFrame(self.exclusion_records)
            exclusions_df.to_csv(self.tracking_dir / "cumulative_exclusions_all.csv", index=False)
            print(f"📊 Exported cumulative exclusions: {len(exclusions_df)} total exclusions")
            
            # 2. Class exclusion summary
            if 'class' in exclusions_df.columns:
                class_summary = exclusions_df['class'].value_counts().reset_index()
                class_summary.columns = ['class', 'exclusion_count']
                class_summary['percentage'] = (class_summary['exclusion_count'] / len(exclusions_df) * 100).round(2)
                class_summary.to_csv(self.tracking_dir / "class_exclusion_summary.csv", index=False)
                print(f"📊 Exported class exclusion summary: {len(class_summary)} classes")
            
            # 3. Video exclusion summary  
            if 'video' in exclusions_df.columns:
                video_summary = exclusions_df['video'].value_counts().reset_index()
                video_summary.columns = ['video', 'exclusion_count']
                video_summary['percentage'] = (video_summary['exclusion_count'] / len(exclusions_df) * 100).round(2)
                video_summary.to_csv(self.tracking_dir / "video_exclusion_summary.csv", index=False)
                print(f"📊 Exported video exclusion summary: {len(video_summary)} videos")
        
        # 4. Iteration summaries
        if self.iteration_summaries:
            summaries_df = pd.DataFrame(self.iteration_summaries)
            summaries_df.to_csv(self.tracking_dir / "iteration_summaries_all.csv", index=False)
            print(f"📊 Exported iteration summaries: {len(summaries_df)} iterations")
        
        # 5. Data flow log
        if self.data_flow_log:
            flow_df = pd.DataFrame(self.data_flow_log)
            flow_df.to_csv(self.tracking_dir / "data_flow_complete.csv", index=False)
            print(f"📊 Exported data flow log: {len(flow_df)} steps")
        
        # 6. Exclusion impact summary
        self._generate_exclusion_impact_summary()
        
        print(f"✅ Cumulative analysis complete. All files saved to: {self.tracking_dir}")
    
    def _generate_exclusion_impact_summary(self):
        """Generate summary of how exclusions impact data sizes across iterations."""
        if not self.data_flow_log:
            return
        
        flow_df = pd.DataFrame(self.data_flow_log)
        
        # Group by iteration and step
        impact_summary = []
        for iteration in sorted(flow_df['iteration'].unique()):
            iter_data = flow_df[flow_df['iteration'] == iteration]
            
            before_size = iter_data[iter_data['step'] == 'evaluation_data_before']['data_size'].iloc[0] if 'evaluation_data_before' in iter_data['step'].values else 0
            after_size = iter_data[iter_data['step'] == 'evaluation_data_after']['data_size'].iloc[0] if 'evaluation_data_after' in iter_data['step'].values else 0
            exclusions_added = iter_data[iter_data['step'] == 'exclusions_added']['data_size'].iloc[0] if 'exclusions_added' in iter_data['step'].values else 0
            fps_found = iter_data[iter_data['step'] == 'false_positives_extracted']['data_size'].iloc[0] if 'false_positives_extracted' in iter_data['step'].values else 0
            
            impact_summary.append({
                'iteration': iteration,
                'eval_data_before_filtering': before_size,
                'eval_data_after_filtering': after_size,
                'samples_filtered_out': before_size - after_size,
                'false_positives_found': fps_found,
                'new_exclusions_added': exclusions_added,
                'filtering_effectiveness': f"{((before_size - after_size) / before_size * 100):.1f}%" if before_size > 0 else "0%"
            })
        
        impact_df = pd.DataFrame(impact_summary)
        impact_df.to_csv(self.tracking_dir / "exclusion_impact_summary.csv", index=False)
        print(f"📊 Exported exclusion impact summary: {len(impact_df)} iterations")

# Global tracker instance (will be initialized by pipeline_manager)
tracker: Optional[IterationTracker] = None

def initialize_tracker(output_base_dir: Path):
    """Initialize the global tracker instance."""
    global tracker
    tracker = IterationTracker(output_base_dir)
    return tracker

def get_tracker() -> Optional[IterationTracker]:
    """Get the current tracker instance."""
    return tracker 