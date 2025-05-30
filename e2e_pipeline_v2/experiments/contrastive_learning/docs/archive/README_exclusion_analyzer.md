# 🎯 Exclusion Impact Analyzer

This script analyzes the impact of frame-level vs video-level exclusion approaches in your contrastive learning pipeline. It provides **exact calculations** of data preservation efficiency and **F-score comparisons** to answer "how much data would we waste with video-level exclusions?"

## 📋 **What It Does**

1. **📊 Frame-Level Analysis**: Analyzes current precise exclusion approach
2. **📹 Video-Level Analysis**: Calculates hypothetical video-level exclusion impact  
3. **🎭 F-Score Simulation**: Compares performance metrics between approaches
4. **📊 Confusion Matrix Analysis**: Detailed confusion matrix comparison
5. **📄 Multiple Reports**: Generates detailed JSON reports + human-readable summaries

## 🚀 **Usage**

### **Basic Usage (Default Paths)**
```bash
# From contrastive_learning_v2 directory
python exclusion_impact_analyzer.py
```

### **Custom Output Directory**
```bash
python exclusion_impact_analyzer.py --output my_analysis_results
```

### **Custom Tracking Directory**
```bash
python exclusion_impact_analyzer.py --tracking-dir /path/to/tracking_exports
```

### **Full Options**
```bash
python exclusion_impact_analyzer.py \
  --tracking-dir ../../../../gitignore_exception/tracking_exports \
  --output exclusion_analysis_2024
```

## 📂 **Generated Reports**

The script creates **8 output files**:

### **Core Analysis**
1. **`frame_level_exclusion_analysis.json`** - Detailed frame-level analysis
2. **`video_level_exclusion_analysis.json`** - Detailed video-level analysis  
3. **`exclusion_impact_comparison.json`** - Complete comparison analysis
4. **`exclusion_impact_summary.txt`** - Human-readable summary

### **Performance Analysis**
5. **`performance_comparison_analysis.json`** - F-score simulation results

### **Confusion Matrix Analysis** ⭐ NEW
6. **`confusion_matrices_analysis.json`** - Detailed confusion matrix data
7. **`confusion_matrices_comparison.csv`** - Spreadsheet-friendly comparison
8. **`confusion_matrices_summary.txt`** - Human-readable confusion matrix summary

## 📊 **Output Example**

```
🎯 ANALYSIS SUMMARY
========================================
Frame-level exclusions:          143
Video-level samples:           6,847
Amplification factor:           47.9x
Data preservation gain:         47.9x better

🎭 F-SCORE SIMULATION
-------------------------
Frame-level F1:            0.5799
Video-level F1 (est):      0.5434
Estimated F1 loss:         0.0365
Estimated loss percent:     6.3%

Recommendation: Video-level exclusions would be extremely wasteful
```

## 📊 **Confusion Matrix CSV Format**

The CSV report includes:
```csv
Metric,Frame-Level (Current),Video-Level (Simulated/Estimated),Difference,Change %
True Positives,78,65,-13,-16.7%
False Positives,55,42,-13,-23.6%
F1,0.5799,0.5434,-0.0365,-6.3%
```

## 🔧 **Requirements**

- **Tracking Data**: Needs `cumulative_exclusions_all.csv` and evaluation data files
- **Python Libraries**: pandas, json, pathlib, csv (standard with most Python installs)
- **Location**: Best placed in `contrastive_learning_v2/` directory

## 📍 **Integration Options**

### **Option 1: Manual Analysis** (Current)
```bash
# Run after pipeline completion
python exclusion_impact_analyzer.py
```

### **Option 2: Pipeline Integration**
Add to your pipeline script:
```python
from exclusion_impact_analyzer import ExclusionImpactAnalyzer

# At end of pipeline
analyzer = ExclusionImpactAnalyzer("tracking_exports", "impact_analysis")
analyzer.run_complete_analysis()
```

### **Option 3: VM/Server Integration**
```bash
# Add to your VM scripts
cd /path/to/contrastive_learning_v2
python exclusion_impact_analyzer.py --output /path/to/shared/results
```

## 🎯 **Key Metrics**

### **Data Efficiency**
- **Amplification Factor**: How many times more data video-level would remove
- **Waste Factor**: Additional samples lost with video-level approach
- **Efficiency Ratio**: Data preservation advantage of frame-level
- **Percentage Impact**: What portion of dataset gets affected

### **Performance Impact** ⭐ NEW
- **F1 Degradation**: Absolute F1 score loss
- **F1 Loss Percentage**: Relative performance degradation
- **Confusion Matrix Changes**: TP, FP, TN, FN differences
- **Precision/Recall Impact**: Individual metric degradation

## 💡 **Analysis Modes**

### **Full Simulation Mode**
- Uses actual evaluation labels if available
- Calculates precise confusion matrices
- Provides exact F-score comparisons

### **Estimation Mode**
- Uses data reduction when labels unavailable
- Conservative estimates (1% F1 drop per 10% data loss)
- Still provides valuable insights

## 📈 **Use Cases**

- **📊 Efficiency Justification**: Prove frame-level approach preserves 47.9x more data
- **📉 Performance Impact**: Show video-level degrades F1 by ~6.3%
- **📋 Decision Making**: Quantitative comparison for approach selection
- **📝 Reporting**: Export to CSV for presentations/papers
- **🔄 Monitoring**: Track exclusion efficiency across experiments

## 💡 **Tips**

- **Run after each experiment** to track exclusion efficiency
- **Compare results** across different pipeline configurations
- **Use CSV exports** for spreadsheet analysis and presentations
- **Monitor amplification factors** - >50x indicates extreme waste
- **Check confusion matrices** for detailed performance impact

## 🔄 **Automation**

To run automatically after each pipeline execution, add to your main script:

```python
# At the end of your pipeline
import subprocess
subprocess.run([
    "python", "exclusion_impact_analyzer.py", 
    "--output", f"impact_analysis_{experiment_name}"
])
```

## 📊 **Complete Workflow**

1. **Run Pipeline**: Execute your contrastive learning pipeline
2. **Generate Tracking**: Ensure tracking data is saved
3. **Run Analyzer**: Execute `python exclusion_impact_analyzer.py`
4. **Review Reports**: Check TXT summaries for quick insights
5. **Analyze Details**: Use JSON/CSV for detailed analysis
6. **Export Results**: Use CSV for presentations/papers