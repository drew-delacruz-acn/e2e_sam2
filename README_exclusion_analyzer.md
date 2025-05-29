# 🎯 Exclusion Impact Analyzer

This script analyzes the impact of frame-level vs video-level exclusion approaches in your contrastive learning pipeline. It provides **exact calculations** of data preservation efficiency to answer "how much data would we waste with video-level exclusions?"

## 📋 **What It Does**

1. **📊 Frame-Level Analysis**: Analyzes current precise exclusion approach
2. **📹 Video-Level Analysis**: Calculates hypothetical video-level exclusion impact  
3. **⚖️ Comparison**: Compares efficiency and provides recommendations
4. **📄 Reports**: Generates detailed JSON reports + human-readable summaries

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

The script creates 4 output files:

1. **`frame_level_exclusion_analysis.json`** - Detailed frame-level analysis
2. **`video_level_exclusion_analysis.json`** - Detailed video-level analysis  
3. **`exclusion_impact_comparison.json`** - Complete comparison analysis
4. **`exclusion_impact_summary.txt`** - Human-readable summary

## 📊 **Output Example**

```
🎯 ANALYSIS SUMMARY
==============================
Frame-level exclusions:          143
Video-level samples:           6,847
Amplification factor:           47.9x
Data preservation gain:         47.9x better
Recommendation: Video-level exclusions would be extremely wasteful
```

## 🔧 **Requirements**

- **Tracking Data**: Needs `cumulative_exclusions_all.csv` and evaluation data files
- **Python Libraries**: pandas, json, pathlib (standard with most Python installs)
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

- **Amplification Factor**: How many times more data video-level would remove
- **Waste Factor**: Additional samples lost with video-level approach
- **Efficiency Ratio**: Data preservation advantage of frame-level
- **Percentage Impact**: What portion of dataset gets affected

## 💡 **Tips**

- **Run after each experiment** to track exclusion efficiency
- **Compare results** across different pipeline configurations
- **Use for justification** of frame-level precision approach
- **Monitor amplification factors** - >50x indicates extreme waste

## 🔄 **Automation**

To run automatically after each pipeline execution, add to your main script:

```python
# At the end of your pipeline
import subprocess
subprocess.run([
    "python", "exclusion_impact_analyzer.py", 
    "--output", f"impact_analysis_{experiment_name}"
])