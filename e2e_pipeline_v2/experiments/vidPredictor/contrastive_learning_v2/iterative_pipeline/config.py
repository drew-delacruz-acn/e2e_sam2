from dataclasses import dataclass, field
from typing import List, Optional # Ensure Optional is imported if you use it for defaults

@dataclass
class PipelineConfig:
    # Data files
    definitive_objects: str
    resnet_predictions: str
    tracking_info: str

    # Pipeline parameters
    iterations: int = 5
    threshold: float = 0.6
    secondary_threshold: Optional[float] = None # Use Optional for clarity
    convergence_threshold: float = 0.001

    # Training parameters
    epochs: int = 50
    margin: float = 0.2
    secondary_margin: Optional[float] = None # Use Optional for clarity

    # Exclusion strategy options
    # These are mutually exclusive, argparse handles this.
    # We'll store the effective decision or the specific flag that was true.
    exclude_training_from_eval: bool = True # Default behavior
    include_training_in_eval: bool = False
    track_training_separately: bool = False
    
    # NEW: Exclusion level strategy
    exclusion_strategy: str = 'frame-level'  # 'frame-level', 'video-level', or 'compare-both'
    
    # Output options
    output: str = 'results_negative'
    test_mode: bool = False

    # Derived or helper attributes can be added in __post_init__ if needed
    def __post_init__(self):
        # Validate exclusion strategy
        valid_strategies = ['frame-level', 'video-level', 'compare-both']
        if self.exclusion_strategy not in valid_strategies:
            raise ValueError(f"exclusion_strategy must be one of {valid_strategies}, got '{self.exclusion_strategy}'")
        
        # If using compare-both, automatically enable track_training_separately
        if self.exclusion_strategy == 'compare-both':
            self.track_training_separately = True
    # def __post_init__(self):
    #     # Example: if you wanted to ensure output path is absolute
    #     self.output_path = Path(self.output).resolve()
    #     pass 