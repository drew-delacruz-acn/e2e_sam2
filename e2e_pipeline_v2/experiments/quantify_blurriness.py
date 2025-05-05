import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('blur_detection.log')
    ]
)
logger = logging.getLogger(__name__)

# Removed CUDA detection code since it's not needed

def laplacian_blur(gray):
    """Calculate blur score using Laplacian variance (higher = sharper)"""
    try:
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception as e:
        logger.error(f"Error in laplacian_blur: {e}")
        return 0

def tenengrad_blur(gray):
    """Calculate blur score using Tenengrad (Sobel gradient magnitude) (higher = sharper)"""
    try:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        g = np.sqrt(gx**2 + gy**2)
        return np.mean(g)
    except Exception as e:
        logger.error(f"Error in tenengrad_blur: {e}")
        return 0

def fft_blur(gray, cutoff_size=30):
    """Calculate blur score using FFT high-frequency energy (higher = sharper)"""
    try:
        start_time = time.time()
        
        # CPU version with NumPy
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        h, w = gray.shape
        cx, cy = w//2, h//2
        fshift[cy-cutoff_size:cy+cutoff_size, cx-cutoff_size:cx+cutoff_size] = 0
        ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(ishift)
        magnitude = np.abs(img_back)
        result = np.mean(magnitude)
            
        logger.debug(f"FFT calculation took {time.time() - start_time:.4f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error in fft_blur: {e}")
        return 0

# -- PARAMETERS --
LAPLACIAN_THRESH = 100
TENENGRAD_THRESH = 20
FFT_THRESH = 10

def get_video_info(cap):
    """Get and log information about the video"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    logger.info(f"Video dimensions: {width}x{height}")
    logger.info(f"FPS: {fps:.2f}")
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Duration: {duration:.2f} seconds")
    
    return {"width": width, "height": height, "fps": fps, 
            "frame_count": frame_count, "duration": duration}

def analyze_video(video_path, resize_factor=1.0, skip_frames=1):
    start_time = time.time()
    logger.info(f"Starting video analysis on: {video_path}")
    logger.info(f"Resize factor: {resize_factor}")
    logger.info(f"Processing every {skip_frames} frame(s)")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None, None, None, None, None, None
    
    # -- MAIN --
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Error opening video file")
            return None, None, None, None, None, None
            
        video_info = get_video_info(cap)
        blur_scores = []
        frame_idx = 0
        processed_count = 0
        processing_times = {
            'laplacian': [], 
            'tenengrad': [], 
            'fft': []
        }
        
        # Sample frames for visualization
        sample_frames = []
        sample_indices = []
        
        # For method comparison visualizations
        all_frames = {} # Store frames by category for later visualization
        
        # Suggest downscaling for large videos
        if video_info["width"] > 1920 and resize_factor == 1.0:
            logger.warning("High resolution video detected. Consider using resize_factor < 1.0 for faster processing")
            
        # Progress tracking variables
        total_frames = video_info["frame_count"]
        last_progress = -1
        progress_update_interval = max(1, total_frames // 20)  # Update progress ~20 times

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames if requested
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
                
            # Calculate and show progress
            progress = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
            if progress > last_progress and frame_idx % progress_update_interval == 0:
                elapsed = time.time() - start_time
                estimated_total = (elapsed / max(1, frame_idx)) * total_frames if frame_idx > 0 else 0
                remaining = max(0, estimated_total - elapsed)
                logger.info(f"Progress: {progress}% ({frame_idx}/{total_frames}) - ETA: {remaining:.1f}s")
                last_progress = progress

            frame_start = time.time()
            logger.debug(f"Processing frame {frame_idx}")
            
            try:
                # Save original frame for sample visualization
                if processed_count % (total_frames // min(10, total_frames // skip_frames or 1)) == 0:
                    sample_frames.append(frame.copy())
                    sample_indices.append(frame_idx)
                
                # Store all frames for method comparison visualization
                # Store a regular sampling of frames (about 20-30 total for performance)
                sample_interval = max(1, (total_frames // skip_frames) // 25)
                if processed_count % sample_interval == 0:
                    all_frames[frame_idx] = frame.copy()
                
                # Resize large frames if needed
                if resize_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Measure performance of each algorithm
                t1 = time.time()
                lap = laplacian_blur(gray)
                t2 = time.time()
                processing_times['laplacian'].append(t2-t1)
                
                t1 = time.time()
                ten = tenengrad_blur(gray)
                t2 = time.time()
                processing_times['tenengrad'].append(t2-t1)
                
                t1 = time.time()
                fft = fft_blur(gray)
                t2 = time.time()
                processing_times['fft'].append(t2-t1)

                blur_scores.append({
                    'frame': frame_idx,
                    'laplacian': lap,
                    'tenengrad': ten,
                    'fft': fft
                })
                
                processed_count += 1
                
                # Log detailed frame timing occasionally
                if processed_count % 10 == 0:
                    logger.debug(f"Frame {frame_idx} processed in {time.time() - frame_start:.4f} seconds")

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")
            
            frame_idx += 1

        cap.release()
        
        # Log performance stats
        logger.info(f"Processed {processed_count} frames out of {frame_idx} total frames")
        if processing_times['laplacian']:
            logger.info(f"Average processing times per frame:")
            logger.info(f"  Laplacian: {np.mean(processing_times['laplacian']):.4f} seconds")
            logger.info(f"  Tenengrad: {np.mean(processing_times['tenengrad']):.4f} seconds")
            logger.info(f"  FFT: {np.mean(processing_times['fft']):.4f} seconds")

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        return None, None, None, None, None, None

    # -- Calculate dynamic thresholds --
    logger.info("Calculating dynamic thresholds from statistics")
    # Extract values
    lap_vals = np.array([x['laplacian'] for x in blur_scores])
    ten_vals = np.array([x['tenengrad'] for x in blur_scores])
    fft_vals = np.array([x['fft'] for x in blur_scores])
    
    # Different statistical threshold methods
    thresholds = {
        # Method 1: Percentile-based
        'percentile': {
            'laplacian': np.percentile(lap_vals, 25),  # Lower 25% are blurry
            'tenengrad': np.percentile(ten_vals, 25),
            'fft': np.percentile(fft_vals, 25)
        },
        # Method 2: Mean - StdDev based
        'stddev': {
            'laplacian': max(0, np.mean(lap_vals) - 1.0 * np.std(lap_vals)),
            'tenengrad': max(0, np.mean(ten_vals) - 1.0 * np.std(ten_vals)),
            'fft': max(0, np.mean(fft_vals) - 1.0 * np.std(fft_vals))
        },
        # Method 3: Fixed thresholds (original)
        'fixed': {
            'laplacian': LAPLACIAN_THRESH,
            'tenengrad': TENENGRAD_THRESH,
            'fft': FFT_THRESH
        }
    }
    
    logger.info("Calculated thresholds:")
    for method, values in thresholds.items():
        logger.info(f"  {method.capitalize()} method:")
        for metric, value in values.items():
            logger.info(f"    {metric}: {value:.2f}")
    
    # -- TEMPORAL ANALYSIS with dynamic thresholds--
    logger.info("Starting temporal analysis with dynamic thresholds")
    # Use percentile method as default
    current_thresholds = thresholds['percentile']
    
    blur_flags = {
        'percentile': [], 
        'stddev': [], 
        'fixed': []
    }
    
    # Track frames by classification for each method
    method_classifications = {
        'percentile': {'sharp': [], 'blurry': []},
        'stddev': {'sharp': [], 'blurry': []},
        'fixed': {'sharp': [], 'blurry': []}
    }
    
    try:
        for i, scores in enumerate(blur_scores):
            lap, ten, fft = scores['laplacian'], scores['tenengrad'], scores['fft']
            frame_id = scores['frame']
            
            # Check each threshold method
            for method, thresh in thresholds.items():
                sharp = (
                    lap > thresh['laplacian'] and
                    ten > thresh['tenengrad'] and
                    fft > thresh['fft']
                )

                # Compare to neighbor average (temporal check)
                if 1 <= i < len(blur_scores) - 1:
                    prev = blur_scores[i-1]['laplacian']
                    nxt = blur_scores[i+1]['laplacian']
                    if lap < 0.5 * ((prev + nxt) / 2):
                        sharp = False
                        if method == 'percentile':  # Only log for the default method
                            logger.debug(f"Frame {scores['frame']} failed temporal check")

                blur_flags[method].append(not sharp)
                
                # Track frame classification for later visualization
                if frame_id in all_frames:
                    target_bucket = 'sharp' if sharp else 'blurry'
                    method_classifications[method][target_bucket].append(frame_id)
            
            # Log detailed analysis for some frames with the default method
            if i % 10 == 0 or not sharp:
                logger.info(f"Frame {scores['frame']:04d}: Sharp={sharp} | Lap={lap:.2f} | Ten={ten:.2f} | FFT={fft:.2f}")
    
    except Exception as e:
        logger.error(f"Error during temporal analysis: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"Video analysis completed in {total_time:.2f} seconds")
    if processed_count > 0:
        logger.info(f"Average time per frame: {total_time/processed_count:.4f} seconds")
    
    # Calculate blur statistics for each method
    for method, flags in blur_flags.items():
        if flags:
            blur_count = sum(flags)
            blur_percentage = (blur_count / len(flags)) * 100
            logger.info(f"{method.capitalize()} method: Found {blur_count} blurry frames out of {len(flags)} ({blur_percentage:.1f}%)")
    
    return blur_scores, blur_flags, thresholds, sample_frames, sample_indices, all_frames, method_classifications

def plot_blur_metrics(blur_scores, thresholds, save_path=None):
    if not blur_scores:
        logger.error("No blur scores to plot")
        return
        
    logger.info("Generating blur metrics plot")
    try:
        # -- PLOT BLUR METRICS --
        frames = [x['frame'] for x in blur_scores]
        lap_vals = [x['laplacian'] for x in blur_scores]
        ten_vals = [x['tenengrad'] for x in blur_scores]
        fft_vals = [x['fft'] for x in blur_scores]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        # Plot 1: Laplacian
        ax1.plot(frames, lap_vals, label='Laplacian Variance')
        for method, thresh in thresholds.items():
            ax1.axhline(y=thresh['laplacian'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                       linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold")
        ax1.set_title("Laplacian Variance (Edge Sharpness)")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Tenengrad
        ax2.plot(frames, ten_vals, label='Tenengrad', color='orange')
        for method, thresh in thresholds.items():
            ax2.axhline(y=thresh['tenengrad'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                       linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold")
        ax2.set_title("Tenengrad (Gradient Magnitude)")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FFT
        ax3.plot(frames, fft_vals, label='FFT Energy', color='green')
        for method, thresh in thresholds.items():
            ax3.axhline(y=thresh['fft'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                       linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold")
        ax3.set_title("FFT High-Frequency Energy")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Value")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
            
        # Generate histograms to help with threshold selection
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Laplacian histogram
        ax1.hist(lap_vals, bins=30, alpha=0.7)
        for method, thresh in thresholds.items():
            ax1.axvline(x=thresh['laplacian'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                      linestyle='--', label=f"{method.capitalize()}")
        ax1.set_title("Laplacian Distribution")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        
        # Tenengrad histogram
        ax2.hist(ten_vals, bins=30, alpha=0.7, color='orange')
        for method, thresh in thresholds.items():
            ax2.axvline(x=thresh['tenengrad'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                      linestyle='--', label=f"{method.capitalize()}")
        ax2.set_title("Tenengrad Distribution")
        ax2.set_xlabel("Value")
        ax2.legend()
        
        # FFT histogram
        ax3.hist(fft_vals, bins=30, alpha=0.7, color='green')
        for method, thresh in thresholds.items():
            ax3.axvline(x=thresh['fft'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                      linestyle='--', label=f"{method.capitalize()}")
        ax3.set_title("FFT Distribution")
        ax3.set_xlabel("Value")
        ax3.legend()
        
        plt.tight_layout()
        
        # Save histograms
        if save_path:
            histogram_path = save_path.replace('.png', '_histograms.png')
            plt.savefig(histogram_path)
            logger.info(f"Histograms saved to {histogram_path}")
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error generating plots: {e}")

def visualize_sample_frames(sample_frames, sample_indices, blur_scores, thresholds, save_dir=None):
    """Visualize sample frames with their blur metrics"""
    if not sample_frames or not blur_scores:
        logger.error("No sample frames or blur scores to visualize")
        return
        
    logger.info(f"Visualizing {len(sample_frames)} sample frames")
    
    try:
        # Get scores for the sample frames
        sample_scores = []
        for idx in sample_indices:
            for score in blur_scores:
                if score['frame'] == idx:
                    sample_scores.append(score)
                    break
        
        # Sort samples by Laplacian score (ascending)
        sorted_indices = np.argsort([score['laplacian'] for score in sample_scores])
        
        # Create a figure with sample frames
        n_samples = len(sample_frames)
        fig, axes = plt.subplots(n_samples, 1, figsize=(10, n_samples * 4))
        if n_samples == 1:
            axes = [axes]
            
        for i, idx in enumerate(sorted_indices):
            frame = sample_frames[idx]
            score = sample_scores[idx]
            frame_idx = sample_indices[idx]
            
            # Determine sharpness status based on percentile method
            sharp = (
                score['laplacian'] > thresholds['percentile']['laplacian'] and
                score['tenengrad'] > thresholds['percentile']['tenengrad'] and
                score['fft'] > thresholds['percentile']['fft']
            )
            
            # Display the frame
            axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Frame {frame_idx}: {'Sharp' if sharp else 'Blurry'}\n"
                             f"Laplacian: {score['laplacian']:.2f} (Threshold: {thresholds['percentile']['laplacian']:.2f})\n"
                             f"Tenengrad: {score['tenengrad']:.2f} (Threshold: {thresholds['percentile']['tenengrad']:.2f})\n"
                             f"FFT: {score['fft']:.2f} (Threshold: {thresholds['percentile']['fft']:.2f})")
            axes[i].axis('off')
            
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            samples_path = os.path.join(save_dir, 'sample_frames.png')
            plt.savefig(samples_path)
            logger.info(f"Sample frames saved to {samples_path}")
            plt.close()
            
            # Save individual frames with scores
            for i, idx in enumerate(sorted_indices):
                frame = sample_frames[idx]
                score = sample_scores[idx]
                frame_idx = sample_indices[idx]
                
                # Add text to frame with scores
                text_frame = frame.copy()
                h, w = text_frame.shape[:2]
                text_size = max(h / 720, 0.5)  # Scale text based on image size
                
                # Create a semi-transparent overlay for text background
                overlay = text_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, int(h/5)), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, text_frame, 0.4, 0, text_frame)
                
                # Add metrics text
                sharp = (
                    score['laplacian'] > thresholds['percentile']['laplacian'] and
                    score['tenengrad'] > thresholds['percentile']['tenengrad'] and
                    score['fft'] > thresholds['percentile']['fft']
                )
                
                status = f"{'SHARP' if sharp else 'BLURRY'}"
                cv2.putText(text_frame, f"Frame {frame_idx}: {status}", 
                           (10, int(30*text_size)), cv2.FONT_HERSHEY_SIMPLEX, 
                           text_size, (0, 255, 255) if sharp else (0, 0, 255), 2)
                
                cv2.putText(text_frame, 
                           f"Lap: {score['laplacian']:.1f} / Ten: {score['tenengrad']:.1f} / FFT: {score['fft']:.1f}", 
                           (10, int(70*text_size)), cv2.FONT_HERSHEY_SIMPLEX, 
                           text_size*0.8, (255, 255, 255), 2)
                
                frame_path = os.path.join(save_dir, f"frame_{frame_idx:04d}_{status.lower()}.jpg")
                cv2.imwrite(frame_path, text_frame)
                
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error visualizing sample frames: {e}")

def visualize_method_comparison(all_frames, method_classifications, thresholds, save_dir=None):
    """Create visualization showing how different methods classify the same frames"""
    if not all_frames or not method_classifications:
        logger.error("No frames or classifications available for method comparison")
        return
        
    logger.info(f"Creating method comparison visualization with {len(all_frames)} frames")
    
    try:
        methods = list(method_classifications.keys())
        
        # Create directories for each method if saving
        if save_dir:
            for method in methods:
                method_dir = os.path.join(save_dir, f"method_{method}")
                os.makedirs(os.path.join(method_dir, "sharp"), exist_ok=True)
                os.makedirs(os.path.join(method_dir, "blurry"), exist_ok=True)
                
            # Also create a comparison directory
            comparison_dir = os.path.join(save_dir, "method_comparison")
            os.makedirs(comparison_dir, exist_ok=True)
        
        # Generate visualizations for each method
        for method in methods:
            classifications = method_classifications[method]
            
            # Save individual images categorized by this method
            if save_dir:
                method_dir = os.path.join(save_dir, f"method_{method}")
                
                # Save sharp frames
                for frame_id in classifications['sharp']:
                    if frame_id in all_frames:
                        frame = all_frames[frame_id].copy()
                        h, w = frame.shape[:2]
                        text_size = max(h / 720, 0.5)
                        
                        # Add classification info
                        cv2.putText(frame, f"{method.capitalize()}: SHARP", 
                                   (10, int(30*text_size)), cv2.FONT_HERSHEY_SIMPLEX, 
                                   text_size, (0, 255, 0), 2)
                        
                        # Save frame
                        cv2.imwrite(os.path.join(method_dir, "sharp", f"frame_{frame_id:04d}.jpg"), frame)
                
                # Save blurry frames
                for frame_id in classifications['blurry']:
                    if frame_id in all_frames:
                        frame = all_frames[frame_id].copy()
                        h, w = frame.shape[:2]
                        text_size = max(h / 720, 0.5)
                        
                        # Add classification info
                        cv2.putText(frame, f"{method.capitalize()}: BLURRY", 
                                   (10, int(30*text_size)), cv2.FONT_HERSHEY_SIMPLEX, 
                                   text_size, (0, 0, 255), 2)
                        
                        # Save frame
                        cv2.imwrite(os.path.join(method_dir, "blurry", f"frame_{frame_id:04d}.jpg"), frame)
        
        # Create a multi-method comparison for selected frames
        # This shows how each method classifies the same frame
        common_frames = sorted(all_frames.keys())
        
        # Take a sample if we have too many frames
        if len(common_frames) > 12:
            # Sample frames across the range, focusing on disagreement cases
            disagreement_frames = []
            for frame_id in common_frames:
                # Check if methods disagree on this frame
                classifications = [method_classifications[m]['sharp' if frame_id in method_classifications[m]['sharp'] else 'blurry'] for m in methods]
                if len(set(tuple(c) for c in classifications)) > 1:  # If there's disagreement
                    disagreement_frames.append(frame_id)
            
            # If we have disagreement frames, prioritize them
            if disagreement_frames:
                if len(disagreement_frames) > 12:
                    common_frames = np.array(disagreement_frames)[np.linspace(0, len(disagreement_frames)-1, 12, dtype=int)]
                else:
                    common_frames = disagreement_frames
            else:
                # Otherwise, take evenly spaced samples
                common_frames = np.array(common_frames)[np.linspace(0, len(common_frames)-1, 12, dtype=int)]
        
        # Create comparison grid - rows are frames, columns are methods
        n_frames = len(common_frames)
        n_methods = len(methods)
        
        # Create a large figure
        fig, axes = plt.subplots(n_frames, n_methods, figsize=(n_methods*5, n_frames*4))
        
        # Handle single row or column case
        if n_frames == 1:
            axes = axes.reshape(1, -1)
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        # Track methods that frequently disagree with others
        disagreements = {m: 0 for m in methods}
        
        # Populate the grid
        for i, frame_id in enumerate(common_frames):
            frame = all_frames[frame_id]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get scores for this frame
            frame_scores = None
            for score in blur_scores:
                if score['frame'] == frame_id:
                    frame_scores = score
                    break
            
            # Find method classifications for this frame
            frame_classifications = {}
            for method in methods:
                if frame_id in method_classifications[method]['sharp']:
                    frame_classifications[method] = 'sharp'
                else:
                    frame_classifications[method] = 'blurry'
            
            # Count disagreements
            opinions = list(frame_classifications.values())
            for m_idx, method in enumerate(methods):
                # Check if this method disagrees with the majority
                if opinions.count(frame_classifications[method]) < len(methods)/2:
                    disagreements[method] += 1
            
            # Display each method's classification
            for j, method in enumerate(methods):
                ax = axes[i, j]
                
                # Display the frame
                ax.imshow(rgb_frame)
                
                # Add method-specific classification and threshold
                is_sharp = frame_id in method_classifications[method]['sharp']
                
                title_color = 'green' if is_sharp else 'red'
                classification = 'SHARP' if is_sharp else 'BLURRY'
                
                if frame_scores:
                    lap, ten, fft = frame_scores['laplacian'], frame_scores['tenengrad'], frame_scores['fft']
                    thresh = thresholds[method]
                    
                    # Format title with metrics and thresholds
                    title = f"{method.capitalize()}: {classification}\n"
                    title += f"Lap: {lap:.1f} > {thresh['laplacian']:.1f}\n"
                    title += f"Ten: {ten:.1f} > {thresh['tenengrad']:.1f}\n"
                    title += f"FFT: {fft:.1f} > {thresh['fft']:.1f}"
                else:
                    title = f"{method.capitalize()}: {classification}"
                
                ax.set_title(title, color=title_color)
                ax.axis('off')
                
                # Add a colored border for easy identification
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(title_color)
                    spine.set_linewidth(5)
        
        plt.tight_layout()
        
        # Save the comparison grid
        if save_dir:
            comparison_path = os.path.join(comparison_dir, "method_comparison_grid.png")
            plt.savefig(comparison_path)
            logger.info(f"Method comparison grid saved to {comparison_path}")
            plt.close()
            
            # Save information about method disagreements
            with open(os.path.join(comparison_dir, "method_analysis.txt"), 'w') as f:
                f.write("Method Disagreement Analysis:\n")
                f.write("Number of frames where each method disagrees with majority opinion:\n\n")
                
                for method, count in disagreements.items():
                    percentage = (count / len(common_frames)) * 100
                    f.write(f"{method.capitalize()}: {count}/{len(common_frames)} frames ({percentage:.1f}%)\n")
                
                # Add threshold information
                f.write("\nThresholds used by each method:\n\n")
                for method, thresh in thresholds.items():
                    f.write(f"{method.capitalize()}:\n")
                    for metric, value in thresh.items():
                        f.write(f"  {metric}: {value:.2f}\n")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error creating method comparison visualization: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Video blur detection with multiple methods')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--resize', type=float, default=1.0, help='Resize factor for video frames (e.g., 0.5 for half size)')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--save_path', type=str, help='Path to save the results plot')
    parser.add_argument('--method', type=str, default='percentile', choices=['percentile', 'stddev', 'fixed'], 
                        help='Threshold method to use for blur detection')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=" * 40)
        logger.info(f"Blur Detection Script started at {timestamp}")
        logger.info("=" * 40)
        
        # Get video path from args or use default
        video_path = args.video if args.video else "/Users/andrewdelacruz/e2e_sam2/data/Young_African_American_Woman_Headphones_1.mp4"
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
        else:
            results_dir = "blur_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Create a folder for this run
            run_dir = os.path.join(results_dir, f"analysis_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Run analysis with options from command line
            blur_scores, blur_flags, thresholds, sample_frames, sample_indices, all_frames, method_classifications = analyze_video(
                video_path, 
                resize_factor=args.resize,
                skip_frames=args.skip
            )
            
            if blur_scores:
                # Save results to file
                plot_path = args.save_path if args.save_path else os.path.join(run_dir, f"blur_plot_{timestamp}.png")
                plot_blur_metrics(blur_scores, thresholds, save_path=plot_path)
                
                # Visualize sample frames
                visualize_sample_frames(sample_frames, sample_indices, blur_scores, thresholds, save_dir=run_dir)
                
                # Create method comparison visualization
                visualize_method_comparison(all_frames, method_classifications, thresholds, save_dir=run_dir)
                
                # Save numerical results
                results_path = os.path.join(run_dir, f"blur_data_{timestamp}.csv")
                try:
                    with open(results_path, 'w') as f:
                        f.write("frame,laplacian,tenengrad,fft,is_blurry_percentile,is_blurry_stddev,is_blurry_fixed\n")
                        for i, score in enumerate(blur_scores):
                            f.write(f"{score['frame']},{score['laplacian']:.2f},{score['tenengrad']:.2f},{score['fft']:.2f},"
                                   f"{blur_flags['percentile'][i]},{blur_flags['stddev'][i]},{blur_flags['fixed'][i]}\n")
                    logger.info(f"Results saved to {results_path}")
                except Exception as e:
                    logger.error(f"Error saving results: {e}")
                
                # Save threshold information
                thresh_path = os.path.join(run_dir, f"thresholds_{timestamp}.txt")
                try:
                    with open(thresh_path, 'w') as f:
                        f.write("Calculated Thresholds:\n")
                        for method, values in thresholds.items():
                            f.write(f"{method.capitalize()} method:\n")
                            for metric, value in values.items():
                                f.write(f"  {metric}: {value:.2f}\n")
                        
                        # Add stats about blur detection
                        f.write("\nBlur Detection Results:\n")
                        for method, flags in blur_flags.items():
                            if flags:
                                blur_count = sum(flags)
                                blur_percentage = (blur_count / len(flags)) * 100
                                f.write(f"{method.capitalize()} method: Found {blur_count} blurry frames out of {len(flags)} ({blur_percentage:.1f}%)\n")
                    
                    logger.info(f"Threshold information saved to {thresh_path}")
                except Exception as e:
                    logger.error(f"Error saving threshold information: {e}")
            
        logger.info("=" * 40)
        logger.info("Blur Detection Script finished")
        logger.info("=" * 40)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")

