import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from datetime import datetime
import argparse
import json # Import json module
import glob # For finding image files

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

def laplacian_blur(gray, mask=None):
    """Calculate blur score using Laplacian variance (higher = sharper)"""
    try:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        if mask is not None and cv2.countNonZero(mask) > 0:
            # Calculate variance only on the masked pixels
            masked_lap = lap[mask == 255]
            if masked_lap.size == 0: # Check if mask resulted in empty selection
                 logger.warning("Laplacian mask resulted in empty selection. Using full frame.")
                 return lap.var()
            return masked_lap.var()
        else:
            return lap.var()
    except Exception as e:
        logger.error(f"Error in laplacian_blur: {e}")
        return 0

def tenengrad_blur(gray, mask=None):
    """Calculate blur score using Tenengrad (Sobel gradient magnitude) (higher = sharper)"""
    try:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        g = np.sqrt(gx**2 + gy**2)
        if mask is not None and cv2.countNonZero(mask) > 0:
            # Calculate mean only on the masked pixels
            masked_g = g[mask == 255]
            if masked_g.size == 0:
                 logger.warning("Tenengrad mask resulted in empty selection. Using full frame.")
                 return np.mean(g)
            return np.mean(masked_g)
        else:
            return np.mean(g)
    except Exception as e:
        logger.error(f"Error in tenengrad_blur: {e}")
        return 0

def fft_blur(gray, cutoff_size=30, mask=None):
    """Calculate blur score using FFT high-frequency energy (higher = sharper)"""
    try:
        start_time = time.time()
        
        # CPU version with NumPy
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        h, w = gray.shape
        cx, cy = w//2, h//2
        # Create a low-pass filter mask (zeros in center, ones elsewhere)
        fft_mask = np.ones((h, w), dtype=np.uint8)
        fft_mask[cy-cutoff_size:cy+cutoff_size, cx-cutoff_size:cx+cutoff_size] = 0
        # Apply the low-pass filter mask by setting low frequencies to zero
        fshift_filtered = fshift * fft_mask
        
        # Shift back and inverse FFT
        ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(ishift)
        magnitude = np.abs(img_back)
        
        if mask is not None and cv2.countNonZero(mask) > 0:
             # Calculate mean magnitude only on the masked pixels
            masked_magnitude = magnitude[mask == 255]
            if masked_magnitude.size == 0:
                 logger.warning("FFT mask resulted in empty selection. Using full frame.")
                 result = np.mean(magnitude)
            else:
                 result = np.mean(masked_magnitude)
        else:
            result = np.mean(magnitude)
            
        logger.debug(f"FFT calculation took {time.time() - start_time:.4f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error in fft_blur: {e}")
        return 0

def extract_foreground_mask(frame, edge_threshold=100, density_threshold=0.3):
    """Extract foreground mask based on edge density"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, edge_threshold // 2, edge_threshold)
        
        # Create edge density map
        kernel = np.ones((5,5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        edge_density = cv2.GaussianBlur(dilated_edges, (21, 21), 0)
        
        # Normalize and threshold
        edge_density = edge_density / 255.0
        mask = (edge_density > density_threshold).astype(np.uint8) * 255
        
        # Clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Ensure mask dimensions match gray image
        if mask.shape != gray.shape:
            logger.warning(f"Mask shape {mask.shape} differs from gray shape {gray.shape}. Resizing mask.")
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        return mask, edge_density, gray, edges # Return intermediate steps
    except Exception as e:
        logger.error(f"Error in extract_foreground_mask: {e}")
        return None, None, None, None

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

def analyze_video(video_path, resize_factor=1.0, skip_frames=1, bg_remove=False, visualize_mask=False, run_dir="."):
    start_time = time.time()
    logger.info(f"Starting video analysis on: {video_path}")
    logger.info(f"Resize factor: {resize_factor}")
    logger.info(f"Processing every {skip_frames} frame(s)")
    logger.info(f"Background removal based on edge density: {'Enabled' if bg_remove else 'Disabled'}")
    if bg_remove and visualize_mask:
        logger.info("Foreground mask visualization: Enabled (will save for first valid frame)")
    
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

        mask_visualization_saved = False # Flag to save only one visualization

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
                
                # --- Foreground Extraction (if enabled) ---
                mask = None
                if bg_remove:
                    mask, edge_density_map, gray_for_mask, edges_for_mask = extract_foreground_mask(frame) # Get intermediate steps
                    if mask is None or cv2.countNonZero(mask) == 0:
                        logger.warning(f"Frame {frame_idx}: Foreground mask invalid or empty. Analyzing full frame.")
                        mask = None # Ensure mask is None if invalid
                    else:
                        logger.debug(f"Frame {frame_idx}: Applied foreground mask.")
                        # --- Save Mask Visualization (if enabled and not already saved) ---
                        if visualize_mask and not mask_visualization_saved:
                             try:
                                 fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                                 
                                 # Original Frame
                                 axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                 axes[0, 0].set_title("Original Frame")
                                 axes[0, 0].axis('off')
                                 
                                 # Canny Edges
                                 axes[0, 1].imshow(edges_for_mask, cmap='gray')
                                 axes[0, 1].set_title("Canny Edges")
                                 axes[0, 1].axis('off')
                                 
                                 # Edge Density Map
                                 im = axes[1, 0].imshow(edge_density_map, cmap='hot')
                                 axes[1, 0].set_title("Edge Density Map")
                                 axes[1, 0].axis('off')
                                 fig.colorbar(im, ax=axes[1, 0])
                                 
                                 # Final Mask
                                 axes[1, 1].imshow(mask, cmap='gray')
                                 axes[1, 1].set_title("Final Foreground Mask")
                                 axes[1, 1].axis('off')
                                 
                                 plt.tight_layout()
                                 vis_save_path = os.path.join(run_dir, f"foreground_mask_visualization_frame_{frame_idx}.png")
                                 plt.savefig(vis_save_path)
                                 plt.close(fig)
                                 logger.info(f"Saved foreground mask visualization to {vis_save_path}")
                                 mask_visualization_saved = True # Set flag so we don't save again
                             except Exception as e:
                                 logger.error(f"Error saving mask visualization for frame {frame_idx}: {e}")

                # Measure performance of each algorithm (with potential mask)
                t1 = time.time()
                lap = laplacian_blur(gray, mask=mask)
                t2 = time.time()
                processing_times['laplacian'].append(t2-t1)
                
                t1 = time.time()
                ten = tenengrad_blur(gray, mask=mask)
                t2 = time.time()
                processing_times['tenengrad'].append(t2-t1)
                
                t1 = time.time()
                fft = fft_blur(gray, mask=mask)
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

def plot_blur_metrics(blur_scores, thresholds, save_path=None, is_frames_mode=False):
    """Generate plots for blur metrics (works for video frames or image sequences)."""
    if not blur_scores:
        logger.error("No blur scores to plot")
        return
        
    logger.info("Generating blur metrics plot")
    try:
        # -- PLOT BLUR METRICS --
        # Use 'index' for frames mode, 'frame' for video mode
        x_values = [x['index' if is_frames_mode else 'frame'] for x in blur_scores]
        x_label = "Image Index" if is_frames_mode else "Frame"
        
        lap_vals = [x['laplacian'] for x in blur_scores]
        ten_vals = [x['tenengrad'] for x in blur_scores]
        fft_vals = [x['fft'] for x in blur_scores]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        # Plot 1: Laplacian
        ax1.plot(x_values, lap_vals, label='Laplacian Variance')
        for method, thresh in thresholds.items():
            ax1.axhline(y=thresh['laplacian'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                       linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold")
        ax1.set_title("Laplacian Variance (Edge Sharpness)")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Tenengrad
        ax2.plot(x_values, ten_vals, label='Tenengrad', color='orange')
        for method, thresh in thresholds.items():
            ax2.axhline(y=thresh['tenengrad'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                       linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold")
        ax2.set_title("Tenengrad (Gradient Magnitude)")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FFT
        ax3.plot(x_values, fft_vals, label='FFT Energy', color='green')
        for method, thresh in thresholds.items():
            ax3.axhline(y=thresh['fft'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'), 
                       linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold")
        ax3.set_title("FFT High-Frequency Energy")
        ax3.set_xlabel(x_label) # Use dynamic x-axis label
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

def calculate_blur_confidence(score, all_scores, thresholds, method='percentile'):
    """
    Calculate confidence score for blur classification (-100 to +100)
    Negative = blurry, Positive = sharp, Magnitude = confidence
    """
    # Extract values and thresholds
    lap, ten, fft = score['laplacian'], score['tenengrad'], score['fft']
    lap_thresh = thresholds[method]['laplacian']
    ten_thresh = thresholds[method]['tenengrad']
    fft_thresh = thresholds[method]['fft']
    
    # Get ranges from all scores for normalization
    all_lap = np.array([s['laplacian'] for s in all_scores])
    all_ten = np.array([s['tenengrad'] for s in all_scores])
    all_fft = np.array([s['fft'] for s in all_scores])
    
    # Calculate normalized distances from thresholds (-1 to +1)
    lap_conf = normalize_to_range(lap - lap_thresh, all_lap - lap_thresh)
    ten_conf = normalize_to_range(ten - ten_thresh, all_ten - ten_thresh)
    fft_conf = normalize_to_range(fft - fft_thresh, all_fft - fft_thresh)
    
    # Calculate overall confidence (weighted average)
    weights = [0.5, 0.3, 0.2]  # Laplacian more important
    overall_conf = weights[0] * lap_conf + weights[1] * ten_conf + weights[2] * fft_conf
    
    # Scale to percentage (-100 to +100)
    confidence = int(100 * overall_conf)
    
    # Determine most influential metric
    influences = [abs(lap_conf * weights[0]), abs(ten_conf * weights[1]), abs(fft_conf * weights[2])]
    most_influential = ['laplacian', 'tenengrad', 'fft'][np.argmax(influences)]
    
    return confidence, most_influential

def normalize_to_range(value, distribution):
    """Normalize a value to -1 to +1 range based on distribution"""
    p10, p90 = np.percentile(distribution, [10, 90])
    range_half = max(abs(p10), abs(p90))
    if range_half == 0:
        return 0
    return np.clip(value / range_half, -1, 1)

def generate_executive_summary(blur_scores, blur_flags, thresholds, all_frames, method_classifications, selected_method='percentile', save_dir=None):
    """Create comprehensive executive summary of blur detection methodology"""
    logger.info("Generating executive summary for stakeholders")
    
    if not save_dir:
        logger.error("Save directory required for executive summary")
        return None, None, None
        
    # Try to import seaborn for nicer plots
    try:
        import seaborn as sns
        have_seaborn = True
    except ImportError:
        logger.warning("Seaborn not installed. Install with 'pip install seaborn' for better visualizations.")
        have_seaborn = False
        
    summary_dir = os.path.join(save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Calculate confidence scores for all frames
    confidence_scores = []
    for i, score in enumerate(blur_scores):
        conf, metric = calculate_blur_confidence(score, blur_scores, thresholds, selected_method)
        confidence_scores.append({
            'frame': score['frame'],
            'confidence': conf,
            'leading_metric': metric,
            'is_blurry': blur_flags[selected_method][i]
        })
    
    # Create main summary figure (2x2 grid)
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Distribution plot with thresholds
    ax1 = fig.add_subplot(gs[0, 0])
    create_distribution_plot(ax1, blur_scores, thresholds, selected_method, have_seaborn)
    
    # 2. Confidence distribution
    ax2 = fig.add_subplot(gs[0, 1])
    create_confidence_histogram(ax2, confidence_scores)
    
    # 3. Sample frames at different confidence levels
    ax3 = fig.add_subplot(gs[1, 0])
    display_confidence_examples(ax3, blur_scores, confidence_scores, all_frames)
    
    # 4. Show comparison pairs explaining classification differences
    ax4 = fig.add_subplot(gs[1, 1])
    create_comparison_visualization(ax4, blur_scores, confidence_scores, all_frames, thresholds, selected_method)
    
    # Add title and save
    fig.suptitle(f"Blur Detection Methodology: {selected_method.capitalize()} Method", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    summary_path = os.path.join(summary_dir, "executive_summary.png")
    plt.savefig(summary_path)
    logger.info(f"Executive summary saved to {summary_path}")
    plt.close(fig)
    
    # Create confidence-scored CSV
    csv_path = os.path.join(summary_dir, "blur_scores_with_confidence.csv")
    with open(csv_path, 'w') as f:
        f.write("frame,laplacian,tenengrad,fft,is_blurry,confidence,leading_metric\n")
        for i, score in enumerate(blur_scores):
            conf_data = confidence_scores[i]
            f.write(f"{score['frame']},{score['laplacian']:.2f},{score['tenengrad']:.2f},{score['fft']:.2f},"
                   f"{conf_data['is_blurry']},{conf_data['confidence']},{conf_data['leading_metric']}\n")
    
    # Generate a text report with methodology explanation
    report_path = os.path.join(summary_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("BLUR DETECTION METHODOLOGY SUMMARY\n")
        f.write("==================================\n\n")
        f.write(f"Method: {selected_method.capitalize()}\n\n")
        f.write("How Blur is Detected:\n")
        f.write("1. Each frame is analyzed using three metrics:\n")
        f.write("   - Laplacian Variance: Measures edge intensity (higher = sharper)\n")
        f.write("   - Tenengrad: Measures gradient magnitude (higher = sharper)\n")
        f.write("   - FFT: Measures high-frequency content (higher = sharper)\n\n")
        f.write("2. Thresholds are calculated based on the selected method:\n")
        for metric, value in thresholds[selected_method].items():
            f.write(f"   - {metric}: {value:.2f}\n")
        f.write("\n3. A frame is classified as blurry when ALL metrics fall below their thresholds\n")
        f.write("4. Additionally, a temporal check flags frames with significant drops compared to neighbors\n\n")
        f.write("5. Confidence scores (-100 to +100) indicate how far from thresholds a frame's metrics are\n")
        f.write("   - Negative scores = blurry, Positive scores = sharp\n")
        f.write("   - The magnitude indicates confidence (higher = more confident)\n\n")
        f.write("Results Summary:\n")
        blur_count = sum(blur_flags[selected_method])
        total = len(blur_flags[selected_method])
        f.write(f"  - Sharp frames: {total - blur_count} ({(total - blur_count)/total*100:.1f}%)\n")
        f.write(f"  - Blurry frames: {blur_count} ({blur_count/total*100:.1f}%)\n")
        
    return summary_path, csv_path, report_path

# Helper functions for the executive summary
def create_distribution_plot(ax, blur_scores, thresholds, method, have_seaborn=False):
    """Create distribution plot with thresholds marked"""
    lap_vals = [score['laplacian'] for score in blur_scores]
    ten_vals = [score['tenengrad'] for score in blur_scores]
    fft_vals = [score['fft'] for score in blur_scores]
    
    # Plot distributions with kernel density estimates or histograms
    if have_seaborn:
        import seaborn as sns
        sns.kdeplot(lap_vals, ax=ax, label="Laplacian", color="blue")
        sns.kdeplot(ten_vals, ax=ax, label="Tenengrad", color="orange")
        sns.kdeplot(fft_vals, ax=ax, label="FFT", color="green")
    else:
        # Use histograms if seaborn not available
        ax.hist(lap_vals, bins=20, alpha=0.3, label="Laplacian", color="blue")
        ax.hist(ten_vals, bins=20, alpha=0.3, label="Tenengrad", color="orange")
        ax.hist(fft_vals, bins=20, alpha=0.3, label="FFT", color="green")
    
    # Add threshold lines
    ax.axvline(x=thresholds[method]['laplacian'], color='blue', linestyle='--', 
               label=f"Laplacian Threshold ({thresholds[method]['laplacian']:.2f})")
    ax.axvline(x=thresholds[method]['tenengrad'], color='orange', linestyle='--',
               label=f"Tenengrad Threshold ({thresholds[method]['tenengrad']:.2f})")
    ax.axvline(x=thresholds[method]['fft'], color='green', linestyle='--',
               label=f"FFT Threshold ({thresholds[method]['fft']:.2f})")
    
    ax.set_title("Blur Metric Distributions with Thresholds")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if have_seaborn else "Count")
    ax.legend(fontsize='small')
    
def create_confidence_histogram(ax, confidence_scores):
    """Create histogram of confidence scores"""
    confidences = [score['confidence'] for score in confidence_scores]
    
    # Create histogram
    bins = np.linspace(-100, 100, 21)  # 10-point bins from -100 to 100
    ax.hist(confidences, bins=bins, alpha=0.7)
    
    # Add vertical line at 0 (threshold)
    ax.axvline(x=0, color='red', linestyle='--', label="Classification Threshold")
    
    # Add explanatory text
    ax.text(-90, ax.get_ylim()[1]*0.9, "← More Blurry", fontsize=10)
    ax.text(10, ax.get_ylim()[1]*0.9, "More Sharp →", fontsize=10)
    
    ax.set_title("Distribution of Confidence Scores")
    ax.set_xlabel("Confidence Score (-100 to +100)")
    ax.set_ylabel("Number of Frames")
    ax.legend()
    
def display_confidence_examples(ax, blur_scores, confidence_scores, all_frames):
    """Display example frames at different confidence levels"""
    # Select frames at different confidence levels
    conf_ranges = [
        ("Very Blurry (< -70)", lambda s: s['confidence'] < -70),
        ("Slightly Blurry (-30 to 0)", lambda s: -30 <= s['confidence'] < 0),
        ("Slightly Sharp (0 to 30)", lambda s: 0 <= s['confidence'] < 30),
        ("Very Sharp (> 70)", lambda s: s['confidence'] > 70)
    ]
    
    examples = []
    for label, condition in conf_ranges:
        matching = [s for s in confidence_scores if condition(s) and s['frame'] in all_frames]
        if matching:
            # Take middle example from each range
            examples.append((label, matching[len(matching)//2]))
    
    # Create a grid within the axes
    grid_size = (2, 2)
    
    # Create mini subplots for each example
    for i, (label, conf_score) in enumerate(examples):
        if i >= grid_size[0] * grid_size[1]:
            break
            
        # Calculate grid position
        row, col = i // grid_size[1], i % grid_size[1]
        
        # Get frame
        frame_id = conf_score['frame']
        frame = all_frames[frame_id]
        
        # Get score
        score = next((s for s in blur_scores if s['frame'] == frame_id), None)
        if not score:
            continue
        
        # Create subplot
        subax = ax.inset_axes([col/grid_size[1], 1-row/grid_size[0]-1/grid_size[0], 
                              1/grid_size[1], 1/grid_size[0]])
        
        # Display frame
        subax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        subax.set_title(f"{label}\nConfidence: {conf_score['confidence']}")
        subax.axis('off')
    
    ax.axis('off')
    ax.set_title("Examples at Different Confidence Levels", y=1.05)
    
def create_comparison_visualization(ax, blur_scores, confidence_scores, all_frames, thresholds, method):
    """Create visualization comparing similar frames with different classifications"""
    # Find a pair of frames with different classifications but similar appearance
    # This could be frames close to the threshold
    blurry_near_threshold = [s for s in confidence_scores if s['is_blurry'] and s['confidence'] > -30]
    sharp_near_threshold = [s for s in confidence_scores if not s['is_blurry'] and s['confidence'] < 30]
    
    # Sort by confidence to get closest to threshold
    blurry_near_threshold.sort(key=lambda x: abs(x['confidence']))
    sharp_near_threshold.sort(key=lambda x: abs(x['confidence']))
    
    # Get a pair if possible
    if blurry_near_threshold and sharp_near_threshold:
        blurry_frame = blurry_near_threshold[0]
        sharp_frame = sharp_near_threshold[0]
        
        # Get corresponding blur_scores entry
        blurry_score = next((s for s in blur_scores if s['frame'] == blurry_frame['frame']), None)
        sharp_score = next((s for s in blur_scores if s['frame'] == sharp_frame['frame']), None)
        
        if blurry_score and sharp_score and blurry_frame['frame'] in all_frames and sharp_frame['frame'] in all_frames:
            # Get frames
            blurry_img = all_frames[blurry_frame['frame']]
            sharp_img = all_frames[sharp_frame['frame']]
            
            # Create side-by-side comparison
            subax1 = ax.inset_axes([0, 0.5, 0.5, 0.5])
            subax1.imshow(cv2.cvtColor(blurry_img, cv2.COLOR_BGR2RGB))
            subax1.set_title(f"Blurry: Frame {blurry_frame['frame']}\nConfidence: {blurry_frame['confidence']}")
            subax1.axis('off')
            
            subax2 = ax.inset_axes([0.5, 0.5, 0.5, 0.5])
            subax2.imshow(cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB))
            subax2.set_title(f"Sharp: Frame {sharp_frame['frame']}\nConfidence: {sharp_frame['confidence']}")
            subax2.axis('off')
            
            # Add metric comparison table
            table_data = [
                ["Metric", "Blurry Frame", "Sharp Frame", "Threshold"],
                ["Laplacian", f"{blurry_score['laplacian']:.2f}", f"{sharp_score['laplacian']:.2f}", f"{thresholds[method]['laplacian']:.2f}"],
                ["Tenengrad", f"{blurry_score['tenengrad']:.2f}", f"{sharp_score['tenengrad']:.2f}", f"{thresholds[method]['tenengrad']:.2f}"],
                ["FFT", f"{blurry_score['fft']:.2f}", f"{sharp_score['fft']:.2f}", f"{thresholds[method]['fft']:.2f}"]
            ]
            
            # Create table at the bottom
            table_ax = ax.inset_axes([0.1, 0, 0.8, 0.4])
            table = table_ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Highlight values below threshold
            for i, row in enumerate(table_data[1:], 1):  # Skip header row
                metric = row[0].lower()
                blurry_val = float(row[1])
                sharp_val = float(row[2])
                threshold = thresholds[method][metric]
                
                if blurry_val < threshold:
                    table[(i, 1)].set_facecolor("#ffcccc")  # Light red for values below threshold
                
                if sharp_val < threshold:
                    table[(i, 2)].set_facecolor("#ffcccc")  # Light red for values below threshold
            
            table_ax.axis('off')
            
            # Add explanation text
            ax.text(0.5, 0.45, 
                   "Values highlighted in red fall below threshold and contribute to the blur classification.",
                   ha='center', fontsize=10)
            
            ax.axis('off')
            ax.set_title("Comparison of Borderline Cases", y=1.05)
        else:
            ax.text(0.5, 0.5, "No suitable comparison frames found", ha='center', va='center')
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, "No suitable comparison frames found", ha='center', va='center')
        ax.axis('off')

def analyze_frame_directory(image_paths, identifier, run_dir, resize_factor=1.0, skip=1, bg_remove=False, visualize_mask=False, method='percentile', confidence=False):
    """Analyze a directory of individual image frames for blur."""
    start_time = time.time()
    logger.info(f"--- Starting analysis for frame set: {identifier} ---")
    logger.info(f"Found {len(image_paths)} image frames.")
    logger.info(f"Resize factor: {resize_factor}")
    logger.info(f"Processing every {skip} image(s)")
    logger.info(f"Background removal: {'Enabled' if bg_remove else 'Disabled'}")
    logger.info(f"Threshold method: {method}")
    if bg_remove and visualize_mask:
        logger.info("Foreground mask visualization: Enabled (will save for first valid frame)")
        
    all_scores = []
    processed_files = []
    processed_indices = []
    processing_times = {
        'laplacian': [], 
        'tenengrad': [], 
        'fft': []
    }
    mask_visualization_saved = False
    processed_count = 0
    total_images_to_process = len(range(0, len(image_paths), skip))
    
    for i in range(0, len(image_paths), skip):
        image_path = image_paths[i]
        frame_idx_for_log = i # Use list index for logging/progress
        progress = int(((processed_count + 1) / total_images_to_process) * 100) if total_images_to_process > 0 else 0
        
        if processed_count % max(1, total_images_to_process // 20) == 0:
             elapsed = time.time() - start_time
             estimated_total = (elapsed / max(1, processed_count)) * total_images_to_process if processed_count > 0 else 0
             remaining = max(0, estimated_total - elapsed)
             logger.info(f"Progress: {progress}% ({processed_count+1}/{total_images_to_process}) - ETA: {remaining:.1f}s")
        
        frame_start = time.time()
        logger.debug(f"Processing image {frame_idx_for_log}: {os.path.basename(image_path)}")
        
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"Failed to load image: {image_path}")
                continue

            # Resize if needed
            if resize_factor != 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                frame = cv2.resize(frame, (new_w, new_h))
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # --- Foreground Extraction (if enabled) ---
            mask = None
            if bg_remove:
                mask, edge_density_map, gray_for_mask, edges_for_mask = extract_foreground_mask(frame)
                if mask is None or cv2.countNonZero(mask) == 0:
                    logger.warning(f"Image {frame_idx_for_log}: Foreground mask invalid or empty. Analyzing full image.")
                    mask = None
                else:
                    logger.debug(f"Image {frame_idx_for_log}: Applied foreground mask.")
                    # --- Save Mask Visualization ---
                    if visualize_mask and not mask_visualization_saved:
                        try:
                            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                            axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            axes[0, 0].set_title("Original Image")
                            axes[0, 0].axis('off')
                            axes[0, 1].imshow(edges_for_mask, cmap='gray')
                            axes[0, 1].set_title("Canny Edges")
                            axes[0, 1].axis('off')
                            im = axes[1, 0].imshow(edge_density_map, cmap='hot')
                            axes[1, 0].set_title("Edge Density Map")
                            axes[1, 0].axis('off')
                            fig.colorbar(im, ax=axes[1, 0])
                            axes[1, 1].imshow(mask, cmap='gray')
                            axes[1, 1].set_title("Final Foreground Mask")
                            axes[1, 1].axis('off')
                            plt.tight_layout()
                            # Use identifier in filename
                            vis_save_path = os.path.join(run_dir, f"foreground_mask_visualization_img_{identifier}_{frame_idx_for_log}.png")
                            plt.savefig(vis_save_path)
                            plt.close(fig)
                            logger.info(f"Saved foreground mask visualization to {vis_save_path}")
                            mask_visualization_saved = True
                        except Exception as e:
                            logger.error(f"Error saving mask visualization for image {frame_idx_for_log}: {e}")

            # Calculate blur metrics
            t1 = time.time()
            lap = laplacian_blur(gray, mask=mask)
            t2 = time.time()
            processing_times['laplacian'].append(t2-t1)
            
            t1 = time.time()
            ten = tenengrad_blur(gray, mask=mask)
            t2 = time.time()
            processing_times['tenengrad'].append(t2-t1)
            
            t1 = time.time()
            fft = fft_blur(gray, mask=mask)
            t2 = time.time()
            processing_times['fft'].append(t2-t1)

            all_scores.append({
                'index': i, # Store original index in list
                'filename': os.path.basename(image_path),
                'laplacian': lap,
                'tenengrad': ten,
                'fft': fft
            })
            processed_files.append(os.path.basename(image_path))
            processed_indices.append(i)
            processed_count += 1
            
            if processed_count % 50 == 0:
                logger.debug(f"Image {frame_idx_for_log} processed in {time.time() - frame_start:.4f} seconds")

        except Exception as e:
            logger.error(f"Error processing image {frame_idx_for_log} ({os.path.basename(image_path)}): {e}")

    logger.info(f"Finished processing {processed_count} images.")
    if processing_times['laplacian']:
        logger.info(f"Average processing times per image:")
        logger.info(f"  Laplacian: {np.mean(processing_times['laplacian']):.4f} seconds")
        logger.info(f"  Tenengrad: {np.mean(processing_times['tenengrad']):.4f} seconds")
        logger.info(f"  FFT: {np.mean(processing_times['fft']):.4f} seconds")
        
    if not all_scores:
        logger.error("No images were successfully processed.")
        return None

    # -- Calculate dynamic thresholds based on *all* processed images --
    logger.info("Calculating dynamic thresholds from image statistics")
    lap_vals = np.array([x['laplacian'] for x in all_scores])
    ten_vals = np.array([x['tenengrad'] for x in all_scores])
    fft_vals = np.array([x['fft'] for x in all_scores])
    
    thresholds = {
        'percentile': {
            'laplacian': np.percentile(lap_vals, 25),
            'tenengrad': np.percentile(ten_vals, 25),
            'fft': np.percentile(fft_vals, 25)
        },
        'stddev': {
            'laplacian': max(0, np.mean(lap_vals) - 1.0 * np.std(lap_vals)),
            'tenengrad': max(0, np.mean(ten_vals) - 1.0 * np.std(ten_vals)),
            'fft': max(0, np.mean(fft_vals) - 1.0 * np.std(fft_vals))
        },
        'fixed': {
            'laplacian': LAPLACIAN_THRESH,
            'tenengrad': TENENGRAD_THRESH,
            'fft': FFT_THRESH
        }
    }
    logger.info("Calculated thresholds:")
    for m, values in thresholds.items():
        logger.info(f"  {m.capitalize()} method:")
        for metric, value in values.items():
            logger.info(f"    {metric}: {value:.2f}")
            
    # -- Classify images based on selected method --
    blur_flags = {tm: [] for tm in thresholds.keys()}
    final_results = []
    confidence_results = []
    
    current_thresholds = thresholds[method]
    
    for i, score in enumerate(all_scores):
        lap, ten, fft = score['laplacian'], score['tenengrad'], score['fft']
        
        # Classify using all threshold methods for storage
        flags = {}
        for tm, thresh in thresholds.items():
            sharp = (
                lap > thresh['laplacian'] and
                ten > thresh['tenengrad'] and
                fft > thresh['fft']
            )
            flags[tm] = not sharp # Store True if blurry
            blur_flags[tm].append(not sharp)
            
        # Add classification flags to the score dict
        score['is_blurry'] = flags
        final_results.append(score)
        
        # Calculate confidence if requested (using the selected method)
        if confidence:
            conf, lead_metric = calculate_blur_confidence(score, all_scores, thresholds, method)
            confidence_results.append({
                 'index': score['index'],
                 'filename': score['filename'],
                 'confidence': conf,
                 'leading_metric': lead_metric,
                 'is_blurry': flags[method] # Blurriness according to selected method
            })

    total_time = time.time() - start_time
    logger.info(f"Image analysis for {identifier} completed in {total_time:.2f} seconds")
    
    # Log overall blur stats for the selected method
    if blur_flags[method]:
         blur_count = sum(blur_flags[method])
         blur_percentage = (blur_count / len(blur_flags[method])) * 100
         logger.info(f"Selected method ({method}): Found {blur_count} blurry images out of {len(blur_flags[method])} ({blur_percentage:.1f}%)")
         
    return {
        "results": final_results,
        "confidence": confidence_results if confidence else [],
        "thresholds": thresholds,
        "blur_flags_summary": blur_flags # Summary flags for each method
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description='Video/Image blur detection with multiple methods')
    parser.add_argument('--input', type=str, required=True, help='Path to video file, directory of videos, or directory of image frames')
    parser.add_argument('--input_type', type=str, required=True, choices=['video', 'frames'], help='Specify if input is video(s) or image frames')
    parser.add_argument('--resize', type=float, default=1.0, help='Resize factor for frames/images (e.g., 0.5 for half size)')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame/image')
    parser.add_argument('--save_path', type=str, help='Path to save the results plot (only used for single video input)')
    parser.add_argument('--method', type=str, default='percentile', choices=['percentile', 'stddev', 'fixed'], 
                        help='Threshold method to use for blur detection')
    
    # New arguments for stakeholder visualizations
    parser.add_argument('--stakeholder', action='store_true', help='Generate stakeholder-friendly visualizations')
    parser.add_argument('--confidence', action='store_true', help='Include confidence scores in output')
    
    # Argument for background removal
    parser.add_argument('--bg_remove', action='store_true', help='Enable foreground extraction based on edge density')
    
    # Argument for visualizing the mask generation
    parser.add_argument('--visualize_mask', action='store_true', help='Save a visualization of the foreground mask creation steps (requires --bg_remove)')
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=" * 40)
        logger.info(f"Blur Detection Script started at {timestamp}")
        logger.info("=" * 40)
        
        # --- Input Handling ---
        input_path = args.input
        video_files_to_process = []
        image_files_to_process = []
        processing_directory_videos = False
        processing_frames = False
        
        if not os.path.exists(input_path):
            logger.error(f"Input path not found: {input_path}")
            exit(1)
        
        if args.input_type == 'video':
            valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
            if os.path.isfile(input_path):
                if input_path.lower().endswith(valid_extensions):
                    video_files_to_process.append(input_path)
                    logger.info(f"Processing single video file: {input_path}")
                else:
                    logger.error(f"Input file is not a supported video type: {input_path}")
                    exit(1)
            elif os.path.isdir(input_path):
                processing_directory_videos = True
                logger.info(f"Processing video files in directory: {input_path}")
                for filename in os.listdir(input_path):
                    if filename.lower().endswith(valid_extensions):
                        video_files_to_process.append(os.path.join(input_path, filename))
                
                if not video_files_to_process:
                    logger.error(f"No supported video files found in directory: {input_path}")
                    exit(1)
                logger.info(f"Found {len(video_files_to_process)} video file(s) to process.")
            else:
                logger.error(f"Input path is neither a file nor a directory: {input_path}")
                exit(1)
                
        elif args.input_type == 'frames':
            if not os.path.isdir(input_path):
                 logger.error(f"Input path must be a directory when input_type is 'frames': {input_path}")
                 exit(1)
                 
            processing_frames = True
            logger.info(f"Processing image frames in directory: {input_path}")
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            # Use glob to find files and sort them
            all_files = []
            for ext in valid_extensions:
                all_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
                all_files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}"))) # Handle uppercase extensions
            
            # Sort files naturally (important for sequences like frame_1, frame_10)
            try:
                import natsort
                image_files_to_process = natsort.natsorted(all_files)
                logger.info("Using natsort for natural sorting of filenames.")
            except ImportError:
                logger.warning("natsort package not found. Using simple alphabetical sort. Install with 'pip install natsort' for better frame sequence sorting.")
                image_files_to_process = sorted(all_files)
                
            if not image_files_to_process:
                logger.error(f"No supported image files found in directory: {input_path}")
                exit(1)
            logger.info(f"Found {len(image_files_to_process)} image file(s) to process.")
            
        else:
            # Should not happen due to argparse choices, but good practice
            logger.error(f"Invalid input_type specified: {args.input_type}")
            exit(1)

        # --- Result Setup ---
        results_dir = "blur_results"
        os.makedirs(results_dir, exist_ok=True)
        run_dir = os.path.join(results_dir, f"analysis_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        processing_multiple_frame_dirs = False # Flag for the new mode
        
        aggregated_results = {} # Initialize dictionary for results
        frame_sets_to_process = [] # List to hold tasks for multiple frame directories
        
        # --- Processing ---
        if processing_frames:
            # --- Process Directory of Frames ---
            logger.info("Starting frame directory analysis...")
            frame_analysis_output = analyze_frame_directory(
                 image_paths=image_files_to_process,
                 identifier=os.path.basename(os.path.normpath(input_path)),
                 run_dir=run_dir,
                 resize_factor=args.resize,
                 skip=args.skip,
                 bg_remove=args.bg_remove,
                 visualize_mask=args.visualize_mask,
                 method=args.method,
                 confidence=(args.confidence or args.stakeholder)
            )
            
            if frame_analysis_output:
                # Structure the output JSON for frames
                aggregated_results[os.path.basename(os.path.normpath(input_path))] = {
                    "type": "frames",
                    "settings": {
                        "resize_factor": args.resize,
                        "skip": args.skip,
                        "bg_remove": args.bg_remove,
                        "method": args.method
                    },
                    "thresholds": frame_analysis_output["thresholds"],
                    "results": frame_analysis_output["results"],
                    "confidence": frame_analysis_output["confidence"]
                }
                logger.info("Frame directory analysis complete.")
                # Generate Plots for the single directory
                plot_path = os.path.join(run_dir, f"blur_plot_frames_{os.path.basename(os.path.normpath(input_path))}_{timestamp}.png")
                plot_blur_metrics(
                    blur_scores=frame_analysis_output["results"],
                    thresholds=frame_analysis_output["thresholds"],
                    save_path=plot_path,
                    is_frames_mode=True
                )
            else:
                logger.error("Frame directory analysis failed to produce results.")
                # Store error information for this specific frame set
                aggregated_results[os.path.basename(os.path.normpath(input_path))] = {"error": "Analysis failed to produce results"}
            
        else:
            # --- Process Single Video or Directory of Videos ---
            logger.info("Starting video analysis...")
            for video_path in video_files_to_process:
                logger.info(f"--- Analyzing {os.path.basename(video_path)} ---")
                
                analysis_results = analyze_video(
                    video_path,
                    resize_factor=args.resize,
                    skip_frames=args.skip,
                    bg_remove=args.bg_remove,
                    visualize_mask=args.visualize_mask,
                    run_dir=run_dir
                )
                
                if analysis_results is None or len(analysis_results) != 7:
                     logger.error(f"Analysis failed for {os.path.basename(video_path)}. Skipping.")
                     aggregated_results[os.path.basename(video_path)] = {"error": "Analysis failed"}
                     continue
                
                blur_scores, blur_flags, thresholds, sample_frames, sample_indices, all_frames, method_classifications = analysis_results
                
                video_base_name = os.path.basename(video_path)
                aggregated_results[video_base_name] = {} 
                
                if blur_scores:
                    aggregated_results[video_base_name]['scores'] = blur_scores
                    aggregated_results[video_base_name]['thresholds'] = thresholds
                    aggregated_results[video_base_name]['blur_flags'] = blur_flags
                    
                    confidence_data = None
                    if args.confidence or args.stakeholder:
                        confidence_data = []
                        for i, score in enumerate(blur_scores):
                            confidence, metric = calculate_blur_confidence(score, blur_scores, thresholds, args.method)
                            confidence_data.append({
                                'frame': score['frame'],
                                'confidence': confidence,
                                'leading_metric': metric,
                                'is_blurry': blur_flags[args.method][i]
                            })
                        aggregated_results[video_base_name]['confidence'] = confidence_data
                        
                    # --- Individual File Saving (Only if NOT processing a directory of videos) ---
                    if not processing_directory_videos:
                        logger.info(f"Saving individual results for {video_base_name}")
                        # Save results plot
                        plot_path = args.save_path if args.save_path else os.path.join(run_dir, f"blur_plot_{video_base_name}_{timestamp}.png")
                        plot_blur_metrics(blur_scores, thresholds, save_path=plot_path)
                        
                        # Save confidence data to CSV (if calculated)
                        if confidence_data:
                            confidence_csv = os.path.join(run_dir, f"confidence_scores_{video_base_name}_{timestamp}.csv")
                            try:
                                with open(confidence_csv, 'w') as f:
                                    f.write("frame,laplacian,tenengrad,fft,is_blurry,confidence,leading_metric\n")
                                    for i, score in enumerate(blur_scores):
                                        conf = confidence_data[i]
                                        f.write(f"{score['frame']}, {score['laplacian']:.2f},{score['tenengrad']:.2f},{score['fft']:.2f},"
                                              f"{conf['is_blurry']}, {conf['confidence']}, {conf['leading_metric']}\n")
                                logger.info(f"Confidence scores saved to {confidence_csv}")
                            except Exception as e:
                                 logger.error(f"Error saving confidence CSV for {video_base_name}: {e}")
                    
                        # Generate stakeholder report if requested
                        if args.stakeholder:
                             logger.info(f"Generating stakeholder summary for {video_base_name}")
                             summary_path, csv_path, report_path = generate_executive_summary(
                                blur_scores, blur_flags, thresholds, all_frames, 
                                method_classifications, args.method, save_dir=run_dir
                             )
                             if summary_path:
                                  logger.info(f"Executive summary saved to {summary_path}")
                                  logger.info(f"Confidence scores saved to {csv_path}")
                                  logger.info(f"Summary report saved to {report_path}")
                             else:
                                  logger.error(f"Failed to generate stakeholder summary for {video_base_name}")
                        
                        # Visualize sample frames
                        visualize_sample_frames(sample_frames, sample_indices, blur_scores, thresholds, save_dir=run_dir)
                        
                        # Create method comparison visualization
                        visualize_method_comparison(all_frames, method_classifications, thresholds, save_dir=run_dir)
                        
                        # Save numerical results
                        results_path = os.path.join(run_dir, f"blur_data_{video_base_name}_{timestamp}.csv")
                        try:
                            with open(results_path, 'w') as f:
                                f.write("frame,laplacian,tenengrad,fft,is_blurry_percentile,is_blurry_stddev,is_blurry_fixed\n")
                                for i, score in enumerate(blur_scores):
                                    f.write(f"{score['frame']}, {score['laplacian']:.2f},{score['tenengrad']:.2f},{score['fft']:.2f},"
                                           f"{blur_flags['percentile'][i]},{blur_flags['stddev'][i]},{blur_flags['fixed'][i]}\n")
                            logger.info(f"Results saved to {results_path}")
                        except Exception as e:
                            logger.error(f"Error saving results CSV for {video_base_name}: {e}")
                        
                        # Save threshold information
                        thresh_path = os.path.join(run_dir, f"thresholds_{video_base_name}_{timestamp}.txt")
                        try:
                            with open(thresh_path, 'w') as f:
                                f.write(f"Calculated Thresholds for {video_base_name}:\n")
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
                            logger.error(f"Error saving threshold information for {video_base_name}: {e}")
                            
                else:
                     logger.warning(f"No blur scores generated for {video_base_name}. Skipping result storage for this video.")
                     aggregated_results[video_base_name] = {"error": "No blur scores generated"}
            
            # Assign video results to the final JSON structure if videos were processed
            if not processing_frames:
                 final_json_output = aggregated_results
                 
        # --- Final JSON Output ---
        if aggregated_results:
            json_output_path = os.path.join(run_dir, "aggregated_results.json")
            try:
                with open(json_output_path, 'w') as f:
                    json.dump(aggregated_results, f, indent=4)
                logger.info(f"Aggregated results saved to: {json_output_path}")
            except Exception as e:
                 logger.error(f"Error saving aggregated JSON results: {e}")
        else:
            logger.error("No results were generated to save.")
                 
        logger.info("=" * 40)
        logger.info("Blur Detection Script finished")
        logger.info("=" * 40)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")

