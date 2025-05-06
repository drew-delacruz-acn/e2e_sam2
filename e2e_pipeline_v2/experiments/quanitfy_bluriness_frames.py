import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from datetime import datetime
import argparse
import json
import re # Import re for sorting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('blur_detection_images.log') # Changed log file name
    ]
)
logger = logging.getLogger(__name__)

# Removed CUDA detection code since it's not needed

# --- Core Blur Calculation Functions (Unchanged) ---
def laplacian_blur(gray, mask=None):
    """Calculate blur score using Laplacian variance (higher = sharper)"""
    try:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        if mask is not None and cv2.countNonZero(mask) > 0:
            masked_lap = lap[mask == 255]
            if masked_lap.size == 0:
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
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        h, w = gray.shape
        cx, cy = w//2, h//2
        fft_mask = np.ones((h, w), dtype=np.uint8)
        fft_mask[cy-cutoff_size:cy+cutoff_size, cx-cutoff_size:cx+cutoff_size] = 0
        fshift_filtered = fshift * fft_mask
        ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(ishift)
        magnitude = np.abs(img_back)
        if mask is not None and cv2.countNonZero(mask) > 0:
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

# --- Foreground Extraction (Unchanged) ---
def extract_foreground_mask(frame, edge_threshold=100, density_threshold=0.3):
    """Extract foreground mask based on edge density"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, edge_threshold // 2, edge_threshold)
        kernel = np.ones((5,5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        edge_density = cv2.GaussianBlur(dilated_edges, (21, 21), 0)
        edge_density = edge_density / 255.0
        mask = (edge_density > density_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if mask.shape != gray.shape:
            logger.warning(f"Mask shape {mask.shape} differs from gray shape {gray.shape}. Resizing mask.")
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask, edge_density, gray, edges
    except Exception as e:
        logger.error(f"Error in extract_foreground_mask: {e}")
        return None, None, None, None

# -- PARAMETERS (Unchanged) --
LAPLACIAN_THRESH = 100
TENENGRAD_THRESH = 20
FFT_THRESH = 10

# --- REMOVED get_video_info ---

# --- MODIFIED Analysis Function ---
def analyze_image_sequence(image_paths, sequence_name, resize_factor=1.0, skip_frames=1, bg_remove=False, visualize_mask=False, run_dir="."):
    """Analyzes a sequence of image files for blurriness."""
    start_time = time.time()
    logger.info(f"Starting image sequence analysis for: {sequence_name}")
    logger.info(f"Found {len(image_paths)} images in sequence.")
    logger.info(f"Resize factor: {resize_factor}")
    logger.info(f"Processing every {skip_frames} frame(s)")
    logger.info(f"Background removal based on edge density: {'Enabled' if bg_remove else 'Disabled'}")
    if bg_remove and visualize_mask:
        logger.info("Foreground mask visualization: Enabled (will save for first valid frame)")

    blur_scores = []
    processed_count = 0
    processing_times = {'laplacian': [], 'tenengrad': [], 'fft': []}
    all_frames_data = {} # Store image data for visualization {processed_idx: frame_data}
    frame_info = {} # Store width/height from first frame
    mask_visualization_saved = False

    total_files = len(image_paths)
    last_progress = -1
    progress_update_interval = max(1, (total_files // skip_frames) // 20 if skip_frames > 0 else total_files // 20) # Update progress ~20 times

    processed_idx = 0 # Index of the frame *actually processed* (after skipping)

    for frame_idx, image_path in enumerate(image_paths):

        # Skip frames if requested
        if frame_idx % skip_frames != 0:
            continue

        # Calculate and show progress based on frame_idx
        progress = int((frame_idx / total_files) * 100) if total_files > 0 else 0
        if progress > last_progress and (processed_idx == 0 or processed_idx % progress_update_interval == 0):
            elapsed = time.time() - start_time
            # Estimate ETA based on processed frames
            estimated_total = (elapsed / max(1, processed_idx)) * (total_files / skip_frames) if processed_idx > 0 else 0
            remaining = max(0, estimated_total - elapsed)
            logger.info(f"Progress: {progress}% ({frame_idx+1}/{total_files}) - Frame {processed_idx+1} - ETA: {remaining:.1f}s")
            last_progress = progress

        frame_start = time.time()
        logger.debug(f"Processing frame index {frame_idx} (Processed Index: {processed_idx}, Path: {os.path.basename(image_path)})")

        try:
            frame = cv2.imread(image_path)
            if frame is None:
                logger.warning(f"Could not read frame {frame_idx} from {image_path}. Skipping.")
                continue

            # --- Get info from first frame ---
            if not frame_info:
                 h, w = frame.shape[:2]
                 frame_info = {"width": w, "height": h, "total_frames": total_files} # Store total frames here
                 logger.info(f"Frame dimensions: {w}x{h}")
                 if w > 1920 and resize_factor == 1.0:
                     logger.warning("High resolution frames detected. Consider using --resize < 1.0 for faster processing")

            # Store frames for method comparison visualization (using processed_idx)
            # Store a regular sampling of frames (about 20-30 total for performance)
            total_processed_estimate = total_files // skip_frames
            sample_interval = max(1, total_processed_estimate // 25)
            if processed_idx % sample_interval == 0:
                 all_frames_data[processed_idx] = frame.copy() # Use processed_idx as key

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
                    logger.warning(f"Frame index {frame_idx}: Foreground mask invalid or empty. Analyzing full frame.")
                    mask = None # Ensure mask is None if invalid
                else:
                    logger.debug(f"Frame index {frame_idx}: Applied foreground mask.")
                    # --- Save Mask Visualization (if enabled and not already saved) ---
                    if visualize_mask and not mask_visualization_saved:
                         try:
                             fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                             axes[0, 0].imshow(cv2.cvtColor(all_frames_data.get(processed_idx, frame), cv2.COLOR_BGR2RGB)) # Use original if available
                             axes[0, 0].set_title("Original Frame")
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
                             vis_save_path = os.path.join(run_dir, f"foreground_mask_visualization_frame_{frame_idx}.png")
                             plt.savefig(vis_save_path)
                             plt.close(fig)
                             logger.info(f"Saved foreground mask visualization to {vis_save_path}")
                             mask_visualization_saved = True # Set flag so we don't save again
                         except Exception as e:
                             logger.error(f"Error saving mask visualization for frame index {frame_idx}: {e}")

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
                'frame': processed_idx, # Use processed index as the primary identifier
                'original_frame_index': frame_idx, # Keep original index for reference
                'path': os.path.basename(image_path), # Store filename for reference
                'laplacian': lap,
                'tenengrad': ten,
                'fft': fft
            })

            processed_count += 1

            # Log detailed frame timing occasionally
            if processed_count % 10 == 0:
                logger.debug(f"Frame index {frame_idx} (Processed: {processed_idx}) processed in {time.time() - frame_start:.4f} seconds")

        except Exception as e:
            logger.error(f"Error processing frame index {frame_idx} ({image_path}): {e}")

        processed_idx += 1 # Increment index for processed frames

    # --- Post-processing ---
    logger.info(f"Finished processing images for sequence: {sequence_name}")
    logger.info(f"Processed {processed_count} frames out of {total_files} total images (skipped {total_files - processed_count})")
    if processing_times['laplacian']:
        logger.info(f"Average processing times per processed frame:")
        logger.info(f"  Laplacian: {np.mean(processing_times['laplacian']):.4f} seconds")
        logger.info(f"  Tenengrad: {np.mean(processing_times['tenengrad']):.4f} seconds")
        logger.info(f"  FFT: {np.mean(processing_times['fft']):.4f} seconds")

    if not blur_scores:
        logger.warning(f"No blur scores generated for sequence {sequence_name}. Cannot calculate thresholds or flags.")
        return None, None, None, None, None # Return None for results

    # -- Calculate dynamic thresholds --
    logger.info("Calculating dynamic thresholds from statistics")
    try:
        lap_vals = np.array([x['laplacian'] for x in blur_scores])
        ten_vals = np.array([x['tenengrad'] for x in blur_scores])
        fft_vals = np.array([x['fft'] for x in blur_scores])

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
        for method, values in thresholds.items():
            logger.info(f"  {method.capitalize()} method:")
            for metric, value in values.items():
                logger.info(f"    {metric}: {value:.2f}")
    except Exception as e:
        logger.error(f"Error calculating thresholds: {e}")
        return blur_scores, None, None, all_frames_data, None # Return scores but no thresholds/flags

    # -- TEMPORAL ANALYSIS (now based on processed index) --
    logger.info("Starting temporal analysis with dynamic thresholds")
    blur_flags = { 'percentile': [], 'stddev': [], 'fixed': [] }
    method_classifications = {
        'percentile': {'sharp': [], 'blurry': []},
        'stddev': {'sharp': [], 'blurry': []},
        'fixed': {'sharp': [], 'blurry': []}
    }

    try:
        for i, scores in enumerate(blur_scores): # i corresponds to processed_idx
            lap, ten, fft = scores['laplacian'], scores['tenengrad'], scores['fft']
            frame_id = scores['frame'] # This is the processed_idx

            # Check each threshold method
            for method, thresh in thresholds.items():
                sharp = (lap > thresh['laplacian'] and ten > thresh['tenengrad'] and fft > thresh['fft'])

                # Compare to neighbor average (temporal check) - uses 'i' which is processed index
                if 1 <= i < len(blur_scores) - 1:
                    prev = blur_scores[i-1]['laplacian']
                    nxt = blur_scores[i+1]['laplacian']
                    # Use a relative threshold to avoid issues with very low scores
                    if lap < 0.5 * ((prev + nxt) / 2) and abs(lap - ((prev + nxt) / 2)) > (thresh['laplacian'] * 0.1): # Check if drop is significant
                         sharp = False
                         if method == 'percentile': # Only log for the default method
                             logger.debug(f"Frame {frame_id} (Original index: {scores['original_frame_index']}) failed temporal check")

                blur_flags[method].append(not sharp)

                # Track frame classification for later visualization (using frame_id = processed_idx)
                if frame_id in all_frames_data:
                    target_bucket = 'sharp' if sharp else 'blurry'
                    method_classifications[method][target_bucket].append(frame_id)

            # Log detailed analysis for some frames with the default method
            if processed_count < 20 or i % (processed_count // 10 if processed_count >= 10 else 1) == 0 or not sharp:
                logger.info(f"Processed Frame {scores['frame']:04d} (Orig Idx: {scores['original_frame_index']}): Sharp={sharp} | Lap={lap:.2f} | Ten={ten:.2f} | FFT={fft:.2f}")

    except Exception as e:
        logger.error(f"Error during temporal analysis: {e}")

    total_time = time.time() - start_time
    logger.info(f"Sequence analysis for {sequence_name} completed in {total_time:.2f} seconds")
    if processed_count > 0:
        logger.info(f"Average time per processed frame: {total_time/processed_count:.4f} seconds")

    # Calculate blur statistics for each method
    for method, flags in blur_flags.items():
        if flags:
            blur_count = sum(flags)
            blur_percentage = (blur_count / len(flags)) * 100
            logger.info(f"{method.capitalize()} method: Found {blur_count} blurry frames out of {len(flags)} processed ({blur_percentage:.1f}%)")

    # Return modified data structures
    return blur_scores, blur_flags, thresholds, all_frames_data, method_classifications


# --- Plotting and Visualization Functions (Mostly adjusted for frame identifiers and inputs) ---

def plot_blur_metrics(blur_scores, thresholds, sequence_name, save_path=None):
    """Plots blur metrics over processed frame indices."""
    if not blur_scores:
        logger.error(f"No blur scores to plot for sequence {sequence_name}")
        return
    if thresholds is None:
        logger.error(f"No thresholds calculated for sequence {sequence_name}, cannot plot.")
        # Optionally plot just the scores without thresholds
        # return

    logger.info(f"Generating blur metrics plot for {sequence_name}")
    try:
        # Use processed_idx ('frame') as the x-axis identifier
        frames = [x['frame'] for x in blur_scores]
        lap_vals = [x['laplacian'] for x in blur_scores]
        ten_vals = [x['tenengrad'] for x in blur_scores]
        fft_vals = [x['fft'] for x in blur_scores]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f"Blur Metrics for Sequence: {sequence_name}", fontsize=14)

        # Plot 1: Laplacian
        ax1.plot(frames, lap_vals, label='Laplacian Variance')
        if thresholds:
            for method, thresh in thresholds.items():
                ax1.axhline(y=thresh['laplacian'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'),
                           linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold ({thresh['laplacian']:.1f})")
        ax1.set_title("Laplacian Variance (Edge Sharpness)")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Tenengrad
        ax2.plot(frames, ten_vals, label='Tenengrad', color='orange')
        if thresholds:
            for method, thresh in thresholds.items():
                ax2.axhline(y=thresh['tenengrad'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'),
                           linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold ({thresh['tenengrad']:.1f})")
        ax2.set_title("Tenengrad (Gradient Magnitude)")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: FFT
        ax3.plot(frames, fft_vals, label='FFT Energy', color='green')
        if thresholds:
            for method, thresh in thresholds.items():
                ax3.axhline(y=thresh['fft'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'),
                           linestyle='--', alpha=0.7, label=f"{method.capitalize()} Threshold ({thresh['fft']:.1f})")
        ax3.set_title("FFT High-Frequency Energy")
        ax3.set_xlabel("Processed Frame Index") # Changed label
        ax3.set_ylabel("Value")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for suptitle

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot for {sequence_name} saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

        # Generate histograms (if thresholds exist)
        if thresholds:
            fig_hist, (axh1, axh2, axh3) = plt.subplots(1, 3, figsize=(15, 5))
            fig_hist.suptitle(f"Metric Distributions for Sequence: {sequence_name}", fontsize=14)

            axh1.hist(lap_vals, bins=30, alpha=0.7)
            for method, thresh in thresholds.items():
                axh1.axvline(x=thresh['laplacian'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'),
                           linestyle='--', label=f"{method.capitalize()}")
            axh1.set_title("Laplacian Distribution")
            axh1.set_xlabel("Value")
            axh1.set_ylabel("Frequency")
            axh1.legend()

            axh2.hist(ten_vals, bins=30, alpha=0.7, color='orange')
            for method, thresh in thresholds.items():
                axh2.axvline(x=thresh['tenengrad'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'),
                          linestyle='--', label=f"{method.capitalize()}")
            axh2.set_title("Tenengrad Distribution")
            axh2.set_xlabel("Value")
            axh2.legend()

            axh3.hist(fft_vals, bins=30, alpha=0.7, color='green')
            for method, thresh in thresholds.items():
                axh3.axvline(x=thresh['fft'], color=('r' if method=='fixed' else 'g' if method=='percentile' else 'b'),
                           linestyle='--', label=f"{method.capitalize()}")
            axh3.set_title("FFT Distribution")
            axh3.set_xlabel("Value")
            axh3.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

            if save_path:
                histogram_path = save_path.replace('.png', '_histograms.png')
                plt.savefig(histogram_path)
                logger.info(f"Histograms for {sequence_name} saved to {histogram_path}")
                plt.close(fig_hist)
            else:
                plt.show()

    except Exception as e:
        logger.error(f"Error generating plots for {sequence_name}: {e}")


# --- Sample frame visualization needs adjustment ---
# visualize_sample_frames needs access to the actual image data (all_frames_data)
# or needs to reload images based on paths stored in blur_scores.
# Let's simplify this for now to use all_frames_data if available.

def visualize_sample_frames(all_frames_data, blur_scores, thresholds, sequence_name, save_dir=None):
    """Visualize sample frames with their blur metrics (uses pre-loaded frames)"""
    if not all_frames_data or not blur_scores or not thresholds:
        logger.warning(f"Insufficient data to visualize sample frames for {sequence_name}")
        return

    # Identify frames present in both all_frames_data and blur_scores
    available_processed_indices = list(all_frames_data.keys())
    scores_for_available_frames = [s for s in blur_scores if s['frame'] in available_processed_indices]

    if not scores_for_available_frames:
        logger.warning(f"No matching scores found for available frames in {sequence_name}")
        return

    logger.info(f"Visualizing {len(scores_for_available_frames)} sample frames for {sequence_name}")

    try:
        # Sort samples by Laplacian score (ascending)
        sorted_scores = sorted(scores_for_available_frames, key=lambda score: score['laplacian'])

        # Create a figure with sample frames (limit number for clarity)
        n_samples = min(len(sorted_scores), 10) # Show max 10 samples
        if n_samples == 0: return

        fig, axes = plt.subplots(n_samples, 1, figsize=(10, n_samples * 4))
        if n_samples == 1: axes = [axes] # Make it iterable

        fig.suptitle(f"Sample Frames for Sequence: {sequence_name}", fontsize=14)

        indices_to_plot = np.linspace(0, len(sorted_scores)-1, n_samples, dtype=int) # Evenly spaced by blurriness

        for i, score_idx in enumerate(indices_to_plot):
            score = sorted_scores[score_idx]
            frame_processed_idx = score['frame']
            frame_original_idx = score['original_frame_index']
            frame_data = all_frames_data[frame_processed_idx]

            # Determine sharpness status based on percentile method (default)
            sharp = (
                score['laplacian'] > thresholds['percentile']['laplacian'] and
                score['tenengrad'] > thresholds['percentile']['tenengrad'] and
                score['fft'] > thresholds['percentile']['fft']
            )

            # Display the frame
            axes[i].imshow(cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Processed Idx: {frame_processed_idx} (Orig: {frame_original_idx}): {'Sharp' if sharp else 'Blurry'}\n"
                             f"Lap: {score['laplacian']:.1f} / Ten: {score['tenengrad']:.1f} / FFT: {score['fft']:.1f} "
                             f"(Thr Pct: L{thresholds['percentile']['laplacian']:.1f} T{thresholds['percentile']['tenengrad']:.1f} F{thresholds['percentile']['fft']:.1f})")
            axes[i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        if save_dir:
            samples_path = os.path.join(save_dir, f'sample_frames_{sequence_name}.png')
            plt.savefig(samples_path)
            logger.info(f"Sample frames plot saved to {samples_path}")
            plt.close(fig)

            # Save individual frames with scores
            individual_frames_dir = os.path.join(save_dir, f"sample_frames_{sequence_name}")
            os.makedirs(individual_frames_dir, exist_ok=True)

            for i, score_idx in enumerate(indices_to_plot):
                score = sorted_scores[score_idx]
                frame_processed_idx = score['frame']
                frame_original_idx = score['original_frame_index']
                frame_data = all_frames_data[frame_processed_idx]

                # Add text to frame with scores
                text_frame = frame_data.copy()
                h, w = text_frame.shape[:2]
                text_size = max(h / 1080, 0.5) # Scale text
                line_height = int(40 * text_size)
                start_y = line_height

                # Create overlay
                overlay = text_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, int(h/4)), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, text_frame, 0.4, 0, text_frame)

                # Add metrics text
                sharp = (
                    score['laplacian'] > thresholds['percentile']['laplacian'] and
                    score['tenengrad'] > thresholds['percentile']['tenengrad'] and
                    score['fft'] > thresholds['percentile']['fft']
                )
                status = f"{'SHARP' if sharp else 'BLURRY'}"
                cv2.putText(text_frame, f"ProcIdx:{frame_processed_idx} (Orig:{frame_original_idx}): {status}",
                           (10, start_y), cv2.FONT_HERSHEY_SIMPLEX,
                           text_size * 0.9, (0, 255, 255) if sharp else (0, 0, 255), 2)
                start_y += line_height
                cv2.putText(text_frame,
                           f"Lap:{score['laplacian']:.1f} Ten:{score['tenengrad']:.1f} FFT:{score['fft']:.1f}",
                           (10, start_y), cv2.FONT_HERSHEY_SIMPLEX,
                           text_size*0.8, (255, 255, 255), 2)
                start_y += line_height
                cv2.putText(text_frame,
                           f"Thr(Pct): L{thresholds['percentile']['laplacian']:.1f} T{thresholds['percentile']['tenengrad']:.1f} F{thresholds['percentile']['fft']:.1f}",
                           (10, start_y), cv2.FONT_HERSHEY_SIMPLEX,
                           text_size*0.7, (200, 200, 200), 1)


                frame_path = os.path.join(individual_frames_dir, f"frame_{frame_processed_idx:04d}_{status.lower()}.jpg")
                cv2.imwrite(frame_path, text_frame)
            logger.info(f"Saved individual sample frames to {individual_frames_dir}")

        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error visualizing sample frames for {sequence_name}: {e}")

# --- Method Comparison Visualization (Adjusted similarly) ---
def visualize_method_comparison(all_frames_data, method_classifications, blur_scores, thresholds, sequence_name, save_dir=None):
    """Create visualization showing how different methods classify the same frames"""
    if not all_frames_data or not method_classifications or not blur_scores or not thresholds:
        logger.warning(f"Insufficient data for method comparison for {sequence_name}")
        return

    available_processed_indices = list(all_frames_data.keys())
    if not available_processed_indices:
        logger.warning(f"No frames available for method comparison visualization for {sequence_name}")
        return

    logger.info(f"Creating method comparison visualization with {len(available_processed_indices)} frames for {sequence_name}")

    try:
        methods = list(method_classifications.keys())

        # Create directories if saving
        comparison_base_dir = None
        if save_dir:
            comparison_base_dir = os.path.join(save_dir, f"method_comparison_{sequence_name}")
            os.makedirs(comparison_base_dir, exist_ok=True)
            for method in methods:
                 method_dir = os.path.join(comparison_base_dir, f"method_{method}")
                 os.makedirs(os.path.join(method_dir, "sharp"), exist_ok=True)
                 os.makedirs(os.path.join(method_dir, "blurry"), exist_ok=True)
            comparison_dir = os.path.join(comparison_base_dir, "comparison_grid")
            os.makedirs(comparison_dir, exist_ok=True)

        # Generate visualizations for each method (saving individual frames)
        if save_dir:
            for method in methods:
                classifications = method_classifications[method]
                method_dir = os.path.join(comparison_base_dir, f"method_{method}")

                for category in ['sharp', 'blurry']:
                    for frame_id in classifications[category]: # frame_id is processed_idx
                        if frame_id in all_frames_data:
                             frame = all_frames_data[frame_id].copy()
                             h, w = frame.shape[:2]
                             text_size = max(h / 1080, 0.5)
                             color = (0, 255, 0) if category == 'sharp' else (0, 0, 255)
                             cv2.putText(frame, f"{method.capitalize()}: {category.upper()}",
                                        (10, int(30*text_size)), cv2.FONT_HERSHEY_SIMPLEX,
                                        text_size, color, 2)
                             cv2.imwrite(os.path.join(method_dir, category, f"frame_{frame_id:04d}.jpg"), frame)

        # Create a multi-method comparison grid for selected frames
        common_frame_ids = sorted(available_processed_indices)

        # Sample frames if too many (focus on disagreement)
        if len(common_frame_ids) > 12:
             # ... (sampling logic similar to original, using common_frame_ids) ...
             # Find disagreement:
             disagreement_frames = []
             for frame_id in common_frame_ids:
                 opinions = []
                 for method in methods:
                     if frame_id in method_classifications[method]['sharp']:
                         opinions.append('sharp')
                     elif frame_id in method_classifications[method]['blurry']:
                         opinions.append('blurry')
                 if len(set(opinions)) > 1: # If there's disagreement
                    disagreement_frames.append(frame_id)

             if disagreement_frames:
                 if len(disagreement_frames) > 12:
                     common_frame_ids = sorted(np.random.choice(disagreement_frames, 12, replace=False))
                 else:
                     common_frame_ids = sorted(disagreement_frames)
             else:
                 # Otherwise, take evenly spaced samples
                 indices = np.linspace(0, len(common_frame_ids)-1, 12, dtype=int)
                 common_frame_ids = [common_frame_ids[i] for i in indices]


        n_frames = len(common_frame_ids)
        n_methods = len(methods)
        if n_frames == 0:
            logger.warning(f"No frames selected for comparison grid for {sequence_name}")
            return

        fig, axes = plt.subplots(n_frames, n_methods, figsize=(n_methods*5, n_frames*4))
        if n_frames == 1: axes = axes.reshape(1, -1)
        if n_methods == 1: axes = axes.reshape(-1, 1)

        fig.suptitle(f"Method Comparison for Sequence: {sequence_name}", fontsize=16)
        disagreements = {m: 0 for m in methods}

        # Populate grid
        for i, frame_id in enumerate(common_frame_ids): # frame_id is processed_idx
            frame_data = all_frames_data[frame_id]
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

            # Get scores for this frame
            frame_scores = next((s for s in blur_scores if s['frame'] == frame_id), None)

            # Find classifications
            frame_classifications = {}
            for method in methods:
                 if frame_id in method_classifications[method]['sharp']:
                     frame_classifications[method] = 'sharp'
                 elif frame_id in method_classifications[method]['blurry']:
                     frame_classifications[method] = 'blurry'
                 else: # Should not happen if blur_flags were calculated correctly
                     frame_classifications[method] = 'unknown'


            # Count disagreements (if more than one opinion exists)
            opinions = list(frame_classifications.values())
            if len(set(opinions)) > 1:
                for method in methods:
                     # Disagrees if its opinion is not the mode (most common)
                     mode = max(set(opinions), key=opinions.count)
                     if frame_classifications.get(method) != mode:
                         disagreements[method] += 1

            # Display each method's classification
            for j, method in enumerate(methods):
                ax = axes[i, j]
                ax.imshow(rgb_frame)

                classification = frame_classifications.get(method, 'unknown')
                is_sharp = (classification == 'sharp')
                title_color = 'green' if is_sharp else ('red' if classification == 'blurry' else 'gray')

                if frame_scores:
                    lap, ten, fft = frame_scores['laplacian'], frame_scores['tenengrad'], frame_scores['fft']
                    thresh = thresholds[method]
                    title = f"{method.capitalize()}: {classification.upper()}\n"
                    title += f"L:{lap:.1f}(>{thresh['laplacian']:.1f}) "
                    title += f"T:{ten:.1f}(>{thresh['tenengrad']:.1f}) "
                    title += f"F:{fft:.1f}(>{thresh['fft']:.1f})"
                else:
                    title = f"{method.capitalize()}: {classification.upper()}"

                ax.set_title(title, color=title_color, fontsize=9)
                ax.axis('off')
                for spine in ax.spines.values(): # Border
                    spine.set_visible(True); spine.set_color(title_color); spine.set_linewidth(3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        # Save comparison grid
        if save_dir:
            comparison_path = os.path.join(comparison_dir, "method_comparison_grid.png")
            plt.savefig(comparison_path)
            logger.info(f"Method comparison grid saved to {comparison_path}")
            plt.close(fig)

            # Save disagreement info
            report_path = os.path.join(comparison_base_dir, "method_analysis.txt")
            with open(report_path, 'w') as f:
                f.write(f"Method Disagreement Analysis for Sequence: {sequence_name}\n")
                f.write(f"Based on {n_frames} sampled frames:\n\n")
                for method, count in disagreements.items():
                     percentage = (count / n_frames) * 100 if n_frames > 0 else 0
                     f.write(f"{method.capitalize()}: {count}/{n_frames} frames disagreed with majority ({percentage:.1f}%)\n")
                # Add threshold info
                # ... (threshold writing logic from original) ...

    except Exception as e:
        logger.error(f"Error creating method comparison visualization for {sequence_name}: {e}")

# --- Confidence Calculation (Adjusted input score keys) ---
def calculate_blur_confidence(score, all_scores, thresholds, method='percentile'):
    """
    Calculate confidence score for blur classification (-100 to +100)
    Uses 'laplacian', 'tenengrad', 'fft' keys from the score dictionary.
    """
    if not thresholds or method not in thresholds:
        logger.warning(f"Cannot calculate confidence, missing thresholds for method '{method}'")
        return 0, 'unknown'
    # ... (rest of the function is likely okay, assuming all_scores format is consistent) ...
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
    confidence = int(np.clip(100 * overall_conf, -100, 100)) # Clip just in case

    # Determine most influential metric
    influences = [abs(lap_conf * weights[0]), abs(ten_conf * weights[1]), abs(fft_conf * weights[2])]
    most_influential = ['laplacian', 'tenengrad', 'fft'][np.argmax(influences)]

    return confidence, most_influential


# --- Normalize Range (Unchanged) ---
def normalize_to_range(value, distribution):
    """Normalize a value to -1 to +1 range based on distribution"""
    if len(distribution) < 2: return 0 # Cannot calculate percentiles
    p10, p90 = np.percentile(distribution[~np.isnan(distribution)], [10, 90]) # Ignore NaNs
    # Handle cases where distribution might be constant
    if p10 == p90:
        range_dist = np.max(distribution) - np.min(distribution)
        if range_dist == 0: return 0 # Truly constant
        range_half = range_dist / 2.0
    else:
        range_half = max(abs(p10), abs(p90), 1e-6) # Avoid division by zero

    # Check for NaN/Inf in value
    if not np.isfinite(value): return 0

    return np.clip(value / range_half, -1, 1)


# --- Executive Summary (Needs adaptation for image sequences) ---
def generate_executive_summary(blur_scores, blur_flags, thresholds, all_frames_data, method_classifications, sequence_name, selected_method='percentile', save_dir=None):
    """Create comprehensive executive summary for an image sequence"""
    logger.info(f"Generating executive summary for sequence: {sequence_name}")

    if not save_dir: logger.error("Save directory required for executive summary"); return None, None, None
    if not blur_scores or not blur_flags or not thresholds or not all_frames_data or not method_classifications:
         logger.error(f"Insufficient data to generate summary for {sequence_name}"); return None, None, None

    # Try seaborn import
    try: import seaborn as sns; have_seaborn = True
    except ImportError: logger.warning("Seaborn not installed. Install with 'pip install seaborn' for better visualizations."); have_seaborn = False

    summary_dir = os.path.join(save_dir, f"summary_{sequence_name}")
    os.makedirs(summary_dir, exist_ok=True)

    # Calculate confidence scores
    confidence_scores = []
    if selected_method in blur_flags:
         for i, score in enumerate(blur_scores):
             conf, metric = calculate_blur_confidence(score, blur_scores, thresholds, selected_method)
             confidence_scores.append({
                 'frame': score['frame'], # Processed index
                 'confidence': conf,
                 'leading_metric': metric,
                 'is_blurry': blur_flags[selected_method][i]
             })
    else:
         logger.warning(f"Selected method '{selected_method}' not found in blur_flags. Cannot calculate confidence.")
         # Optionally create confidence_scores with default values or skip confidence parts


    # Create main summary figure
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)

    # 1. Distribution plot
    ax1 = fig.add_subplot(gs[0, 0])
    # Ensure helper function can handle potentially missing thresholds/seaborn
    create_distribution_plot(ax1, blur_scores, thresholds, selected_method, have_seaborn)

    # 2. Confidence distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if confidence_scores: # Only plot if confidence was calculated
        create_confidence_histogram(ax2, confidence_scores)
    else:
        ax2.text(0.5, 0.5, "Confidence scores not available.", ha='center', va='center'); ax2.axis('off')


    # 3. Sample frames at different confidence levels
    ax3 = fig.add_subplot(gs[1, 0])
    if confidence_scores:
         # Pass all_frames_data instead of all_frames
        display_confidence_examples(ax3, blur_scores, confidence_scores, all_frames_data)
    else:
         ax3.text(0.5, 0.5, "Confidence scores not available for examples.", ha='center', va='center'); ax3.axis('off')


    # 4. Comparison pairs
    ax4 = fig.add_subplot(gs[1, 1])
    if confidence_scores:
        create_comparison_visualization(ax4, blur_scores, confidence_scores, all_frames_data, thresholds, selected_method)
    else:
         ax4.text(0.5, 0.5, "Confidence scores not available for comparison.", ha='center', va='center'); ax4.axis('off')


    fig.suptitle(f"Blur Detection Summary: {sequence_name} ({selected_method.capitalize()} Method)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    summary_path = os.path.join(summary_dir, "executive_summary.png")
    plt.savefig(summary_path)
    logger.info(f"Executive summary plot saved to {summary_path}")
    plt.close(fig)

    # Create confidence-scored CSV
    csv_path = None
    if confidence_scores:
        csv_path = os.path.join(summary_dir, "blur_scores_with_confidence.csv")
        with open(csv_path, 'w') as f:
            f.write("processed_index,original_index,filename,laplacian,tenengrad,fft,is_blurry,confidence,leading_metric\n")
            for i, score in enumerate(blur_scores):
                conf_data = confidence_scores[i]
                f.write(f"{score['frame']},{score['original_frame_index']},{score['path']},"
                       f"{score['laplacian']:.2f},{score['tenengrad']:.2f},{score['fft']:.2f},"
                       f"{conf_data['is_blurry']},{conf_data['confidence']},{conf_data['leading_metric']}\n")
        logger.info(f"Confidence scores CSV saved to {csv_path}")


    # Generate text report
    report_path = os.path.join(summary_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"BLUR DETECTION METHODOLOGY SUMMARY: {sequence_name}\n")
        f.write("==========================================\n\n")
        f.write(f"Method Used: {selected_method.capitalize()}\n\n")
        f.write("How Blur is Detected:\n")
        # ... (Explanation similar to original) ...
        f.write("\nThresholds Calculated:\n")
        if selected_method in thresholds:
             for metric, value in thresholds[selected_method].items():
                 f.write(f"   - {metric.capitalize()}: {value:.2f}\n")
        else:
             f.write("   - Thresholds not available for selected method.\n")
        # ... (Rest of explanation) ...
        f.write("\nResults Summary:\n")
        if selected_method in blur_flags and blur_flags[selected_method]:
             blur_count = sum(blur_flags[selected_method])
             total = len(blur_flags[selected_method])
             f.write(f"  - Sharp frames (processed): {total - blur_count} ({ (total - blur_count)/total*100:.1f }%)\n")
             f.write(f"  - Blurry frames (processed): {blur_count} ({ blur_count/total*100:.1f }%)\n")
        else:
             f.write("  - Blur classification results not available.\n")

    logger.info(f"Summary report saved to {report_path}")
    return summary_path, csv_path, report_path


# --- Helper functions for summary (Need adaptation for frame data) ---
# create_distribution_plot - Mostly okay, ensure it handles empty/None thresholds
# create_confidence_histogram - Okay
# display_confidence_examples - Needs to use all_frames_data
# create_comparison_visualization - Needs to use all_frames_data

def create_distribution_plot(ax, blur_scores, thresholds, method, have_seaborn=False):
    """Create distribution plot with thresholds marked"""
    if not blur_scores: ax.text(0.5, 0.5, "No scores.", ha='center'); return
    lap_vals = [score['laplacian'] for score in blur_scores]
    ten_vals = [score['tenengrad'] for score in blur_scores]
    fft_vals = [score['fft'] for score in blur_scores]

    if have_seaborn:
        import seaborn as sns
        sns.kdeplot(lap_vals, ax=ax, label="Laplacian", color="blue", warn_singular=False)
        sns.kdeplot(ten_vals, ax=ax, label="Tenengrad", color="orange", warn_singular=False)
        sns.kdeplot(fft_vals, ax=ax, label="FFT", color="green", warn_singular=False)
    else:
        ax.hist(lap_vals, bins=20, alpha=0.3, label="Laplacian", color="blue")
        ax.hist(ten_vals, bins=20, alpha=0.3, label="Tenengrad", color="orange")
        ax.hist(fft_vals, bins=20, alpha=0.3, label="FFT", color="green")

    # Add threshold lines only if thresholds exist for the method
    if thresholds and method in thresholds:
        ax.axvline(x=thresholds[method]['laplacian'], color='blue', linestyle='--',
                   label=f"Lap Thresh ({thresholds[method]['laplacian']:.1f})")
        ax.axvline(x=thresholds[method]['tenengrad'], color='orange', linestyle='--',
                   label=f"Ten Thresh ({thresholds[method]['tenengrad']:.1f})")
        ax.axvline(x=thresholds[method]['fft'], color='green', linestyle='--',
                   label=f"FFT Thresh ({thresholds[method]['fft']:.1f})")

    ax.set_title("Blur Metric Distributions & Thresholds")
    ax.set_xlabel("Value"); ax.set_ylabel("Density" if have_seaborn else "Count")
    ax.legend(fontsize='small')

def create_confidence_histogram(ax, confidence_scores):
    """Create histogram of confidence scores"""
    if not confidence_scores: ax.text(0.5, 0.5, "No confidence scores.", ha='center'); return
    confidences = [score['confidence'] for score in confidence_scores]
    bins = np.linspace(-100, 100, 21)
    n, _, patches = ax.hist(confidences, bins=bins, alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label="Classification Threshold")
    max_y = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
    ax.text(-95, max_y*0.9, "← More Blurry", fontsize=9, ha='left')
    ax.text(95, max_y*0.9, "More Sharp →", fontsize=9, ha='right')
    ax.set_title("Distribution of Confidence Scores"); ax.set_xlabel("Confidence Score (-100 to +100)")
    ax.set_ylabel("Number of Frames"); ax.legend()

def display_confidence_examples(ax, blur_scores, confidence_scores, all_frames_data):
    """Display example frames at different confidence levels (uses all_frames_data)"""
    if not confidence_scores or not all_frames_data: ax.text(0.5, 0.5, "No data for examples.", ha='center'); ax.axis('off'); return

    conf_ranges = [
        ("Very Blurry (<-70)", lambda s: s['confidence'] < -70),
        ("Slightly Blurry (-30..0)", lambda s: -30 <= s['confidence'] < 0),
        ("Slightly Sharp (0..30)", lambda s: 0 <= s['confidence'] < 30),
        ("Very Sharp (>70)", lambda s: s['confidence'] > 70)
    ]
    examples = []
    # Find frames available in both confidence_scores and all_frames_data
    available_indices = set(all_frames_data.keys())
    for label, condition in conf_ranges:
        matching = [s for s in confidence_scores if condition(s) and s['frame'] in available_indices]
        if matching:
            # Take middle example if possible
            examples.append((label, matching[len(matching)//2]))

    if not examples: ax.text(0.5, 0.5, "No frames found in confidence ranges.", ha='center'); ax.axis('off'); return

    # Create a grid (adjust if fewer than 4 examples found)
    grid_size = (2, 2)
    ax.axis('off')
    ax.set_title("Examples at Different Confidence Levels", y=1.0)

    for i, (label, conf_score) in enumerate(examples):
         if i >= grid_size[0] * grid_size[1]: break
         row, col = i // grid_size[1], i % grid_size[1]
         frame_id = conf_score['frame'] # Processed index
         frame_data = all_frames_data[frame_id]
         subax = ax.inset_axes([col/grid_size[1], 1-row/grid_size[0]-1/grid_size[0],
                                1/grid_size[1], 1/grid_size[0]])
         subax.imshow(cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB))
         subax.set_title(f"{label}\nConf: {conf_score['confidence']} (Idx:{frame_id})", fontsize=9)
         subax.axis('off')


def create_comparison_visualization(ax, blur_scores, confidence_scores, all_frames_data, thresholds, method):
    """Create visualization comparing similar frames with different classifications"""
    if not confidence_scores or not all_frames_data or not thresholds or method not in thresholds:
        ax.text(0.5, 0.5, "Insufficient data for comparison.", ha='center', va='center'); ax.axis('off'); return

    available_indices = set(all_frames_data.keys())
    blurry_near = sorted([s for s in confidence_scores if s['is_blurry'] and s['confidence'] > -40 and s['frame'] in available_indices], key=lambda x: abs(x['confidence']))
    sharp_near = sorted([s for s in confidence_scores if not s['is_blurry'] and s['confidence'] < 40 and s['frame'] in available_indices], key=lambda x: abs(x['confidence']))

    if blurry_near and sharp_near:
        blurry_conf = blurry_near[0]
        sharp_conf = sharp_near[0]
        blurry_score = next((s for s in blur_scores if s['frame'] == blurry_conf['frame']), None)
        sharp_score = next((s for s in blur_scores if s['frame'] == sharp_conf['frame']), None)

        if blurry_score and sharp_score:
            blurry_img = all_frames_data[blurry_conf['frame']]
            sharp_img = all_frames_data[sharp_conf['frame']]

            subax1 = ax.inset_axes([0, 0.55, 0.5, 0.45]) # Top left
            subax1.imshow(cv2.cvtColor(blurry_img, cv2.COLOR_BGR2RGB))
            subax1.set_title(f"Blurry: Idx {blurry_conf['frame']}\nConf: {blurry_conf['confidence']}", fontsize=9)
            subax1.axis('off')

            subax2 = ax.inset_axes([0.5, 0.55, 0.5, 0.45]) # Top right
            subax2.imshow(cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB))
            subax2.set_title(f"Sharp: Idx {sharp_conf['frame']}\nConf: {sharp_conf['confidence']}", fontsize=9)
            subax2.axis('off')

            table_data = [
                ["Metric", f"Blurry (Idx {blurry_conf['frame']})", f"Sharp (Idx {sharp_conf['frame']})", "Threshold"],
                ["Laplacian", f"{blurry_score['laplacian']:.1f}", f"{sharp_score['laplacian']:.1f}", f"{thresholds[method]['laplacian']:.1f}"],
                ["Tenengrad", f"{blurry_score['tenengrad']:.1f}", f"{sharp_score['tenengrad']:.1f}", f"{thresholds[method]['tenengrad']:.1f}"],
                ["FFT", f"{blurry_score['fft']:.1f}", f"{sharp_score['fft']:.1f}", f"{thresholds[method]['fft']:.1f}"]
            ]

            table_ax = ax.inset_axes([0.05, 0.05, 0.9, 0.4]) # Bottom table
            table = table_ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.3)

            # Highlight values below threshold
            for i, row in enumerate(table_data[1:], 1):
                metric = row[0].lower()
                try: # Handle potential non-numeric values if errors occurred
                    blurry_val = float(row[1]); sharp_val = float(row[2])
                    threshold_val = thresholds[method][metric]
                    if blurry_val < threshold_val: table[(i, 1)].set_facecolor("#ffcccc")
                    if sharp_val < threshold_val: table[(i, 2)].set_facecolor("#ffcccc")
                except (ValueError, KeyError): continue # Skip coloring if error
            table_ax.axis('off')

            ax.text(0.5, 0.5, "Comparison of Borderline Cases", ha='center', va='center', fontsize=10)
            ax.axis('off')
            ax.set_title("Comparison of Borderline Cases", y=1.0) # Main title for this subplot area

        else: ax.text(0.5, 0.5, "Comparison scores missing.", ha='center'); ax.axis('off')
    else: ax.text(0.5, 0.5, "No suitable comparison frames found.", ha='center'); ax.axis('off')


# --- Argument Parsing (Modified for Directory Input) ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Image sequence blur detection based on subdirectories')
    parser.add_argument('--input', type=str, required=True, help='Path to the main directory containing scene subdirectories with images')
    parser.add_argument('--resize', type=float, default=1.0, help='Resize factor for image frames (e.g., 0.5 for half size)')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame in each sequence')
    # Removed --save_path as output is now directory-based
    parser.add_argument('--method', type=str, default='percentile', choices=['percentile', 'stddev', 'fixed'],
                        help='Threshold method to use for blur detection per sequence')
    parser.add_argument('--stakeholder', action='store_true', help='Generate stakeholder-friendly summary for each sequence')
    parser.add_argument('--confidence', action='store_true', help='Include confidence scores in output (implied by --stakeholder)')
    parser.add_argument('--bg_remove', action='store_true', help='Enable foreground extraction based on edge density')
    parser.add_argument('--visualize_mask', action='store_true', help='Save a visualization of the foreground mask creation steps (requires --bg_remove)')

    return parser.parse_args()

# --- Natural Sort Helper ---
def natural_sort_key(s):
    """ Sorting key for strings containing numbers (e.g., frame_1, frame_10) """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# --- Main Execution Block (Modified for Image Sequences) ---
if __name__ == "__main__":
    try:
        args = parse_arguments()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=" * 40)
        logger.info(f"Image Sequence Blur Detection Script started at {timestamp}")
        logger.info(f"Input Directory: {args.input}")
        logger.info("=" * 40)

        # --- Input Handling: Find Scene Subdirectories and Image Files ---
        input_path = args.input
        sequences_to_process = {} # Dictionary: {sequence_name: [list_of_image_paths]}
        valid_image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

        if not os.path.isdir(input_path):
            logger.error(f"Input path is not a valid directory: {input_path}")
            exit(1)

        logger.info(f"Scanning directory for scene subdirectories: {input_path}")
        subdir_count = 0
        for item_name in os.listdir(input_path):
            item_path = os.path.join(input_path, item_name)
            if os.path.isdir(item_path):
                logger.debug(f"Checking potential scene subdirectory: {item_name}")
                image_files = []
                try:
                    # List files and filter by extension
                    image_files = [os.path.join(item_path, f) for f in os.listdir(item_path)
                                   if f.lower().endswith(valid_image_extensions)]
                except OSError as e:
                    logger.warning(f"Could not read directory {item_path}: {e}. Skipping.")
                    continue

                if image_files:
                    # Sort image files naturally
                    image_files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))
                    sequences_to_process[item_name] = image_files
                    logger.info(f"Found scene '{item_name}' with {len(image_files)} images.")
                    subdir_count += 1
                else:
                    logger.debug(f"Subdirectory '{item_name}' contains no supported image files.")

        if not sequences_to_process:
            logger.error(f"No subdirectories containing supported image files found in: {input_path}")
            exit(1)

        logger.info(f"Found {len(sequences_to_process)} scene(s) to process.")

        # --- Result Setup ---
        results_base_dir = "blur_results_images" # Changed base directory name
        os.makedirs(results_base_dir, exist_ok=True)
        run_dir = os.path.join(results_base_dir, f"analysis_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Results will be saved in: {run_dir}")

        all_results_summary = {} # For aggregated JSON output

        # --- Processing Loop ---
        for sequence_name, image_paths in sequences_to_process.items():
            logger.info("-" * 10 + f" Analyzing Scene: {sequence_name} " + "-" * 10)

            # Create subdirectory for this sequence's results
            sequence_run_dir = os.path.join(run_dir, sequence_name)
            os.makedirs(sequence_run_dir, exist_ok=True)

            # Call the modified analysis function
            analysis_results = analyze_image_sequence(
                image_paths,
                sequence_name, # Pass sequence name
                resize_factor=args.resize,
                skip_frames=args.skip,
                bg_remove=args.bg_remove,
                visualize_mask=args.visualize_mask,
                run_dir=sequence_run_dir # Pass sequence-specific dir for mask viz
            )

            # Check if analysis was successful
            if analysis_results is None or analysis_results[0] is None: # Check if blur_scores is None
                 logger.error(f"Analysis failed or produced no scores for sequence {sequence_name}. Skipping result saving.")
                 all_results_summary[sequence_name] = {"error": "Analysis failed or no scores generated"}
                 continue

            # Unpack results (assuming analyze_image_sequence returns 5 items)
            blur_scores, blur_flags, thresholds, all_frames_data, method_classifications = analysis_results
            all_results_summary[sequence_name] = {} # Initialize summary entry

            if blur_scores:
                # Add basic info to summary
                all_results_summary[sequence_name]['processed_frames'] = len(blur_scores)
                all_results_summary[sequence_name]['total_images'] = len(image_paths)
                all_results_summary[sequence_name]['thresholds'] = thresholds
                all_results_summary[sequence_name]['blur_flags'] = blur_flags # Store flags for all methods

                # Include confidence scores if requested
                confidence_data = None
                if (args.confidence or args.stakeholder) and thresholds and args.method in blur_flags:
                    confidence_data = []
                    for i, score in enumerate(blur_scores):
                        confidence, metric = calculate_blur_confidence(score, blur_scores, thresholds, args.method)
                        confidence_data.append({
                            'frame': score['frame'], # Processed index
                            'confidence': confidence,
                            'leading_metric': metric,
                            'is_blurry': blur_flags[args.method][i]
                        })
                    all_results_summary[sequence_name]['confidence_scores'] = confidence_data

                # --- Save Individual Sequence Results ---
                logger.info(f"Saving results for sequence: {sequence_name}")

                # Save results plot
                plot_path = os.path.join(sequence_run_dir, f"blur_plot_{sequence_name}.png")
                plot_blur_metrics(blur_scores, thresholds, sequence_name, save_path=plot_path)

                # Save confidence data to CSV (if calculated)
                if confidence_data:
                    confidence_csv = os.path.join(sequence_run_dir, f"confidence_scores_{sequence_name}.csv")
                    try:
                        with open(confidence_csv, 'w') as f:
                             # Match header in generate_executive_summary CSV
                             f.write("processed_index,original_index,filename,laplacian,tenengrad,fft,is_blurry,confidence,leading_metric\n")
                             for i, score in enumerate(blur_scores):
                                 conf = confidence_data[i]
                                 # Find original index and filename from blur_scores
                                 orig_idx = score.get('original_frame_index', 'N/A')
                                 filename = score.get('path', 'N/A')
                                 f.write(f"{score['frame']},{orig_idx},{filename},"
                                         f"{score['laplacian']:.2f},{score['tenengrad']:.2f},{score['fft']:.2f},"
                                         f"{conf['is_blurry']},{conf['confidence']},{conf['leading_metric']}\n")
                        logger.info(f"Confidence scores CSV saved to {confidence_csv}")
                    except Exception as e:
                         logger.error(f"Error saving confidence CSV for {sequence_name}: {e}")

                # Generate stakeholder report if requested
                if args.stakeholder:
                     logger.info(f"Generating stakeholder summary for {sequence_name}")
                     # Pass necessary data, including all_frames_data
                     summary_res = generate_executive_summary(
                        blur_scores, blur_flags, thresholds, all_frames_data,
                        method_classifications, sequence_name, args.method, save_dir=sequence_run_dir # Save within sequence dir
                     )
                     if summary_res and summary_res[0]:
                          logger.info(f"Executive summary artifacts generated in {os.path.join(sequence_run_dir, f'summary_{sequence_name}')}")
                     else:
                          logger.error(f"Failed to generate stakeholder summary for {sequence_name}")

                # Visualize sample frames (using all_frames_data)
                visualize_sample_frames(all_frames_data, blur_scores, thresholds, sequence_name, save_dir=sequence_run_dir)

                # Create method comparison visualization (using all_frames_data)
                visualize_method_comparison(all_frames_data, method_classifications, blur_scores, thresholds, sequence_name, save_dir=sequence_run_dir)

                # Save raw numerical results (all methods' flags)
                results_path = os.path.join(sequence_run_dir, f"blur_data_{sequence_name}.csv")
                try:
                    with open(results_path, 'w') as f:
                        f.write("processed_index,original_index,filename,laplacian,tenengrad,fft,is_blurry_percentile,is_blurry_stddev,is_blurry_fixed\n")
                        for i, score in enumerate(blur_scores):
                            orig_idx = score.get('original_frame_index', 'N/A')
                            filename = score.get('path', 'N/A')
                            f.write(f"{score['frame']},{orig_idx},{filename},"
                                    f"{score['laplacian']:.2f},{score['tenengrad']:.2f},{score['fft']:.2f},"
                                    f"{blur_flags['percentile'][i]},{blur_flags['stddev'][i]},{blur_flags['fixed'][i]}\n")
                    logger.info(f"Raw results CSV saved to {results_path}")
                except Exception as e:
                    logger.error(f"Error saving results CSV for {sequence_name}: {e}")

                # Save threshold information
                thresh_path = os.path.join(sequence_run_dir, f"thresholds_{sequence_name}.txt")
                try:
                    with open(thresh_path, 'w') as f:
                        f.write(f"Calculated Thresholds for Sequence: {sequence_name}\n")
                        if thresholds:
                            for method, values in thresholds.items():
                                f.write(f"\n{method.capitalize()} method:\n")
                                for metric, value in values.items():
                                    f.write(f"  {metric.capitalize()}: {value:.2f}\n")
                        else:
                            f.write("\nThresholds could not be calculated.\n")

                        # Add stats about blur detection
                        f.write("\nBlur Detection Results (Processed Frames):\n")
                        if blur_flags:
                            for method, flags in blur_flags.items():
                                if flags: # Check if list is not empty
                                    blur_count = sum(flags)
                                    blur_percentage = (blur_count / len(flags)) * 100
                                    f.write(f"{method.capitalize()} method: Found {blur_count} blurry frames out of {len(flags)} ({blur_percentage:.1f}%)\n")
                                else:
                                     f.write(f"{method.capitalize()} method: No flags calculated.\n")
                        else:
                            f.write("No blur flags calculated.\n")

                    logger.info(f"Threshold information saved to {thresh_path}")
                except Exception as e:
                    logger.error(f"Error saving threshold information for {sequence_name}: {e}")
            else:
                 logger.warning(f"No blur scores generated for {sequence_name}. Skipping result saving.")
                 all_results_summary[sequence_name] = {"error": "No blur scores generated"}


        # --- Aggregated JSON Output ---
        json_output_path = os.path.join(run_dir, "aggregated_summary.json")
        try:
            # Clean up non-serializable data if necessary before saving
            # For now, we assume thresholds and flags are serializable
            with open(json_output_path, 'w') as f:
                json.dump(all_results_summary, f, indent=4)
            logger.info(f"Aggregated summary for all sequences saved to: {json_output_path}")
        except TypeError as e:
             logger.error(f"Error saving aggregated JSON results (likely non-serializable data): {e}")
             # Consider adding more robust serialization (e.g., converting numpy types) if this occurs
        except Exception as e:
             logger.error(f"Error saving aggregated JSON results: {e}")

        logger.info("=" * 40)
        logger.info("Image Sequence Blur Detection Script finished")
        logger.info(f"Overall results directory: {run_dir}")
        logger.info("=" * 40)

    except SystemExit:
         logger.info("Exiting script.") # Handle exit() calls gracefully
    except Exception as e:
        logger.exception(f"Unhandled exception in main execution block: {e}") # Log full traceback
        exit(1) # Ensure script exits with error status on unhandled exception
