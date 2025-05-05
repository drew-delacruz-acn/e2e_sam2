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

# Check for CUDA availability
USE_CUDA = False
HAVE_CUPY = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        USE_CUDA = True
        logger.info("CUDA is available for OpenCV")
        try:
            import cupy as cp
            HAVE_CUPY = True
            logger.info("CuPy is available for GPU-accelerated FFT")
        except ImportError:
            logger.info("CuPy not found, using NumPy for FFT")
    else:
        logger.info("CUDA is not available, using CPU processing")
except:
    logger.info("Could not check CUDA availability, using CPU processing")

def laplacian_blur(gray, use_cuda=USE_CUDA):
    try:
        if use_cuda and HAVE_CUPY:
            # Convert to CuPy array for GPU processing
            gray_gpu = cp.asarray(gray)
            # No direct Laplacian in CuPy, but we can use a kernel
            kernel = cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32)
            laplacian = cp.abs(cp.correlate2d(gray_gpu, kernel, mode='same'))
            return float(cp.var(laplacian).get())
        elif use_cuda:
            # Use OpenCV CUDA module
            gray_cuda = cv2.cuda_GpuMat()
            gray_cuda.upload(gray)
            laplacian_cuda = cv2.cuda.createLaplacianFilter(cv2.CV_64F, 1)
            result_cuda = laplacian_cuda.apply(gray_cuda)
            result = result_cuda.download()
            return result.var()
        else:
            # CPU version
            return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception as e:
        logger.error(f"Error in laplacian_blur: {e}")
        # Fallback to CPU if CUDA failed
        try:
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0

def tenengrad_blur(gray, use_cuda=USE_CUDA):
    try:
        if use_cuda:
            # Use OpenCV CUDA module for Sobel
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(gray)
            
            # Create CUDA Sobel filters
            sobel_x = cv2.cuda.createSobelFilter(cv2.CV_64F, 1, 0)
            sobel_y = cv2.cuda.createSobelFilter(cv2.CV_64F, 0, 1)
            
            # Apply filters
            gx_cuda = sobel_x.apply(gpu_mat)
            gy_cuda = sobel_y.apply(gpu_mat)
            
            # Download results
            gx = gx_cuda.download()
            gy = gy_cuda.download()
            
            # Calculate magnitude
            g = np.sqrt(gx**2 + gy**2)
            return np.mean(g)
        else:
            # CPU version
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            g = np.sqrt(gx**2 + gy**2)
            return np.mean(g)
    except Exception as e:
        logger.error(f"Error in tenengrad_blur: {e}")
        # Fallback to CPU
        try:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            g = np.sqrt(gx**2 + gy**2)
            return np.mean(g)
        except:
            return 0

def fft_blur(gray, cutoff_size=30, use_cuda=USE_CUDA):
    try:
        start_time = time.time()
        
        if use_cuda and HAVE_CUPY:
            # GPU-accelerated FFT with CuPy
            gray_gpu = cp.asarray(gray)
            f = cp.fft.fft2(gray_gpu)
            fshift = cp.fft.fftshift(f)
            
            # Create mask
            h, w = gray.shape
            cx, cy = w//2, h//2
            mask = cp.ones_like(fshift)
            mask[cy-cutoff_size:cy+cutoff_size, cx-cutoff_size:cx+cutoff_size] = 0
            
            fshift = fshift * mask
            ishift = cp.fft.ifftshift(fshift)
            img_back = cp.fft.ifft2(ishift)
            magnitude = cp.abs(img_back)
            result = float(cp.mean(magnitude).get())
        else:
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
        # Fallback to CPU if needed
        try:
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            h, w = gray.shape
            cx, cy = w//2, h//2
            fshift[cy-cutoff_size:cy+cutoff_size, cx-cutoff_size:cx+cutoff_size] = 0
            ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(ishift)
            magnitude = np.abs(img_back)
            return np.mean(magnitude)
        except:
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

def analyze_video(video_path, resize_factor=1.0, use_cuda=USE_CUDA, skip_frames=1):
    start_time = time.time()
    logger.info(f"Starting video analysis on: {video_path}")
    logger.info(f"Using CUDA: {use_cuda}")
    logger.info(f"Resize factor: {resize_factor}")
    logger.info(f"Processing every {skip_frames} frame(s)")
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None, None
    
    # -- MAIN --
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Error opening video file")
            return None, None
            
        video_info = get_video_info(cap)
        blur_scores = []
        frame_idx = 0
        processed_count = 0
        processing_times = {
            'laplacian': [], 
            'tenengrad': [], 
            'fft': []
        }
        
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
                # Resize large frames if needed
                if resize_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                    frame = cv2.resize(frame, (new_w, new_h))
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Measure performance of each algorithm
                t1 = time.time()
                lap = laplacian_blur(gray, use_cuda)
                t2 = time.time()
                processing_times['laplacian'].append(t2-t1)
                
                t1 = time.time()
                ten = tenengrad_blur(gray, use_cuda)
                t2 = time.time()
                processing_times['tenengrad'].append(t2-t1)
                
                t1 = time.time()
                fft = fft_blur(gray, use_cuda=use_cuda)
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
        return None, None

    # -- TEMPORAL ANALYSIS --
    logger.info("Starting temporal analysis")
    blur_flags = []
    try:
        for i, scores in enumerate(blur_scores):
            lap, ten, fft = scores['laplacian'], scores['tenengrad'], scores['fft']
            sharp = (
                lap > LAPLACIAN_THRESH and
                ten > TENENGRAD_THRESH and
                fft > FFT_THRESH
            )

            # Compare to neighbor average (temporal check)
            if 1 <= i < len(blur_scores) - 1:
                prev = blur_scores[i-1]['laplacian']
                nxt = blur_scores[i+1]['laplacian']
                if lap < 0.5 * ((prev + nxt) / 2):
                    sharp = False
                    logger.debug(f"Frame {scores['frame']} failed temporal check")

            blur_flags.append(not sharp)
            
            # Log detailed analysis for some frames
            if i % 10 == 0 or not sharp:
                logger.info(f"Frame {scores['frame']:04d}: Sharp={sharp} | Lap={lap:.2f} | Ten={ten:.2f} | FFT={fft:.2f}")
    
    except Exception as e:
        logger.error(f"Error during temporal analysis: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"Video analysis completed in {total_time:.2f} seconds")
    if processed_count > 0:
        logger.info(f"Average time per frame: {total_time/processed_count:.4f} seconds")
    
    # Calculate blur statistics
    if blur_flags:
        blur_count = sum(blur_flags)
        blur_percentage = (blur_count / len(blur_flags)) * 100
        logger.info(f"Found {blur_count} blurry frames out of {len(blur_flags)} ({blur_percentage:.1f}%)")
    
    return blur_scores, blur_flags

def plot_blur_metrics(blur_scores, save_path=None):
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

        plt.figure(figsize=(10, 6))
        plt.plot(frames, lap_vals, label='Laplacian')
        plt.plot(frames, ten_vals, label='Tenengrad')
        plt.plot(frames, fft_vals, label='FFT')
        plt.title("Blur Metrics per Frame")
        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.legend()
        
        # Add threshold lines
        plt.axhline(y=LAPLACIAN_THRESH, color='blue', linestyle='--', alpha=0.5)
        plt.axhline(y=TENENGRAD_THRESH, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=FFT_THRESH, color='green', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"Error generating plot: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Video blur detection with multiple methods')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--resize', type=float, default=1.0, help='Resize factor for video frames (e.g., 0.5 for half size)')
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--save_path', type=str, help='Path to save the results plot')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=" * 40)
        logger.info(f"Blur Detection Script started at {timestamp}")
        logger.info("=" * 40)
        
        video_path = "/home/ubuntu/code/drew/e2e_sam2/data/Scenes 001-020__5B14-5a- only_20230816051800694.mp4"  # Replace with your video file
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
        else:
            results_dir = "blur_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Determine if CUDA should be used
            use_cuda = USE_CUDA and not args.no_cuda
            
            # Run analysis with options from command line
            blur_scores, blur_flags = analyze_video(
                video_path, 
                resize_factor=args.resize,
                use_cuda=use_cuda,
                skip_frames=args.skip
            )
            
            if blur_scores:
                # Save results to file
                plot_path = args.save_path if args.save_path else os.path.join(results_dir, f"blur_plot_{timestamp}.png")
                plot_blur_metrics(blur_scores, save_path=plot_path)
                
                # Save numerical results
                results_path = os.path.join(results_dir, f"blur_data_{timestamp}.csv")
                try:
                    with open(results_path, 'w') as f:
                        f.write("frame,laplacian,tenengrad,fft,is_blurry\n")
                        for i, score in enumerate(blur_scores):
                            f.write(f"{score['frame']},{score['laplacian']:.2f},{score['tenengrad']:.2f},{score['fft']:.2f},{blur_flags[i]}\n")
                    logger.info(f"Results saved to {results_path}")
                except Exception as e:
                    logger.error(f"Error saving results: {e}")
            
        logger.info("=" * 40)
        logger.info("Blur Detection Script finished")
        logger.info("=" * 40)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
