import unittest
import os
import json
import numpy as np
from unittest.mock import patch, mock_open, MagicMock, PropertyMock

# Import the detector
from src.cd_fsod_detector import CDFSODDetector


class TestCDFSODDetector(unittest.TestCase):
    def setUp(self):
        # Create mock detection data for testing
        self.mock_detections = {
            "0.json": [
                {"coordinates": [10, 10, 50, 50], "label": "monitor", "confidence": 0.9},
                {"coordinates": [100, 100, 150, 150], "label": "uniform", "confidence": 0.8}
            ],
            "1.json": [
                {"coordinates": [12, 12, 52, 52], "label": "monitor", "confidence": 0.85},
                {"coordinates": [200, 200, 250, 250], "label": "time_stick", "confidence": 0.7}
            ],
            "2.json": [
                {"coordinates": [15, 15, 55, 55], "label": "monitor", "confidence": 0.83}
            ],
            # Gap for monitor (frames 3-12)
            "13.json": [
                {"coordinates": [20, 20, 60, 60], "label": "monitor", "confidence": 0.88}
            ],
            "14.json": [
                {"coordinates": [22, 22, 62, 62], "label": "monitor", "confidence": 0.86}
            ],
            # Uniform disappears after frame 0 and reappears in frame 20
            "20.json": [
                {"coordinates": [105, 105, 155, 155], "label": "uniform", "confidence": 0.82}
            ]
        }
        
        # Mock directory path
        self.json_dir = "/mock/json/directory"
        
    # Create a helper class for mock images
    class MockImage(np.ndarray):
        def __new__(cls, input_array, frame_idx):
            # Create a new numpy array
            obj = np.asarray(input_array).view(cls)
            # Add frame_idx as a separate attribute
            obj._frame_idx = frame_idx
            return obj
            
        def __array_finalize__(self, obj):
            if obj is None: return
            self._frame_idx = getattr(obj, '_frame_idx', None)
            
        @property
        def frame_idx(self):
            return self._frame_idx
    
    @patch('os.listdir')
    def test_initialization_loads_all_json_files(self, mock_listdir):
        json_files = ["0.json", "1.json", "2.json", "13.json", "14.json", "20.json"]
        mock_listdir.return_value = json_files
        
        # Create a dictionary mapping file paths to their contents for the mock_open
        mock_file_data = {}
        for json_file in json_files:
            file_path = os.path.join(self.json_dir, json_file)
            mock_file_data[file_path] = json.dumps(self.mock_detections[json_file])
        
        # Create a context manager for the patched open function
        m = mock_open()
        
        # Define a custom side effect function for the mock
        def side_effect(filename, *args, **kwargs):
            if filename in mock_file_data:
                file_mock = m.return_value
                file_mock.read.return_value = mock_file_data[filename]
                return file_mock
            raise FileNotFoundError(f"Mock file not found: {filename}")
        
        # Patch both open and json.load
        with patch('builtins.open', side_effect=side_effect), \
             patch('json.load', side_effect=lambda f: json.loads(f.read())):
            
            # Create detector with the mocked json directory
            detector = CDFSODDetector(self.json_dir)
            
            # Verify that listdir was called
            self.assertEqual(mock_listdir.call_count, 1)
            
            # Skip checking call_count on m since with our implementation,
            # it's not directly tracking the number of file opens
            # Just verify detector was initialized successfully
            self.assertIsInstance(detector, CDFSODDetector)
            self.assertEqual(len(detector.detections_by_frame), len(json_files))
    
    @patch('os.listdir')
    def test_first_appearance_detection(self, mock_listdir):
        json_files = ["0.json", "1.json"]
        mock_listdir.return_value = json_files
        
        # Create a dictionary mapping file paths to their contents
        mock_file_data = {}
        for json_file in json_files:
            file_path = os.path.join(self.json_dir, json_file)
            mock_file_data[file_path] = json.dumps(self.mock_detections[json_file])
        
        # Create a context manager for the patched open function
        m = mock_open()
        
        # Define a custom side effect function for the mock
        def side_effect(filename, *args, **kwargs):
            if filename in mock_file_data:
                file_mock = m.return_value
                file_mock.read.return_value = mock_file_data[filename]
                return file_mock
            raise FileNotFoundError(f"Mock file not found: {filename}")
        
        # Patch both open and json.load
        with patch('builtins.open', side_effect=side_effect), \
             patch('json.load', side_effect=lambda f: json.loads(f.read())):
            
            # Create detector
            detector = CDFSODDetector(self.json_dir)
            
            # Create mock images with frame indices using our helper class
            frame0_image = self.MockImage(np.zeros((100, 100, 3)), 0)
            frame1_image = self.MockImage(np.zeros((100, 100, 3)), 1)
            
            # Get actual results from the detector
            frame0_results = detector.detect(frame0_image, ["monitor", "uniform", "time_stick"])
            frame1_results = detector.detect(frame1_image, ["monitor", "uniform", "time_stick"])
            
            # Verify the number of detections in each frame
            self.assertEqual(len(frame0_results["boxes"]), 2)  # Two objects in frame 0
            self.assertEqual(len(frame1_results["boxes"]), 1)  # Only new time_stick in frame 1
            
            # Verify the labels in each frame
            self.assertIn("monitor", frame0_results["labels"])
            self.assertIn("uniform", frame0_results["labels"])
            self.assertIn("time_stick", frame1_results["labels"])
    
    @patch('os.listdir')
    def test_reappearance_after_gap(self, mock_listdir):
        json_files = ["0.json", "1.json", "2.json", "13.json", "14.json"]
        mock_listdir.return_value = json_files
        
        # Create a dictionary mapping file paths to their contents
        mock_file_data = {}
        for json_file in json_files:
            file_path = os.path.join(self.json_dir, json_file)
            mock_file_data[file_path] = json.dumps(self.mock_detections[json_file])
        
        # Create a context manager for the patched open function
        m = mock_open()
        
        # Define a custom side effect function for the mock
        def side_effect(filename, *args, **kwargs):
            if filename in mock_file_data:
                file_mock = m.return_value
                file_mock.read.return_value = mock_file_data[filename]
                return file_mock
            raise FileNotFoundError(f"Mock file not found: {filename}")
        
        # Patch both open and json.load
        with patch('builtins.open', side_effect=side_effect), \
             patch('json.load', side_effect=lambda f: json.loads(f.read())):
            
            # Create detector with a min_gap_frames of 5 (so 10-frame gap will trigger reappearance)
            detector = CDFSODDetector(self.json_dir, min_gap_frames=5)
            
            # Create mock image with frame index
            frame13_image = self.MockImage(np.zeros((100, 100, 3)), 13)
            
            # Get actual results from the detector
            frame13_results = detector.detect(frame13_image, ["monitor"])
            
            # Verify monitor is detected as reappearing
            self.assertEqual(len(frame13_results["boxes"]), 1)
            self.assertIn("monitor", frame13_results["labels"])
    
    @patch('os.listdir')
    def test_min_gap_threshold(self, mock_listdir):
        json_files = list(self.mock_detections.keys())
        mock_listdir.return_value = json_files
        
        # Create a dictionary mapping file paths to their contents
        mock_file_data = {}
        for json_file in json_files:
            file_path = os.path.join(self.json_dir, json_file)
            mock_file_data[file_path] = json.dumps(self.mock_detections[json_file])
        
        # Create a context manager for the patched open function
        m = mock_open()
        
        # Define a custom side effect function for the mock
        def side_effect(filename, *args, **kwargs):
            if filename in mock_file_data:
                file_mock = m.return_value
                file_mock.read.return_value = mock_file_data[filename]
                return file_mock
            raise FileNotFoundError(f"Mock file not found: {filename}")
        
        # Patch both open and json.load
        with patch('builtins.open', side_effect=side_effect), \
             patch('json.load', side_effect=lambda f: json.loads(f.read())):
            
            # Test with different min_gap_frames values
            
            # 1. Small gap (should detect reappearance at frame 13 since 13-2 > 5)
            detector_small_gap = CDFSODDetector(self.json_dir, min_gap_frames=5)
            
            # 2. Large gap (should NOT detect reappearance at frame 13 since 13-2 < 15)
            detector_large_gap = CDFSODDetector(self.json_dir, min_gap_frames=15)
            
            # Create mock image with frame index
            frame13_image = self.MockImage(np.zeros((100, 100, 3)), 13)
            
            # Get results with both detectors
            small_gap_results = detector_small_gap.detect(frame13_image, ["monitor"])
            large_gap_results = detector_large_gap.detect(frame13_image, ["monitor"])
            
            # With small gap threshold, monitor should be detected as reappearing
            self.assertEqual(len(small_gap_results["boxes"]), 1)
            
            # With large gap threshold, monitor should NOT be detected (gap too small)
            self.assertEqual(len(large_gap_results["boxes"]), 0)
    
    @patch('os.listdir')
    def test_confidence_threshold_filtering(self, mock_listdir):
        # Setup custom detections with varying confidence scores
        confidence_test_detections = {
            "0.json": [
                {"coordinates": [10, 10, 50, 50], "label": "monitor", "confidence": 0.9},
                {"coordinates": [100, 100, 150, 150], "label": "uniform", "confidence": 0.3}  # Low confidence
            ]
        }
        
        json_files = ["0.json"]
        mock_listdir.return_value = json_files
        
        # Create a dictionary mapping file paths to their contents
        mock_file_data = {}
        for json_file in json_files:
            file_path = os.path.join(self.json_dir, json_file)
            mock_file_data[file_path] = json.dumps(confidence_test_detections[json_file])
        
        # Create a context manager for the patched open function
        m = mock_open()
        
        # Define a custom side effect function for the mock
        def side_effect(filename, *args, **kwargs):
            if filename in mock_file_data:
                file_mock = m.return_value
                file_mock.read.return_value = mock_file_data[filename]
                return file_mock
            raise FileNotFoundError(f"Mock file not found: {filename}")
        
        # Patch both open and json.load
        with patch('builtins.open', side_effect=side_effect), \
             patch('json.load', side_effect=lambda f: json.loads(f.read())):
            
            # Test with different confidence thresholds
            
            # 1. Low threshold (should detect both objects)
            detector_low_threshold = CDFSODDetector(self.json_dir, confidence_threshold=0.2)
            
            # 2. High threshold (should only detect monitor)
            detector_high_threshold = CDFSODDetector(self.json_dir, confidence_threshold=0.5)
            
            # Create mock image with frame index
            frame0_image = self.MockImage(np.zeros((100, 100, 3)), 0)
            
            # Get results with both detectors
            low_threshold_results = detector_low_threshold.detect(frame0_image, ["monitor", "uniform"])
            high_threshold_results = detector_high_threshold.detect(frame0_image, ["monitor", "uniform"])
            
            # With low threshold, both objects should be detected
            self.assertEqual(len(low_threshold_results["boxes"]), 2)
            
            # With high threshold, only monitor should be detected
            self.assertEqual(len(high_threshold_results["boxes"]), 1)
            self.assertIn("monitor", high_threshold_results["labels"])
    
    @patch('os.listdir')
    def test_text_query_filtering(self, mock_listdir):
        json_files = ["0.json"]
        mock_listdir.return_value = json_files
        
        # Create a dictionary mapping file paths to their contents
        mock_file_data = {}
        for json_file in json_files:
            file_path = os.path.join(self.json_dir, json_file)
            mock_file_data[file_path] = json.dumps(self.mock_detections[json_file])
        
        # Create a context manager for the patched open function
        m = mock_open()
        
        # Define a custom side effect function for the mock
        def side_effect(filename, *args, **kwargs):
            if filename in mock_file_data:
                file_mock = m.return_value
                file_mock.read.return_value = mock_file_data[filename]
                return file_mock
            raise FileNotFoundError(f"Mock file not found: {filename}")
        
        # Patch both open and json.load
        with patch('builtins.open', side_effect=side_effect), \
             patch('json.load', side_effect=lambda f: json.loads(f.read())):
            
            detector = CDFSODDetector(self.json_dir)
            
            # Create mock image with frame index
            frame0_image = self.MockImage(np.zeros((100, 100, 3)), 0)
            
            # Get results with different text queries
            all_results = detector.detect(frame0_image, ["monitor", "uniform", "time_stick"])
            monitor_only_results = detector.detect(frame0_image, ["monitor"])
            no_match_results = detector.detect(frame0_image, ["keyboard"])
            
            # With all queries, should detect both monitor and uniform
            self.assertEqual(len(all_results["boxes"]), 2)
            
            # With only monitor query, should only detect monitor
            self.assertEqual(len(monitor_only_results["boxes"]), 1)
            self.assertIn("monitor", monitor_only_results["labels"])
            
            # With keyboard query, should detect nothing
            self.assertEqual(len(no_match_results["boxes"]), 0)
    
    @patch('os.listdir')
    def test_continuous_detection_exclusion(self, mock_listdir):
        """Test that objects are only detected on first appearance and reappearance, not in continuous frames."""
        # Create test data with an object that appears in consecutive frames
        continuous_test_detections = {
            "0.json": [
                {"coordinates": [10, 10, 50, 50], "label": "monitor", "confidence": 0.9}
            ],
            "1.json": [
                {"coordinates": [12, 12, 52, 52], "label": "monitor", "confidence": 0.85}
            ],
            "2.json": [
                {"coordinates": [14, 14, 54, 54], "label": "monitor", "confidence": 0.88}
            ],
            # Gap of 11 frames (3-14)
            "15.json": [
                {"coordinates": [20, 20, 60, 60], "label": "monitor", "confidence": 0.82}
            ],
            "16.json": [
                {"coordinates": [22, 22, 62, 62], "label": "monitor", "confidence": 0.80}
            ],
            # Another gap of 15 frames (17-31)
            "32.json": [
                {"coordinates": [30, 30, 70, 70], "label": "monitor", "confidence": 0.85}
            ],
            "33.json": [
                {"coordinates": [32, 32, 72, 72], "label": "monitor", "confidence": 0.83}
            ]
        }
        
        json_files = list(continuous_test_detections.keys())
        mock_listdir.return_value = json_files
        
        # Create a dictionary mapping file paths to their contents
        mock_file_data = {}
        for json_file in json_files:
            file_path = os.path.join(self.json_dir, json_file)
            mock_file_data[file_path] = json.dumps(continuous_test_detections[json_file])
        
        # Create a context manager for the patched open function
        m = mock_open()
        
        # Define a custom side effect function for the mock
        def side_effect(filename, *args, **kwargs):
            if filename in mock_file_data:
                file_mock = m.return_value
                file_mock.read.return_value = mock_file_data[filename]
                return file_mock
            raise FileNotFoundError(f"Mock file not found: {filename}")
        
        # Patch both open and json.load
        with patch('builtins.open', side_effect=side_effect), \
             patch('json.load', side_effect=lambda f: json.loads(f.read())):
            
            # Create detector with a min_gap_frames of 10
            detector = CDFSODDetector(self.json_dir, min_gap_frames=10, iou_threshold=0.3)
            
            # Create mock images with frame indices
            frame0_image = self.MockImage(np.zeros((100, 100, 3)), 0)
            frame1_image = self.MockImage(np.zeros((100, 100, 3)), 1)
            frame2_image = self.MockImage(np.zeros((100, 100, 3)), 2)
            frame15_image = self.MockImage(np.zeros((100, 100, 3)), 15)
            frame16_image = self.MockImage(np.zeros((100, 100, 3)), 16)
            frame32_image = self.MockImage(np.zeros((100, 100, 3)), 32)
            frame33_image = self.MockImage(np.zeros((100, 100, 3)), 33)
            
            # Get results for each frame
            frame0_results = detector.detect(frame0_image, ["monitor"])
            frame1_results = detector.detect(frame1_image, ["monitor"])
            frame2_results = detector.detect(frame2_image, ["monitor"])
            frame15_results = detector.detect(frame15_image, ["monitor"])
            frame16_results = detector.detect(frame16_image, ["monitor"])
            frame32_results = detector.detect(frame32_image, ["monitor"])
            frame33_results = detector.detect(frame33_image, ["monitor"])
            
            # Verify that monitor is only detected in frame 0 (first appearance),
            # frame 15 (reappearance after gap > 10 frames), and
            # frame 32 (reappearance after gap > 10 frames)
            self.assertEqual(len(frame0_results["boxes"]), 1, "Monitor should be detected on first appearance")
            self.assertEqual(len(frame1_results["boxes"]), 0, "Monitor should not be detected in consecutive frame")
            self.assertEqual(len(frame2_results["boxes"]), 0, "Monitor should not be detected in consecutive frame")
            self.assertEqual(len(frame15_results["boxes"]), 1, "Monitor should be detected on reappearance after gap")
            self.assertEqual(len(frame16_results["boxes"]), 0, "Monitor should not be detected in consecutive frame after reappearance")
            self.assertEqual(len(frame32_results["boxes"]), 1, "Monitor should be detected on second reappearance after gap")
            self.assertEqual(len(frame33_results["boxes"]), 0, "Monitor should not be detected in consecutive frame after second reappearance")
            
            # Verify labels when detections are present
            self.assertIn("monitor", frame0_results["labels"], "Monitor should be in frame 0 labels")
            self.assertIn("monitor", frame15_results["labels"], "Monitor should be in frame 15 labels")
            self.assertIn("monitor", frame32_results["labels"], "Monitor should be in frame 32 labels")


if __name__ == "__main__":
    unittest.main() 