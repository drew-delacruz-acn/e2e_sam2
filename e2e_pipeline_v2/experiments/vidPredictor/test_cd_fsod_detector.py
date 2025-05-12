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
    def test_integration_with_label_mapping(self, mock_listdir):
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
            
            # Create a label mapping
            label_mapping = {
                "monitor": "screen",
                "uniform": "clothing"
            }
            
            detector = CDFSODDetector(self.json_dir, label_mapping=label_mapping)
            
            # Create mock image with frame index
            frame0_image = self.MockImage(np.zeros((100, 100, 3)), 0)
            
            # Get results with mapped queries
            mapped_results = detector.detect(frame0_image, ["screen", "clothing"])
            
            # Should detect both objects with mapped labels
            self.assertEqual(len(mapped_results["boxes"]), 2)
            self.assertIn("screen", mapped_results["labels"])
            self.assertIn("clothing", mapped_results["labels"])


if __name__ == "__main__":
    unittest.main() 