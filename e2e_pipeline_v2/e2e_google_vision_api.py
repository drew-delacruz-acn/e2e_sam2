from Object_Detection.google_vision import ObjectDetector
from Image_to_Frame.image_to_frame_neo4j import Neo4jSearch, in_video, gt_objects_in_video
from utils import extract_frames
from PIL import Image, ImageDraw, ImageFont, ImageOps
from dotenv import load_dotenv
from collections import defaultdict
import json
import os
import glob
from Image_to_Frame.test_set import vid_list
load_dotenv()

NEO4J_DB = 'ipid'
valid_objects = ["Loki's armor (THE AVENGERS)", "Sylvie's machete", "Boastful Loki's hammer", 'Tesseract', 'TemPad', 'Thanos copter', "Loki's TVA jacket", "Sylvie's horned headpiece", "Loki's dagger", 'Aligator Loki Plushie', 'Laevateinn', 'Reset Charge', 'Time Collar', 'Asgardian device', "Loki's Scepter", 'Time Stick']

def crop_bounding_boxes(image_path, detection):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x_min, y_min, x_max, y_max = detection["coordinates"]
    left = int(x_min * width)
    top = int(y_min * height)
    right = int(x_max * width)
    bottom = int(y_max * height)
    cropped = image.crop((left, top, right, bottom))
    coords = [left, top, right, bottom]
    return cropped, coords

def run_object_detection(image_path):
    detector = ObjectDetector()
    detections = detector.detect_objects(image_path)
    return detections

def list_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return file_paths

def add_white_border(image_path, output_path, min_width=128, min_height=128):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Calculate padding
    pad_left = max((min_width - width) // 2, 0)
    pad_right = max(min_width - width - pad_left, 0)
    pad_top = max((min_height - height) // 2, 0)
    pad_bottom = max(min_height - height - pad_top, 0)

    # Apply padding (fill with white)
    padded_image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill='white')
    padded_image.save(output_path)

    return padded_image

def main(video_folder, model, threshold, masked):
    os.makedirs(os.path.dirname('./e2e_data/frames'), exist_ok=True)
    os.makedirs(os.path.dirname('./e2e_data/objects'), exist_ok=True)
    os.makedirs(os.path.dirname('./e2e_data/detections'), exist_ok=True)

    driver = Neo4jSearch(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"), NEO4J_DB, model)
    list_results = []
    
    for video_file in os.listdir(video_folder):
        if not video_file.endswith('.mp4'):
            continue
        if video_file not in vid_list:
            continue
        video_path = os.path.join(video_folder, video_file)
        video_name = video_file.replace('.mp4','')
        frame_folder = f'./e2e_data/frames/{video_name}'
        if os.path.exists(frame_folder):
            print('SKIPPING EXTRACTION')
            frame_paths = list_file_paths(frame_folder)
        else:
            frame_paths = extract_frames(video_path, frame_folder)

        for frame in frame_paths:
            frame_num = int(os.path.splitext(os.path.basename(frame))[0])
            
            detections_path = f'./e2e_data/detections/{video_name}/{frame_num}.json'
            os.makedirs(os.path.dirname(detections_path), exist_ok=True)
            if os.path.exists(detections_path):
                print('SKIPPING DETECTION')
                with open(detections_path, 'r') as f:
                    detections = json.load(f)
            else:
                detections = run_object_detection(frame)
                with open(detections_path, 'w') as f:
                    json.dump(detections, f, indent=4)
            
            for idx, det in enumerate(detections):
                img_name = det['label']
                new_img_path = f"./e2e_data/objects/{video_name}/{frame_num}_{img_name}_{idx}.png"
                os.makedirs(f"./e2e_data/objects/{video_name}", exist_ok=True)

                if os.path.exists(new_img_path):
                    print('SKIPPING CROPPING')
                else:
                    output_image, coords = crop_bounding_boxes(frame, det)
                    output_image.save(new_img_path)

                try:
                    search_results = driver.run_search(new_img_path, threshold, masked)
                except:
                    new_new_img = add_white_border(new_img_path, new_img_path.split(".")[0]+"_new.png")
                    search_results = driver.run_search( new_img_path.split(".")[0]+"_new.png", threshold, masked)
                if len(search_results):
                    for item in search_results:
                        predicted_object = item['predicted_object']
                        score = item['similarity_score']
                        version = item['object_version']
                        result = in_video(predicted_object, video_file)

                        list_results.append({
                            'video': video_file,
                            'frame': frame_num,
                            'predicted_object': predicted_object,
                            'object_version': version,
                            'similarity_score': score,
                            'model': model,
                            'ground_truth_object_in_video': result,
                            # 'bbox_coordinates': coords,
                            'bbox_path': new_img_path
                        })

    with open(f'/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_data/results/e2e_results_{model}_allVids.json', 'w') as f:
        json.dump(list_results, f, indent=4)

    return list_results

def parse_occurrence(occurrence_str):
    if "_" not in occurrence_str:
        return None, None, None
    try:
        video_part, frame_part = occurrence_str.rsplit("_", 1)
        start, end = map(int, frame_part.split(":"))
        return video_part, start, end
    except Exception:
        return None, None, None

def build_gt_frame_map(gt_schema):
    gt_frame_map = defaultdict(lambda: defaultdict(set))
    for item in gt_schema:
        obj_name = item["name"]
        for occ in item["occurences"]:
            video, start, end = parse_occurrence(occ)
            if video and start is not None and end is not None:
                for frame in range(start, end + 1):
                    gt_frame_map[video][frame].add(obj_name)
    return gt_frame_map

def build_gt_frame_map_scaled(gt_schema, actual_frame_count, model_fps=24, labelstudio_fps=30):
    gt_frame_map = defaultdict(lambda: defaultdict(set))

    for item in gt_schema:
        obj_name = item["name"].strip()

        for occ in item["occurences"]:
            video, start, end = parse_occurrence(occ)
            if video not in actual_frame_count:
                continue

            model_total_frames = actual_frame_count[video]

            # Convert LS frame index to model frame index via FPS ratio scaling
            scaled_start = int((start * model_fps) / labelstudio_fps)
            scaled_end = int((end * model_fps) / labelstudio_fps)

            # Clip to actual available model frames
            scaled_start = max(0, min(scaled_start, model_total_frames - 1))
            scaled_end = max(0, min(scaled_end, model_total_frames - 1))

            for frame in range(scaled_start, scaled_end + 1):
                gt_frame_map[video][frame].add(obj_name)

    return gt_frame_map

def build_pred_frame_map(results):
    pred_frame_map = defaultdict(lambda: defaultdict(set))
    for pred in results:
        video = pred["video"]
        frame = pred["frame"]
        obj = pred["predicted_object"]
        pred_frame_map[video][frame].add(obj)
    return pred_frame_map

import os

def get_actual_frame_counts_from_folder(frames_root_path):
    frame_counts = {}

    for video_folder in os.listdir(frames_root_path):
        video_folder_path = os.path.join(frames_root_path, video_folder)
        if os.path.isdir(video_folder_path):
            frame_files = [f for f in os.listdir(video_folder_path) if f.endswith('.jpg')]
            # Assuming frames are labeled as '0.png', '1.png', ...
            frame_indices = []
            for f in frame_files:
                try:
                    index = int(os.path.splitext(f)[0])
                    frame_indices.append(index)
                except ValueError:
                    continue
            if frame_indices:
                max_index = max(frame_indices)
                # +1 because frames are 0-indexed
                frame_counts[video_folder + ".mp4"] = max_index + 1

    return frame_counts

def calculate_fscore_frame(results_json_path, gt_schema_path, model_fps=24, labelstudio_fps=30):
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    with open(gt_schema_path, 'r') as f:
        gt_schema = json.load(f)

    gt_frame_map = build_gt_frame_map_scaled(
        gt_schema,
        actual_frame_count=get_actual_frame_counts_from_folder("/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_data/frames"), 
        model_fps=model_fps,
        labelstudio_fps=labelstudio_fps
    )
    pred_frame_map = build_pred_frame_map(results)

    metrics_by_video = {}
    total_TP = total_FP = total_FN = total_TN = 0
    total_frames = 0

    for video in gt_frame_map.keys() | pred_frame_map.keys():
        valid_objects = set(obj.strip() for obj in gt_objects_in_video(video))
        all_frames = set(gt_frame_map[video].keys()) | set(pred_frame_map[video].keys())
        for frame in all_frames:
            total_frames += 1
            gt_objs = gt_frame_map[video].get(frame, set())
            pred_objs = pred_frame_map[video].get(frame, set())

            for obj in valid_objects:
                in_gt = obj in gt_objs
                in_pred = obj in pred_objs

                if in_gt and in_pred:
                    total_TP += 1
                elif not in_gt and in_pred:
                    total_FP += 1
                elif in_gt and not in_pred:
                    total_FN += 1
                elif not in_gt and not in_pred:
                    total_TN += 1

        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_metrics = {
    "avg_true_positives": total_TP / total_frames,
    "avg_false_positives": total_FP / total_frames,
    "avg_false_negatives": total_FN / total_frames,
    "avg_true_negatives": total_TN / total_frames,
    "avg_precision": total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0,
    "avg_recall": total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0,
    "avg_f1_score": (2 * total_TP) / (2 * total_TP + total_FP + total_FN) if (2 * total_TP + total_FP + total_FN) > 0 else 0.0
    }
    print(avg_metrics)

    return metrics_by_video

def calculate_fscore_frame_agg_by_video(results_json_path, gt_schema_path, model_fps=24, labelstudio_fps=30):
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    with open(gt_schema_path, 'r') as f:
        gt_schema = json.load(f)

    gt_frame_map = build_gt_frame_map_scaled(
        gt_schema,
        actual_frame_count=get_actual_frame_counts_from_folder("/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_data/frames"), 
        model_fps=model_fps,
        labelstudio_fps=labelstudio_fps
    )
    pred_frame_map = build_pred_frame_map(results)

    metrics_by_video = {}

    for video in gt_frame_map.keys() | pred_frame_map.keys():
        valid_objects = set(obj.strip() for obj in gt_objects_in_video(video))
        all_frames = set(gt_frame_map[video].keys()) | set(pred_frame_map[video].keys())
        TP = FP = FN = TN = 0

        for frame in all_frames:
            gt_objs = gt_frame_map[video].get(frame, set())
            pred_objs = pred_frame_map[video].get(frame, set())

            for obj in valid_objects:
                in_gt = obj in gt_objs
                in_pred = obj in pred_objs

                if in_gt and in_pred:
                    TP += 1
                elif not in_gt and in_pred:
                    FP += 1
                elif in_gt and not in_pred:
                    FN += 1
                elif not in_gt and not in_pred:
                    TN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_by_video[video] = {
            "true_positives": TP,
            "false_positives": FP,
            "false_negatives": FN,
            "true_negatives": TN,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    if metrics_by_video:
        n = len(metrics_by_video)
        avg_metrics = {
            "avg_true_positives": sum(m["true_positives"] for m in metrics_by_video.values()) / n,
            "avg_false_positives": sum(m["false_positives"] for m in metrics_by_video.values()) / n,
            "avg_false_negatives": sum(m["false_negatives"] for m in metrics_by_video.values()) / n,
            "avg_true_negatives": sum(m["true_negatives"] for m in metrics_by_video.values()) / n,
            "avg_precision": sum(m["precision"] for m in metrics_by_video.values()) / n,
            "avg_recall": sum(m["recall"] for m in metrics_by_video.values()) / n,
            "avg_f1_score": sum(m["f1_score"] for m in metrics_by_video.values()) / n
        }
        print(avg_metrics)
        metrics_by_video["overall"] = avg_metrics

    return metrics_by_video
    
def calculate_fscore_video(results_json_path):
    with open(results_json_path, 'r') as f:
        results = json.load(f)

    # Group results by video
    video_predictions = defaultdict(list)
    for res in results:
        video_predictions[res['video']].append(res)

    metrics_by_video = {}
    for video, preds in video_predictions.items():
        gt_objects = set(gt_objects_in_video(video))
        predicted_objects = set()
        TP = FP = FN = 0
        for pred in preds:
            obj = pred["predicted_object"]
            predicted_objects.add(obj)
        FN = len(set(obj for obj in gt_objects if obj in valid_objects) - predicted_objects)
        TP = len(set(obj for obj in gt_objects if obj in valid_objects and obj in predicted_objects))

        FP = len(set(obj for obj in predicted_objects if obj in valid_objects and obj not in gt_objects))
        TN = len(set(valid_objects) - gt_objects - predicted_objects)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics_by_video[video] = {
            "true_positives": TP,
            "false_positives": FP,
            "false_negatives": FN,
            "true_negatives": TN,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    avg_f1 = sum(metrics["f1_score"] for metrics in metrics_by_video.values()) / len(metrics_by_video) if metrics_by_video else 0.0
    avg_recall = sum(metrics["recall"] for metrics in metrics_by_video.values()) / len(metrics_by_video) if metrics_by_video else 0.0
    avg_precision = sum(metrics["precision"] for metrics in metrics_by_video.values()) / len(metrics_by_video) if metrics_by_video else 0.0
    avg_TP = sum(metrics["true_positives"] for metrics in metrics_by_video.values()) / len(metrics_by_video) if metrics_by_video else 0.0
    avg_FP = sum(metrics["false_positives"] for metrics in metrics_by_video.values()) / len(metrics_by_video) if metrics_by_video else 0.0
    avg_FN = sum(metrics["false_negatives"] for metrics in metrics_by_video.values()) / len(metrics_by_video) if metrics_by_video else 0.0
    avg_TN = sum(metrics["true_negatives"] for metrics in metrics_by_video.values()) / len(metrics_by_video) if metrics_by_video else 0.0
    print('Average F1: ', avg_f1)
    print('Average Recall: ', avg_recall)
    print('Average Precision: ', avg_precision)
    print('Average True Positives: ', avg_TP)
    print('Average False Positives: ', avg_FP)
    print('AVERAGE False Negatives: ', avg_FN)
    print('AVERAGE True Negatives: ', avg_TN)
    return metrics_by_video


if __name__ == "__main__":
    # image_path = '/home/ubuntu/code/uas-ip-identification/back_end/pipeline/Object_Detection/testimage2.png'

    # FOR TESTING THRESHOLDS
    # video_folder = '/home/ubuntu/code/uas-ip-identification/testVid'
    threshold=0.7
    model = 'resnet'

    video_folder = '/home/ubuntu/code/marvel'
    # model = 'google'
    # threshold=0.65 # USED FOR RESNET
    # threshold=0.76 # USED FOR GOOGLE
    # threshold=0.7 # USED FOR VIT
    masked=False #ignore

    # res = main(video_folder, model, threshold, masked)
    calculate_fscore_frame_agg_by_video('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_data/results/e2e_results_vit_allVids.json', '/home/ubuntu/code/uas-ip-identification/testing/schema_object.json')

    # print('FRAME')
    # res = calculate_fscore_frame('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_data/results/e2e_results_vit_allVids.json', '/home/ubuntu/code/uas-ip-identification/testing/schema_object.json')
    # print("VIDEO")
    # res = calculate_fscore_video('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_data/results/e2e_results_vit_allVids.json')

    # r = get_actual_frame_counts_from_folder('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_data/frames')
    # print(r)

    # print(res)
    # print(res)
    # print('FRAME LEVEL')
    # print('VIT UNMASKED')
    # res = calculate_fscore_frame('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_results_vit_unmasked.json')
    # print('VIT MASKED')
    # res = calculate_fscore_frame('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_results_vit_masked.json')
    # print('RESNET UNMASKED')
    # res = calculate_fscore_frame('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_results_resnet_unmasked.json')
    # print('RESNET MASKED')
    # res = calculate_fscore_frame('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_results_resnet_masked.json')
    # print('----------------------------------------------------------------------------------------------------------------------------------------------------------------')
    # print('VIDEO LEVEL')
    # print('VIT UNMASKED')
    # res = calculate_fscore_video('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_results_vit_unmasked.json')
    # print('VIT MASKED')
    # res = calculate_fscore_video('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_results_vit_masked.json')
    # print('RESNET UNMASKED')
    # res = calculate_fscore_video('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_results_resnet_unmasked.json')
    # print('RESNET MASKED')
    # res = calculate_fscore_video('/home/ubuntu/code/uas-ip-identification/back_end/pipeline/e2e_results_resnet_masked.json')

