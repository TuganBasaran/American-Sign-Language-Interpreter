import os
import cv2
import pickle
import numpy as np
import time
from hand_tracker import HandTracker

def process_data(data_dir, output_file):
    tracker = HandTracker(mode=True, max_hands=1)
    
    data = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    classes = sorted(os.listdir(data_dir))
    classes = [c for c in classes if not c.startswith('.')]
    
    print(f"Classes to process: {len(classes)}")
    print("Processing started... (This may take a while, please wait)")
    
    total_images = 0
    processed_count = 0
    success_count = 0
    t0 = time.time()
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        file_names = os.listdir(class_dir)
        total_images += len(file_names)
        
        print(f"Scanning class: {class_name} ({len(file_names)} images)")
        
        for file_name in file_names:
            img_path = os.path.join(class_dir, file_name)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            features = tracker.get_normalized_landmarks(img)
            
            processed_count += 1
            
            if features is not None:
                data.append(features)
                labels.append(class_name)
                success_count += 1
            
            if processed_count % 1000 == 0:
                elapsed = time.time() - t0
                print(f"Processed: {processed_count}, Successful: {success_count}, Time Elapsed: {elapsed:.1f}s")

    print(f"\nProcessing Complete!")
    print(f"Total files: {total_images}")
    print(f"Successfully extracted landmarks: {success_count}")
    print(f"Success rate: %{success_count/total_images*100:.1f}")
    
    print(f"Saving data: {output_file} ...")
    with open(output_file, 'wb') as f:
        pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)
    print("Save successful.")

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "archive", "asl_alphabet_train", "asl_alphabet_train")
    OUTPUT_FILE = "data/landmarks_dataset.pkl"
    
    if not os.path.exists(DATA_PATH):
         DATA_PATH = os.path.join("data", "archive", "asl_alphabet_train")

    process_data(DATA_PATH, OUTPUT_FILE)
