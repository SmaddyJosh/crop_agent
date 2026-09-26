import os
import shutil
from bing_image_downloader import downloader

classes = {
    "Bacterial_Spot": "Tomato Bacterial Spot leaf disease close up",
    "Early_Blight": "Tomato Early Blight leaf disease close up",
    "Healthy": "Tomato Healthy green leaf close up",
    "Late_blight": "Tomato Late Blight leaf disease close up",
    "Leaf_Mold": "Tomato Leaf Mold disease close up",
    "Target_Spot": "Tomato Target Spot leaf disease close up",
    "black_spot": "Rose black spot leaf disease close up"  # Black spot is common on roses, sometimes tomatoes
}

dataset_dir = "datasets/all_data"
download_dir = "downloads"

for class_name, query in classes.items():
    print(f"Downloading images for {class_name}...")
    
    # Download 40 images
    downloader.download(
        query, 
        limit=40,  
        output_dir=download_dir, 
        adult_filter_off=True, 
        force_replace=False, 
        timeout=10, 
        verbose=False
    )
    
    # Move images to the correct folder
    source_dir = os.path.join(download_dir, query)
    target_dir = os.path.join(dataset_dir, class_name)
    
    os.makedirs(target_dir, exist_ok=True)
    
    if os.path.exists(source_dir):
        for idx, filename in enumerate(os.listdir(source_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                src_path = os.path.join(source_dir, filename)
                # Rename to avoid collisions
                new_filename = f"web_{idx}_{filename}"
                dst_path = os.path.join(target_dir, new_filename)
                
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Error moving {filename}: {e}")

# Clean up
if os.path.exists(download_dir):
    shutil.rmtree(download_dir)

print("Finished downloading and adding images to the dataset!")
