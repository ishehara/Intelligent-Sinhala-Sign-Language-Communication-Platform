"""
Create a subset dataset with most common signs for initial training.
"""
from pathlib import Path
import shutil
import json

def create_subset_dataset(
    source_dir: str,
    target_dir: str,
    num_classes: int = 50,
    min_videos_per_class: int = 10
):
    """
    Create subset with most common signs.
    
    Args:
        source_dir: Original dataset directory
        target_dir: Where to create subset
        num_classes: Number of classes to include
        min_videos_per_class: Minimum videos required per class
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Count videos per class
    class_counts = {}
    for category in source_path.iterdir():
        if not category.is_dir():
            continue
        for sign_class in category.iterdir():
            if not sign_class.is_dir():
                continue
            videos = list(sign_class.glob('*.mp4'))
            if len(videos) >= min_videos_per_class:
                class_counts[sign_class] = len(videos)
    
    # Select top N classes
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    selected_classes = [cls[0] for cls in sorted_classes[:num_classes]]
    
    print(f"Selected {len(selected_classes)} classes with most videos:")
    
    # Copy selected classes
    for class_path in selected_classes:
        category = class_path.parent.name
        sign_name = class_path.name
        
        # Create target structure
        target_class_path = target_path / category / sign_name
        target_class_path.mkdir(parents=True, exist_ok=True)
        
        # Copy videos
        for video in class_path.glob('*.mp4'):
            shutil.copy2(video, target_class_path / video.name)
        
        print(f"  {category}/{sign_name}: {class_counts[class_path]} videos")
    
    print(f"\nSubset created at: {target_path}")
    print(f"Total classes: {len(selected_classes)}")
    print(f"Total videos: {sum(class_counts[cls] for cls in selected_classes)}")


if __name__ == "__main__":
    create_subset_dataset(
        source_dir="G:/research/Intelligent-Sinhala-Sign-Language-Communication-Platform/datasets/signVideo",
        target_dir="G:/research/Intelligent-Sinhala-Sign-Language-Communication-Platform/datasets/signVideo_subset50",
        num_classes=50,
        min_videos_per_class=10
    )
