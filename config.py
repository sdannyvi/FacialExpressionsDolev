from typing import Any


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def resolve_path(relative_path):
    """Get relative path, and returns absolute path."""
    absolute_path = PROJECT_ROOT / relative_path
    return absolute_path


def validate_image_paths(file_paths: list[str], csv_path):
    """
    file_paths: list of file paths
    csv_path: dataset path 
    Stops the run if any image path is missing.
    """
    for idx, file_path in enumerate(file_paths):
        absolute_path = resolve_path(file_path)
        if not absolute_path.exists():
            raise FileNotFoundError(f"{csv_path}: missing image at row {idx}: {absolute_path}")
    print(f"{csv_path}: all {len(file_paths)} image paths OK")





# FER Plus coded label
fer_plus_encode = {
    'contempt': 7,
    'neutral': 6,
    'sad': 4,
    'happy': 3,
    'surprise': 5,
    'angry': 0,
    'fear': 2,
    'disgust': 1
}