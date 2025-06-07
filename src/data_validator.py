import json
import os
import shutil
from typing import List, Dict, Any, Union


def validate_data(ann_filepath: str, dont_backup: bool = False) -> bool:
    """
    Validate the ann.json file according to the specified requirements.

    Args:
        ann_filepath (str): Path to the ann.json file
        dont_backup (bool): If True, skip creating backup file

    Returns:
        bool: True if validation passes, False otherwise

    Raises:
        FileNotFoundError: If ann.json doesn't exist
        PermissionError: If ann.json is not writable
        ValueError: If validation fails
    """

    # Check if ann.json exists
    if not os.path.exists(ann_filepath):
        raise FileNotFoundError(f"ann.json file not found: {ann_filepath}")

    # Check if ann.json is writable
    if not os.access(ann_filepath, os.W_OK):
        raise PermissionError(f"ann.json file is not writable: {ann_filepath}")

    # Load the JSON data
    try:
        with open(ann_filepath, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {ann_filepath}: {e}")

    # Validate the data structure
    if not isinstance(data, list):
        raise ValueError("ann.json should contain a list of entries")

    # Check each entry
    for i, entry in enumerate(data):
        _validate_entry(entry, i)

    # Check for existence of all image files
    _check_image_files_exist(data)

    # Create backup if requested
    if not dont_backup:
        backup_filepath = "ann_backup.json"
        try:
            shutil.copy2(ann_filepath, backup_filepath)
        except Exception as e:
            raise ValueError(f"Failed to create backup file: {e}")

    return True


def _validate_entry(entry: Dict[str, Any], index: int) -> None:
    """
    Validate a single entry in the ann.json file.

    Args:
        entry: Dictionary representing one annotation entry
        index: Index of the entry for error reporting

    Raises:
        ValueError: If entry validation fails
    """

    # Check required fields
    required_fields = ["filepath", "pred_bboxes", "ann_bboxes", "state"]
    for field in required_fields:
        if field not in entry:
            raise ValueError(f"Entry {index}: Missing required field '{field}'")

    # Validate filepath
    if not isinstance(entry["filepath"], str):
        raise ValueError(f"Entry {index}: 'filepath' must be a string")

    # Validate pred_bboxes
    if not isinstance(entry["pred_bboxes"], list):
        raise ValueError(f"Entry {index}: 'pred_bboxes' must be a list")

    for j, bbox in enumerate(entry["pred_bboxes"]):
        _validate_bbox(bbox, f"Entry {index}, pred_bboxes[{j}]")

    # Validate ann_bboxes
    if not isinstance(entry["ann_bboxes"], list):
        raise ValueError(f"Entry {index}: 'ann_bboxes' must be a list")

    for j, bbox in enumerate(entry["ann_bboxes"]):
        _validate_bbox(bbox, f"Entry {index}, ann_bboxes[{j}]")

    # Validate state
    valid_states = ["unseen", "predicted", "annotated"]
    if entry["state"] not in valid_states:
        raise ValueError(
            f"Entry {index}: 'state' must be one of {valid_states}, got '{entry['state']}'"
        )


def _validate_bbox(bbox: List[int], context: str) -> None:
    """
    Validate a bounding box format.

    Args:
        bbox: Bounding box as [left, top, right, bottom]
        context: Context string for error reporting

    Raises:
        ValueError: If bbox validation fails
    """

    if not isinstance(bbox, list):
        raise ValueError(f"{context}: Bounding box must be a list")

    if len(bbox) != 4:
        raise ValueError(
            f"{context}: Bounding box must have exactly 4 values [left, top, right, bottom]"
        )

    for i, coord in enumerate(bbox):
        if not isinstance(coord, int):
            raise ValueError(f"{context}: Coordinate {i} must be an int")

        if coord < 0:
            raise ValueError(
                f"{context}: Coordinate {i} must be non-negative, got {coord}"
            )


def _check_image_files_exist(data: List[Dict[str, Any]]) -> None:
    """
    Check that all image files referenced in the data exist.

    Args:
        data: List of annotation entries

    Raises:
        FileNotFoundError: If any image file doesn't exist
    """

    missing_files = []
    for i, entry in enumerate(data):
        filepath = entry["filepath"]
        if not os.path.exists(filepath):
            missing_files.append(f"Entry {i}: {filepath}")

    if missing_files:
        raise FileNotFoundError(
            f"The following image files are missing:\n" + "\n".join(missing_files)
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate ann.json file")
    parser.add_argument("ann_file", help="Path to the ann.json file")
    parser.add_argument(
        "--dont_backup", action="store_true", help="Skip creating backup file"
    )

    args = parser.parse_args()

    try:
        validate_data(args.ann_file, args.dont_backup)
        print(f"✓ Validation passed for {args.ann_file}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        exit(1)
