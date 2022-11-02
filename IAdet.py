import logging
import os
from pathlib import Path
import argparse

from PIL import Image
from gui import run_app

logger = logging.Logger(name="detectionloop")



def parse_and_scan_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The path to the directory containing the images to annotate. Defaults to current directory.", default=".")
    parser.add_argument("--width", help="The width of the canvas displaying the images.", type=int, default=500)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-A", "--all_extensions", action="store_true", help="Look for all extensions accepted by PIL. If ignored, restrict search to PNG and JPEG files (as defined in `IAdet.py` script).")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.warning(
            "Passed data directory is not a directory! Taking parent"
        )
        data_dir = data_dir.parent

    if args.all_extensions:
        exts = Image.registered_extensions()
        supported_extensions = {ex for ex, f in exts.items() if f in Image.OPEN}
    else:
        supported_extensions = {".png", ".jpg", ".jpeg"}
    filenames = []
    for extension in supported_extensions:
        if args.verbose:
            print(f"Checking for {extension} files...")
        filenames += list(data_dir.glob("**/*" + extension))
    filenames = sorted(filenames)
    if args.verbose:
        print("Done listing files!")
        print("-"*70)
    if args.reset:
        for filename in ["annotated_iadet.json", "to_annotate_iadet.json"]:
            if os.path.isfile(filename):
                os.remove(filename)
    return filenames, args.verbose, args.width


def main():
    filenames, verbose, canvas_width = parse_and_scan_dir()
    if verbose:
        print("Launching GUI...")
    run_app(filenames, canvas_width)
    print('Woohoo!')





if __name__ == "__main__":
    main()
