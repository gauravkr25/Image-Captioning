from __future__ import annotations

import argparse
from pathlib import Path


def _print_manual_steps(destination: Path):
    print("No Kaggle API credentials were found.")
    print(f"Manually download the Flickr8k dataset to '{destination}' or attach it in a notebook.")
    print("Then ensure the folder contains an 'Images/' directory and either 'captions.txt' or 'Flickr8k.token.txt'.")
    print("Typical Kaggle dataset slug: 'adityajn105/flickr8k'")


def main():
    parser = argparse.ArgumentParser(description="Download or guide the user to the Flickr8k data bundle.")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Destination directory for the dataset.")
    args = parser.parse_args()

    destination = args.root.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"Kaggle credentials found at {kaggle_json}.")
        print(f"Run: kaggle datasets download -d adityajn105/flickr8k -p {destination} --unzip")
        print(f"Then verify the extracted images and captions under {destination}.")
    else:
        _print_manual_steps(destination)


if __name__ == "__main__":
    main()
