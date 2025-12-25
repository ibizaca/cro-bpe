from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path

class HuggingFaceDownloader:
    """
    A utility class to download datasets from Hugging Face Hub.
    """

    def __init__(self, dataset_name: str, split: str | None = None):
        """
        Initializes the downloader with the dataset name and split.

        Args:
            dataset_name: The name of the dataset on Hugging Face Hub.
            split: The split of the dataset to download (default is "train").
        """
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = load_dataset(dataset_name, split=split)

    def download(self, download_path: Path, image_column: str = "image"):
        """
        Downloads the images from the dataset to the specified path.

        Args:
            download_path: The local path where images will be downloaded.
            image_column: The column name in the dataset that contains image URLs or paths.
        """
        download_path.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(len(self.dataset)), desc="Downloading images"):
            item = self.dataset[idx]
            image = item[image_column]
            image_path = download_path / f"{idx}.jpg"
            image.save(image_path)
