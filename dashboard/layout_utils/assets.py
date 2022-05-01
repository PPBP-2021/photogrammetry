import functools
import json
from pathlib import Path


@functools.lru_cache()
def get_asset_images():
    """Get all images in the assets directory.

    Returns
    -------
    List[Tuple[pathlib.Path, pathlib.Path, dict]]
        List of Triples, containing the path for the left and right stereo image as well as the dict containing that stereo image pairs configs.
    """

    asset_path = Path("./dashboard/assets")
    configs = [p for p in asset_path.iterdir() if p.suffix == ".json"]
    returns = []
    for config in configs:
        with open(config) as f:
            config = json.load(f)
            returns.append(
                (
                    asset_path / config["left_image"],
                    asset_path / config["right_image"],
                    config,
                )
            )

    return returns


@functools.lru_cache(maxsize=8)
def get_asset_image_dict(image_path: str):
    """Get the image dict for the given image path.

    Parameters
    ----------
    image_path : str
        The name of the left image, given from the image button.

    Returns
    -------
    dict
        The image dict.
    """
    for image in get_asset_images():
        if image_path in str(image[0]):
            # Update the image dict with the real image path
            image[2]["left_image"] = str(image[0])
            image[2]["right_image"] = str(image[1])
            return image[2]

    raise ValueError(f"No image found for {image_path}")
