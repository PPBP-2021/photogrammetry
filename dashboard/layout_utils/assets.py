import functools
import json
from pathlib import Path


@functools.lru_cache()
def get_asset_images_segmentate():
    """Get all images in the assets directory.

    Returns
    -------
    List[Tuple[pathlib.Path, dict]]
        List of Tuples, containing the path for the image and the dict with the image's path.
    """

    asset_path = Path("./dashboard/assets")
    configs = [p for p in asset_path.iterdir() if str(p).endswith("segmentate.json")]
    returns = []
    for config in configs:
        with open(config) as f:
            config = json.load(f)
            returns.append((asset_path / config["image"], config))

    return returns


@functools.lru_cache(maxsize=8)
def get_asset_image_dict_segmentate(image_path: str):
    """Get the image dict for the given image path.

    Parameters
    ----------
    image_path : str
        The name of the image, given from the image button.

    Returns
    -------
    dict
        The image dict.
    """
    for image in get_asset_images_segmentate():
        if image_path in str(image[0]):
            # Update the image dict with the real image path
            image[1]["image"] = str(image[0])
            return image[1]

    raise ValueError(f"No image found for {image_path}")


@functools.lru_cache()
def get_asset_images_stereo():
    """Get all images in the assets directory.

    Returns
    -------
    List[Tuple[pathlib.Path, pathlib.Path, dict]]
        List of Triples, containing the path for the left and right stereo image as well as the dict containing that stereo image pairs configs.
    """

    asset_path = Path("./dashboard/assets")
    configs = [p for p in asset_path.iterdir() if str(p).endswith("stereo.json")]
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
def get_asset_image_dict_stereo(image_path: str):
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
    for image in get_asset_images_stereo():
        if image_path in str(image[0]):
            # Update the image dict with the real image path
            image[2]["left_image"] = str(image[0])
            image[2]["right_image"] = str(image[1])
            return image[2]

    raise ValueError(f"No image found for {image_path}")


@functools.lru_cache()
def get_asset_images_final_model():
    """Get all images in the assets directory.

    Returns
    -------
    List[Tuple[pathlib.Path, pathlib.Path, dict]]
        List of Triples, containing the path for the left and right stereo image as well as the dict containing that stereo image pairs configs.
    """

    asset_path = Path("./dashboard/assets")
    configs = [p for p in asset_path.iterdir() if str(p).endswith("3D.json")]
    returns = []
    for config in configs:
        with open(config) as f:
            config = json.load(f)
            returns.append(
                (
                    asset_path / config["front_image_pair"][0],
                    asset_path / config["front_image_pair"][1],
                    asset_path / config["back_image_pair"][0],
                    asset_path / config["back_image_pair"][1],
                    asset_path / config["left_image_pair"][0],
                    asset_path / config["left_image_pair"][1],
                    asset_path / config["right_image_pair"][0],
                    asset_path / config["right_image_pair"][1],
                    config,
                )
            )

    return returns


@functools.lru_cache(maxsize=8)
def get_asset_image_dict_final_model(image_path: str):
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
    for image in get_asset_images_final_model():
        if image_path in str(image[0]):
            # Update the image dict with the real image path

            image[8]["front_image_pair"] = [str(image[0]), str(image[1])]
            image[8]["back_image_pair"] = [str(image[2]), str(image[3])]
            image[8]["left_image_pair"] = [str(image[4]), str(image[5])]
            image[8]["right_image_pair"] = [str(image[6]), str(image[7])]
            return image[8]

    raise ValueError(f"No image found for {image_path}")
