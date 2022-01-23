from pathlib import Path
import functools
import json


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
                (asset_path/config["left_image"],
                 asset_path/config["right_image"],
                 config)
            )

    return returns
