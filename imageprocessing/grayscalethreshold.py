from typing import Union
import cv2
import numpy as np
import plotly.express as px


def segmentate_grayscale(image: Union[np.ndarray, str], threshold: float) -> np.ndarray:
    if isinstance(image, str):
        image = cv2.imread(image)  # cv2 works with BGR instead of RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = px.imshow(image)
    fig.show()
