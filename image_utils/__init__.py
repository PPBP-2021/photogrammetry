import cv2
import plotly.express as px
import numpy as np


def show_img_grayscale(image: np.ndarray, title="") -> None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fig = px.imshow(image, color_continuous_scale="gray", title=title)
    fig.show()


def show_img(image: np.ndarray, title: str = "") -> None:
    px.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), title=title).show()
