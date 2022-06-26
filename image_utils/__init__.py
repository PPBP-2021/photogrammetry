import cv2
import numpy as np
import open3d
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go


def show_img_grayscale(image: np.ndarray, title="") -> None:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fig = px.imshow(image, color_continuous_scale="gray", title=title)
    fig.show()


def show_img(image: np.ndarray, title: str = "") -> None:
    px.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), title=title).show()


def triangle_mesh_to_fig(mesh: open3d.geometry.TriangleMesh) -> go.Figure:
    """Takes a open3d TriangleMesh and returns a plotly Mesh3d Figure.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The open3d Mesh to convert to a Plotly Mesh3d Figure

    Returns
    -------
    go.Figure
        The final figure, can be shown using dash.Graph
    """

    R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, np.pi))
    mesh.rotate(R, center=(0, 0, 0))

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    # split the array into x, y, z coordinates
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # split into the i, j, k vert indices
    i = tris[:, 0]
    j = tris[:, 1]
    k = tris[:, 2]

    colorscale = [[0, "black"], [1, "white"]]
    mesh3D = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        flatshading=True,
        colorscale=px.colors.sequential.Viridis,
        intensity=y,
        showscale=True,
    )

    layout = go.Layout(
        paper_bgcolor="rgb(1,1,1)", font_color="white", scene=dict(aspectmode="auto")
    )
    fig = go.Figure(data=[mesh3D], layout=layout)
    fig.data[0].update(
        lighting=dict(
            ambient=0.3,
            diffuse=1,
            fresnel=0.1,
            specular=1,
            roughness=0.4,
            facenormalsepsilon=0,
        )
    )
    return fig


def show_point_cloud(point_cloud: open3d.geometry.PointCloud):
    # pandas data frame for the scatter plot
    points = np.asarray(point_cloud.points)
    frm = pd.DataFrame(data=points, columns=["X", "Y", "Z"])

    pc_fig = px.scatter_3d(
        frm,
        x="X",
        y="Z",
        z="Y",
        color="Z",
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    pc_fig.update_traces(marker_size=1)

    pc_fig.show()
