import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import stl


def show_img_grayscale(image: np.ndarray, title="") -> None:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fig = px.imshow(image, color_continuous_scale="gray", title=title)
    fig.show()


def show_img(image: np.ndarray, title: str = "") -> None:
    px.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), title=title).show()


def show_stl(mesh: stl.mesh.Mesh, title: str = "") -> None:
    def stl2mesh3d(stl_mesh):
        # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
        # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
        p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
        # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
        # extract unique vertices from all mesh triangles
        vertices, ixr = np.unique(stl_mesh.vectors.reshape(
            p*q, r), return_inverse=True, axis=0)
        I = np.take(ixr, [3*k for k in range(p)])
        J = np.take(ixr, [3*k+1 for k in range(p)])
        K = np.take(ixr, [3*k+2 for k in range(p)])
        return vertices, I, J, K

    vertices, I, J, K = stl2mesh3d(mesh)
    x, y, z = vertices.T

    colorscale = [[0, '#e5dee5'], [1, '#e5dee5']]
    mesh3D = go.Mesh3d(x=x,
                       y=y,
                       z=z,
                       i=I,
                       j=J,
                       k=K,
                       flatshading=True,
                       colorscale=colorscale,
                       intensity=z,
                       name=title,
                       showscale=False)

    layout = go.Layout(paper_bgcolor='rgb(1,1,1)',
                       title_text=title, title_x=0.5,
                       font_color='white',
                       scene_camera=dict(eye=dict(x=1.25, y=-1.25, z=1)),
                       scene_xaxis_visible=False,
                       scene_yaxis_visible=False,
                       scene_zaxis_visible=False,
                       scene=dict(
                           aspectmode='data'
                       ))
    fig = go.Figure(data=[mesh3D], layout=layout)
    fig.data[0].update(lighting=dict(ambient=0.18,
                                     diffuse=1,
                                     fresnel=.1,
                                     specular=1,
                                     roughness=.1,
                                     facenormalsepsilon=0))
    fig.show()
