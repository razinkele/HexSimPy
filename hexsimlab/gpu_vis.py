raise NotImplementedError(
    "hexsimlab.gpu_vis is an incomplete prototype. "
    "The 'grid' variable is undefined. Not usable in its current form."
)

import vispy.scene

canvas = vispy.scene.SceneCanvas(keys="interactive")

view = canvas.central_widget.add_view()

image = vispy.scene.visuals.Image(grid)

view.add(image)

canvas.show()
