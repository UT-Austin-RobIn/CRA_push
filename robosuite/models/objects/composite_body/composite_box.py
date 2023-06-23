import numpy as np

from robosuite.models.objects import BoxObject, CompositeBodyObject, CylinderObject, BallObject
from robosuite.utils.mjcf_utils import BLUE, RED, CustomMaterial, array_to_string


class CompositeBoxObject(CompositeBodyObject):
    """
    An example object that demonstrates the CompositeBodyObject functionality. This object consists of two cube bodies
    joined together by a hinge joint.

    Args:
        name (str): Name of this object

        box1_size (3-array): (L, W, H) half-sizes for the first box

        box2_size (3-array): (L, W, H) half-sizes for the second box

        use_texture (bool): set True if using wood textures for the blocks
    """

    def __init__(
        self,
        name,
        box1_size=[0.14, 0.08, 0.1],
        use_texture=True,
    ):
        # Set box sizes
        self.box1_size = np.array(box1_size)

        # Set texture attributes
        self.use_texture = use_texture
        self.box1_material = None
        self.box2_material = None
        self.box1_rgba = RED
        self.box2_rgba = BLUE

        # Define materials we want to use for this object
        if self.use_texture:
            # Remove RGBAs
            self.box1_rgba = None
            self.box2_rgba = None

            # Set materials for each box
            tex_attrib = {
                "type": "cube",
            }
            mat_attrib = {
                "texrepeat": "3 3",
                "specular": "0.4",
                "shininess": "0.1",
            }
            self.box1_material = CustomMaterial(
                texture="WoodRed",
                tex_name="box1_tex",
                mat_name="box1_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )

        # Create objects
        objects = []
        objects.append(
            BoxObject(
                name=f"box{1}",
                size=self.box1_size,
                rgba=self.box1_rgba,
                material=self.box1_material,
                friction=[0.1, 0.005, 0.0001],
                density=10000,
            )
        )

        self.ball_size = [0.005]
        self.sphere_material = CustomMaterial(
                texture="PlasterPink",
                tex_name="sphere_tex",
                mat_name="sphere_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
        self.ball_rgba = [1,1,1,1]
        self.ball_rgba_2 = [1,0,0,1]

        for i in range(8):
            objects.append(
                BallObject(
                    name=f"ball{i}",
                    size=self.ball_size,
                    rgba=self.ball_rgba_2,
                    friction=[0.1, 0.005, 0.0001],
                    density=10000,
                    solref = [-1, -1],
                    solimp = [0, 0, 0]
                )
        )

        # Define positions -- second box should lie on top of first box with edge aligned at hinge joint
        # Hinge visualizer should be aligned at hinge joint location
        positions = [
            [0, 0, 0],  # First box is centered at top-level body anyways

            [-0.13, 0.07, 0.09],
            [-0.13, -0.07, 0.09],
            [-0.13, 0.07, -0.09],
            [-0.13, -0.07, -0.09],

            [0.13, 0.07, 0.09],
            [0.13, -0.07, 0.09],
            [0.13, 0.07, -0.09],
            [0.13, -0.07, -0.09],
        ]

        quats = [
            None,  # Default quaternion for box 1
        ]
        for i in range(8):
            quats.append(None)

        parents = [
            None,  # box 1 attached to top-level body
        ]

        for i in range(8):
            parents.append(objects[0].root_body)

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=positions,
            object_quats=quats,
            object_parents=parents,
            body_joints={},
        )
