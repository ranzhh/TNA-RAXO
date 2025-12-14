"""
Blender rendering pipeline for 3D objects.

This module provides a BlenderProcessor class for rendering 3D objects from single or multiple views.
Supports various 3D file formats (STL, PLY, OBJ, etc.) with automatic camera framing.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import bpy
import mathutils


class BlenderProcessor:
    """
    A processor for rendering 3D objects using Blender's Python API.

    Supports:
    - Multiple 3D file formats (STL, PLY, OBJ, FBX, etc.)
    - Automatic camera framing based on object size
    - Single and multi-view rendering
    - Configurable lighting and materials
    """

    # Supported file formats and their import operators
    SUPPORTED_FORMATS = {
        ".stl": "wm.stl_import",
        ".ply": "wm.ply_import",
        ".obj": "wm.obj_import",
        ".fbx": "import_scene.fbx",
        ".gltf": "import_scene.gltf",
        ".glb": "import_scene.gltf",
    }

    def __init__(
        self,
        resolution: int = 512,
        samples: int = 64,
        render_engine: str = "BLENDER_EEVEE_NEXT",
    ):
        """
        Initialize the Blender processor.

        Args:
            resolution: Render resolution (width and height in pixels)
            samples: Number of render samples for anti-aliasing
            render_engine: Blender render engine to use
        """
        self.resolution = resolution
        self.samples = samples
        self.render_engine = render_engine

        self.scene = None
        self.camera = None
        self.camera_target = None
        self.lights = []

    def render_single_view(
        self,
        obj_path: str,
        output_path: str,
        camera_distance: Optional[float] = None,
    ) -> str:
        """
        Render a single view of an object.

        Args:
            obj_path: Path to the 3D object file
            output_path: Path where the rendered image will be saved
            camera_distance: Optional fixed camera distance (auto-calculated if None)

        Returns:
            Path to the rendered image
        """
        # Set up the scene
        self._setup_scene()

        # Load the object
        obj = self._load_object(obj_path)

        # Adjust camera distance based on object size
        if camera_distance is None:
            self._auto_frame_camera(obj)
        else:
            self._set_camera_distance(camera_distance)

        # Render the scene
        return self._render_scene(output_path)

    def render_multi_view(
        self,
        obj_path: str,
        output_dir: str,
        n_views: int = 8,
        camera_distance: Optional[float] = None,
        elevation_min: float = -30.0,
        elevation_max: float = 30.0,
    ) -> List[str]:
        """
        Render an object from multiple views around the object.

        Args:
            obj_path: Path to the 3D object file
            output_dir: Directory where rendered images will be saved
            n_views: Number of views to render (evenly distributed around object)
            camera_distance: Optional fixed camera distance (auto-calculated if None)
            elevation_min: Minimum elevation angle in degrees (negative = below object)
            elevation_max: Maximum elevation angle in degrees (positive = above object)

        Returns:
            List of paths to rendered images
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Set up the scene
        self._setup_scene()

        # Load the object
        obj = self._load_object(obj_path)

        # Calculate camera distance once
        if camera_distance is None:
            distance = self._calculate_camera_distance(obj)
        else:
            distance = camera_distance

        # Render from multiple angles
        image_paths = []
        for i in range(n_views):
            angle = (2 * math.pi * i) / n_views

            # Vary elevation angle across views to get views from different heights
            # Distribute elevation angles evenly between min and max
            if n_views > 1:
                elevation_fraction = i / (n_views - 1)
            else:
                elevation_fraction = 0.5

            elevation_deg = elevation_min + (elevation_max - elevation_min) * elevation_fraction

            self._position_camera_at_angle(angle, distance, elevation_deg)

            output_path = os.path.join(output_dir, f"view_{i:03d}.png")
            self._render_scene(output_path)
            image_paths.append(output_path)
            print(
                f"Rendered view {i + 1}/{n_views}: azimuth={math.degrees(angle):.1f}°, elevation={elevation_deg:.1f}°"
            )

        return image_paths

    def _setup_scene(self) -> None:
        """Create a new Blender scene with camera, lighting, and render settings."""
        # Clear existing scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Get the scene
        self.scene = bpy.context.scene

        # Set render settings
        self.scene.render.engine = self.render_engine
        if self.render_engine == "BLENDER_EEVEE_NEXT":
            self.scene.eevee.taa_render_samples = self.samples
        elif self.render_engine == "CYCLES":
            self.scene.cycles.samples = self.samples

        self.scene.render.resolution_x = self.resolution
        self.scene.render.resolution_y = self.resolution
        self.scene.render.image_settings.file_format = "PNG"

        # Add camera
        bpy.ops.object.camera_add(location=(3, -3, 2))
        self.camera = bpy.context.object
        self.scene.camera = self.camera

        # Create target for camera tracking
        bpy.ops.object.empty_add(location=(0, 0, 0))
        self.camera_target = bpy.context.object
        self.camera_target.name = "CameraTarget"

        # Point camera at origin using constraint
        constraint = self.camera.constraints.new(type="TRACK_TO")
        constraint.target = self.camera_target
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"

        # Set up 3-point lighting
        self._setup_three_point_lighting()

        # Set up world background
        self._setup_world()

    def _setup_three_point_lighting(self) -> None:
        """Set up professional 3-point lighting (key, fill, back lights)."""
        # Key light (main light from front-top) - increased for black background
        bpy.ops.object.light_add(type="AREA", location=(2, -2, 3))
        key_light = bpy.context.object
        key_light.name = "KeyLight"
        key_light.data.energy = 150  # Increased from 50 for black background
        key_light.data.size = 2
        self._add_light_tracking(key_light)
        self.lights.append(key_light)

        # Fill light (softer light from side)
        bpy.ops.object.light_add(type="AREA", location=(-2, 1, 2))
        fill_light = bpy.context.object
        fill_light.name = "FillLight"
        fill_light.data.energy = 80  # Increased from 30
        fill_light.data.size = 2
        self._add_light_tracking(fill_light)
        self.lights.append(fill_light)

        # Back light (rim lighting)
        bpy.ops.object.light_add(type="AREA", location=(-1, 2, 1.5))
        back_light = bpy.context.object
        back_light.name = "BackLight"
        back_light.data.energy = 60  # Increased from 20
        back_light.data.size = 1.5
        self._add_light_tracking(back_light)
        self.lights.append(back_light)

    def _add_light_tracking(self, light: bpy.types.Object) -> None:
        """Add tracking constraint to light so it always points at target."""
        constraint = light.constraints.new(type="TRACK_TO")
        constraint.target = self.camera_target
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"

    def _setup_world(self) -> None:
        """Set up world background and ambient lighting (no floor/ground plane)."""
        world = bpy.data.worlds.new("World")
        self.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes["Background"]
        bg_node.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # Pure black background
        bg_node.inputs[1].default_value = 0.0  # No ambient light from background

        # Disable any ground/floor effects
        self.scene.render.film_transparent = False  # Solid background, no alpha

    def _load_object(self, obj_path: str) -> bpy.types.Object:
        """
        Load a 3D object file into the scene.

        Args:
            obj_path: Path to the object file

        Returns:
            The loaded Blender object

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        obj_path = Path(obj_path)

        # Check if file exists
        if not obj_path.exists():
            raise FileNotFoundError(f"Object file not found: {obj_path}")

        # Check if format is supported
        file_ext = obj_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            supported = ", ".join(self.SUPPORTED_FORMATS.keys())
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported formats: {supported}"
            )

        # Import the file based on its format
        import_operator = self.SUPPORTED_FORMATS[file_ext]

        try:
            # Get the operator and call it
            ops_parts = import_operator.split(".")
            ops_category = getattr(bpy.ops, ops_parts[0])
            ops_function = getattr(ops_category, ops_parts[1])
            ops_function(filepath=str(obj_path))
        except AttributeError:
            # Fallback for older Blender versions
            if file_ext == ".stl":
                bpy.ops.import_mesh.stl(filepath=str(obj_path))
            elif file_ext == ".ply":
                bpy.ops.import_mesh.ply(filepath=str(obj_path))
            elif file_ext == ".obj":
                bpy.ops.import_scene.obj(filepath=str(obj_path))
            else:
                raise

        # Get the imported object
        if not bpy.context.selected_objects:
            raise RuntimeError(f"Failed to import object from {obj_path}")

        obj = bpy.context.selected_objects[0]

        # Center the object at origin
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)

        # Add material to the object
        self._add_material_to_object(obj)

        return obj

    def _add_material_to_object(
        self,
        obj: bpy.types.Object,
        color: tuple = (0.8, 0.6, 0.4, 1.0),
    ) -> None:
        """
        Add a material to the object.

        Args:
            obj: The Blender object
            color: RGBA color tuple (values 0-1)
        """
        mat = bpy.data.materials.new(name="ObjectMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = 0.3
        bsdf.inputs["Roughness"].default_value = 0.5

        # Assign material to object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    def _calculate_camera_distance(self, obj: bpy.types.Object) -> float:
        """
        Calculate optimal camera distance based on object size.

        Args:
            obj: The Blender object to frame

        Returns:
            Optimal camera distance
        """
        # Get the object's bounding box in world coordinates
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

        # Calculate bounding box dimensions
        min_x = min(v.x for v in bbox_corners)
        max_x = max(v.x for v in bbox_corners)
        min_y = min(v.y for v in bbox_corners)
        max_y = max(v.y for v in bbox_corners)
        min_z = min(v.z for v in bbox_corners)
        max_z = max(v.z for v in bbox_corners)

        # Calculate dimensions
        width = max_x - min_x
        depth = max_y - min_y
        height = max_z - min_z
        max_dim = max(width, depth, height)

        print(
            f"Object dimensions: W={width:.3f}, D={depth:.3f}, H={height:.3f}, Max={max_dim:.3f}"
        )

        # Calculate camera distance based on object size and FOV
        camera_data = self.camera.data
        fov = camera_data.angle

        # Calculate distance needed to fit the object (with padding)
        padding_factor = 1.5  # 50% padding
        distance = (max_dim * padding_factor) / (2 * math.tan(fov / 2))

        # Ensure minimum distance
        distance = max(distance, 1.0)

        print(f"Calculated camera distance: {distance:.3f}")
        return distance

    def _auto_frame_camera(self, obj: bpy.types.Object) -> None:
        """
        Automatically position camera to frame the object.

        Args:
            obj: The Blender object to frame
        """
        distance = self._calculate_camera_distance(obj)
        self._set_camera_distance(distance)

    def _set_camera_distance(
        self,
        distance: float,
        angle: float = math.radians(45),
        height_ratio: float = 0.5,
    ) -> None:
        """
        Position camera at specified distance and angle.

        Args:
            distance: Distance from origin
            angle: Angle around Z-axis in radians
            height_ratio: Height relative to distance
        """
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        z = distance * height_ratio

        self.camera.location = (x, y, z)

    def _position_camera_at_angle(
        self,
        angle: float,
        distance: float,
        elevation_deg: float = 0.0,
    ) -> None:
        """
        Position camera at a specific angle around the object using spherical coordinates.

        Args:
            angle: Azimuth angle in radians around the Z-axis (0 = +X axis)
            distance: Distance from the origin
            elevation_deg: Elevation angle in degrees above/below horizon
                          (0 = horizon, +90 = directly above, -90 = directly below)
        """
        # Convert elevation from degrees to radians
        elevation_rad = math.radians(elevation_deg)

        # Calculate position using spherical coordinates
        # elevation_rad = 0 means on the horizon (XY plane)
        # elevation_rad > 0 means above the object
        # elevation_rad < 0 means below the object

        # Distance in XY plane
        r_xy = distance * math.cos(elevation_rad)

        x = r_xy * math.cos(angle)
        y = r_xy * math.sin(angle)
        z = distance * math.sin(elevation_rad)

        self.camera.location = (x, y, z)

    def _render_scene(self, file_path: str) -> str:
        """
        Render the current scene to a file.

        Args:
            file_path: Path where the rendered image will be saved

        Returns:
            The file path of the rendered image
        """
        self.scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered: {file_path}")
        return file_path


def check_image_not_black(image_path: str, threshold: float = 10.0) -> bool:
    """
    Check if an image is not completely black using Blender.

    Args:
        image_path: Path to the image file
        threshold: Minimum average pixel value (0-255) to consider image as not black

    Returns:
        True if image is not black, False otherwise
    """
    img = bpy.data.images.load(image_path)
    pixels = list(img.pixels)

    # Calculate average (RGBA format, values 0-1)
    num_pixels = len(pixels) // 4
    total = sum(
        0.299 * pixels[i * 4] + 0.587 * pixels[i * 4 + 1] + 0.114 * pixels[i * 4 + 2]
        for i in range(num_pixels)
    )

    avg_value = (total / num_pixels) * 255
    bpy.data.images.remove(img)

    print(f"Average pixel value: {avg_value:.2f}")
    return avg_value > threshold


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Render 3D objects using Blender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Render single view
            blender --background --python blender.py -- --input model.stl --output render.png
            
            # Render multiple views
            blender --background --python blender.py -- --input model.stl --output-dir renders/ --views 8
            
            # Use different format and resolution
            blender --background --python blender.py -- --input model.ply --output render.png --resolution 1024
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input 3D model file (STL, PLY, OBJ, etc.)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output image path (for single view rendering)",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        help="Output directory (for multi-view rendering)",
    )

    parser.add_argument(
        "--views",
        "-v",
        type=int,
        default=1,
        help="Number of views to render (1 for single view, >1 for multi-view)",
    )

    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=512,
        help="Render resolution (width and height in pixels)",
    )

    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=64,
        help="Number of render samples for anti-aliasing",
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Camera distance (auto-calculated if not specified)",
    )

    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        default="BLENDER_EEVEE_NEXT",
        choices=["BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"],
        help="Render engine to use",
    )

    parser.add_argument(
        "--check-output",
        action="store_true",
        help="Check if output image is not completely black",
    )

    # Parse arguments (skip Blender's arguments)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    # Validate arguments
    if args.views == 1 and not args.output:
        parser.error("--output is required for single view rendering")
    if args.views > 1 and not args.output_dir:
        parser.error("--output-dir is required for multi-view rendering")

    # Create processor
    processor = BlenderProcessor(
        resolution=args.resolution,
        samples=args.samples,
        render_engine=args.engine,
    )

    # Render
    if args.views == 1:
        # Single view
        print(f"Rendering single view of {args.input}")
        result_path = processor.render_single_view(
            args.input,
            args.output,
            camera_distance=args.distance,
        )

        if args.check_output:
            if check_image_not_black(result_path):
                print("✓ Success! The rendered image is not black.")
            else:
                print("✗ Warning: The rendered image appears to be mostly black.")
                sys.exit(1)
    else:
        # Multi-view
        print(f"Rendering {args.views} views of {args.input}")
        image_paths = processor.render_multi_view(
            args.input,
            args.output_dir,
            n_views=args.views,
            camera_distance=args.distance,
        )
        print(f"✓ Successfully rendered {len(image_paths)} views")

    print("Rendering complete!")


if __name__ == "__main__":
    main()
