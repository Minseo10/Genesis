import genesis as gs

gs.init()

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    # renderer=gs.renderers.RayTracer(),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

cam = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=True,
)

scene.build()

# render rgb, depth, segmentation, normal
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

# print("rgb:", rgb)
# print("depth: ", depth)
# print("segmentation: ", segmentation)
# print("normal: ", normal)

cam.start_recording()
import numpy as np

for i in range(10000): # how long? 몇 초?
    scene.step()
    cam.set_pose(
        pos=(3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat=(0, 0, 0.5),
    )
    rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True) # 이렇게 하면 depth 창에 영상처럼 나옴

# cam.stop_recording(save_to_filename="video.mp4", fps=60)
