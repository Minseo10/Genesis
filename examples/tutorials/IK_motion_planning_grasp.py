import genesis as gs
import numpy as np

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
locknlock = scene.add_entity(
            morph=gs.morphs.URDF(
                file='../../locknlock.urdf',
                fixed=True,
                # merge_fixed_links=False,
                pos=(0.65, 0.2, 0.02)
            ),
            # vis_mode="collision",
            # visualize_contact=True,
        )

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)
########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# set control gains
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

end_effector = franka.get_link("hand")

# set physics parameter of the block
damping = cube.get_dofs_damping()
stiffness = cube.get_dofs_stiffness()
inertia = cube.get_links_inertial_mass()
invweight = cube.get_links_invweight()
armature = cube.get_dofs_armature()
print("damping: ", damping)
print("stiffness: ", stiffness)
print("inertia: ", inertia)
print("invweight: ", invweight)
print("armature: ", armature)

# set physics parameter of the block
damping = franka.get_dofs_damping()
stiffness = franka.get_dofs_stiffness()
inertia = franka.get_links_inertial_mass()
invweight = franka.get_links_invweight()
armature = franka.get_dofs_armature()
print("damping: ", damping)
print("stiffness: ", stiffness)
print("inertia: ", inertia)
print("invweight: ", invweight)
print("armature: ", armature)

# move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
# gripper open pos
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 2s duration
)
# draw the planned path
path_debug = scene.draw_debug_path(path, franka)

# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()

# remove the drawn path
scene.clear_debug_object(path_debug)

# allow robot to reach the last waypoint
for i in range(100):
    scene.step()

# reach
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.130]),
    quat=np.array([0, 1, 0, 0]),
)
print(qpos)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()

# grasp
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

for i in range(100):
    scene.step()

# lift
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),

    quat=np.array([0, 1, 0, 0]),
)
print(qpos)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(10000):
    scene.step()
