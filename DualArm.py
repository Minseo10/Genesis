from mpmath import euler

import genesis as gs
import time
import numpy as np
from math import radians
import torch

# gs.init(backend=gs.gpu)


class DualArm:
    def __init__(self, ray_tracer=False):
        # Initialize a scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-4, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=40,
                max_FPS=200,
            ),
            show_viewer=True,
            show_FPS = False,
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=4,  # for more stable grasping contact
            ),
            # rigid_options=gs.options.RigidOptions(
            #     dt=0.01,
            #     enable_collision=True,
            #     enable_self_collision=True,
            # ),
            vis_options = gs.options.VisOptions(
                show_world_frame = False, # visualize the coordinate frame of `world` at its origin
                world_frame_size = 1.0, # length of the world frame in meter
                show_link_frame  = False, # do not visualize coordinate frames of entity links
                show_cameras     = True, # do not visualize mesh and frustum of the cameras added
                plane_reflection = True, # turn on plane reflection
                ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
            ),
            # renderer = gs.renderers.Rasterizer(), # by default
            renderer=gs.renderers.RayTracer(  # type: ignore
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path="textures/indoor_bright.png",
                    ),
                ),
                env_radius=15.0,
                env_euler=(0, 0, 180),
                lights=[
                    {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (10.0, 10.0, 10.0)},
                ],
            ),
        )

        # Photo-realistic rendering
        self.ray_tracer = ray_tracer
        if self.ray_tracer:
            self.cam_0 = self.scene.add_camera(
                res=(1600, 900),
                pos=(-4, 0.0, 1.5),
                lookat=(0.0, 0.0, 0.8),
                fov=60,
                GUI=True,
                spp=512,
            )

        # Define joints indices
        self.left_arm = np.array([0, 2, 4, 6, 8, 10])
        self.right_arm = np.array([1, 3, 5, 7, 9, 11])
        self.left_fingers = np.array([16, 22, 28, 17, 23, 29, 18, 24, 30])
        self.right_fingers = np.array([19, 25, 31, 20, 26, 32, 21, 27, 33])
        self.left_palm = np.array([12, 13])
        self.right_palm = np.array([14, 15])

        self.left_gripper = np.concatenate((
            self.left_fingers, self.left_palm
        ))
        self.right_gripper = np.concatenate((
            self.right_fingers, self.right_palm
        ))

        # Define control gains
        # TODO: optimal control gains
        self.kp_arm = np.array([4500, 4500, 3500, 2000, 2000, 2000])
        # self.kd_arm = 2 * 7 * np.sqrt(self.kp_arm)
        self.kd_arm = np.array([450, 450, 350, 200, 200, 200])
        self.kp_gripper = np.array([100] * 11)
        # self.kd_gripper = 2 * 7 * np.sqrt(self.kp_gripper)
        self.kd_gripper = np.array([10] * 11)
        self.force_arm = np.concatenate([np.tile([87, 87, 87, 12, 12, 12], 2), [100] * 22])

        # add default entities (robot, plane, table, camera)
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file='/home/minseo/robot_ws/src/dual_ur_robotiq/dual_ur_robotiq_description/urdf/dual_ur_robotiq_genesis.urdf',
                fixed=True,
                merge_fixed_links=False,
            ),
            # vis_mode="collision",
            visualize_contact=True,
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="genesis/assets/meshes/table/Desk_OBJ.obj",
                pos=(-0.55, 0.0, 0.58),  # height 0.7
                euler=(90, 0, 90),
                scale=(0.004, 0.0029, 0.0025),
                fixed=True,
            ),
            # vis_mode = "collision",
        )


    def set_control_gains(self, robot):
        robot.set_dofs_kp(
            np.concatenate((self.kp_arm, self.kp_arm, self.kp_gripper, self.kp_gripper)),
            np.concatenate((self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )
        robot.set_dofs_kv(
            np.concatenate((self.kd_arm, self.kd_arm, self.kd_gripper, self.kd_gripper)),
            np.concatenate((self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )
        robot.set_dofs_force_range(
            -self.force_arm,
            self.force_arm,
            np.concatenate((self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )

    def open_gripper(self, robot, left=True):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        robot.control_dofs_position(np.array([0.0] * 11), gripper_dofs)

        for i in range(100):
            self.scene.step()

    # TODO: close gripper with pinch mode
    def close_gripper(self, robot, object, left=True):
        finger_dofs = self.left_fingers if left else self.right_fingers
        gripper_dofs = self.left_gripper if left else self.right_gripper
        # robot.control_dofs_position(np.array([0.55, 0.0, -0.5236, 0.55, 0.0, -0.5323, 0.55, 0.0, -0.5323, -0.15708, 0.15708]), gripper_dofs)
        robot.control_dofs_position(np.array([0.78, 0.0, -0.5, 0.78, 0.0, -0.5, 0.78, 0.0, -0.5, -0.15708, 0.15708]), gripper_dofs)

        # robot.control_dofs_force(np.array([2, 2, 2] * 3 + [-2, 2]), gripper_dofs)

        for i in range(100):
            contacts_info = robot.get_contacts(with_entity=object)
            print(contacts_info)
            self.scene.step()
            if self.ray_tracer:
                self.cam_0.render()

    def control_gripper_contact(self, robot, left=True, desired_force=10.0, pinch=True):
        finger_links = [32, 40, 46, 33, 41, 47, 34, 42, 48] if left else [36, 43, 49, 37, 44, 50, 38, 45, 51]
        finger_dofs = self.left_fingers if left else self.right_fingers
        dt = np.zeros(9)

        if pinch:
            while True:
                contact_force = robot.get_links_net_contact_force()[finger_links].cpu().numpy()
                print("net contact: ", contact_force)
                forces_size = np.linalg.norm(contact_force, axis=1)
                print('forces_size', forces_size)
                # Finger forces split
                finger_1_force = forces_size[:3]
                finger_2_force = forces_size[3:6]
                finger_middle_force = forces_size[6:9]

                # Helper function to decide control direction
                def update_dt(forces, dt_indices):
                    if np.all(forces < desired_force):
                        dt[dt_indices[0]] = 0.05  # Close
                        dt[dt_indices[1]] = 0.0
                        dt[dt_indices[2]] = -0.05  # Open
                    elif forces[0] >= desired_force and np.all(forces[1:] < desired_force):
                        dt[dt_indices[0]] = 0.0
                        dt[dt_indices[1]] = 0.05
                        dt[dt_indices[2]] = 0.0
                    elif forces[1] >= desired_force > forces[2]:
                        dt[dt_indices[0]] = 0.0
                        dt[dt_indices[1]] = 0.0
                        dt[dt_indices[2]] = 0.05
                    elif forces[2] >= desired_force:
                        dt[dt_indices[0]] = 0.0
                        dt[dt_indices[1]] = 0.0
                        dt[dt_indices[2]] = 0.05   # ???
                        return True  # Force threshold met
                    return False

                # Update control for each finger
                stop_finger_1 = update_dt(finger_1_force, [0, 1, 2])
                stop_finger_2 = update_dt(finger_2_force, [3, 4, 5])
                stop_finger_middle = update_dt(finger_middle_force, [6, 7, 8])

                # Break the loop if all fingers have sufficient contact force
                if stop_finger_1 and stop_finger_2 and stop_finger_middle:
                    break

                # Control joints
                qpos = robot.get_dofs_position(finger_dofs).cpu().numpy()
                new_qpos = qpos + dt
                robot.control_dofs_position(new_qpos, finger_dofs)

                # Step the simulation
                self.scene.step()


    def pinch_gripper(self, robot, left=True):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        robot.control_dofs_position(np.array([0.05, 0.0, -0.05, 0.05, 0.0, -0.05, 0.05, 0.0, -0.05, -0.15708, 0.15708]), gripper_dofs)

        for i in range(100):
            self.scene.step()
            if self.ray_tracer:
                self.cam_0.render()

    def motion_planning(self, robot, target, object, left=True, holding=False, pose=True):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        arm_dofs = self.left_arm if left else self.right_arm

        # target is end effector pose
        if pose:
            end_effector = robot.get_link("left_gripper_tool0") if left else robot.get_link("right_gripper_tool0")
            qpos = robot.inverse_kinematics(
                link=end_effector,
                pos=target[:3],  # np array [x, y, z] (in meters)
                quat=target[3:],  # np array [x,y,z,w] (normalized quaternion)
            )
            joint_angle = robot.get_dofs_position()
            qpos[arm_dofs] = qpos[arm_dofs]

            if holding:
                path = robot.plan_path(qpos_goal=qpos, qpos_start=joint_angle, planner="RRTConnect", ignore_collision=True)
            else:
                path = robot.plan_path(qpos_goal=qpos, qpos_start=joint_angle, planner="RRTConnect")

            # if holding an object then use force to gripper
            for waypoint in path:
                robot.control_dofs_position(waypoint)
                if holding:
                    contacts_info = robot.get_contacts(with_entity=object)
                    # print('lifting contacts_info', contacts_info)
                self.scene.step()
                if self.ray_tracer:
                    self.cam_0.render()
            # allow robot to reach the last waypoint
            for i in range(100):
                self.scene.step()
                if self.ray_tracer:
                    self.cam_0.render()

        # target is joint angle radian
        else:
            if holding:
                path = robot.plan_path(target, planner="RRTConnect", ignore_collision=True)
            else:
                path = robot.plan_path(target, planner="RRTConnect")

            for waypoint in path:
                robot.control_dofs_position(waypoint)
                if holding:
                    contacts_info = robot.get_contacts(with_entity=object)
                    # print('lifting contacts_info', contacts_info)
                self.scene.step()
                if self.ray_tracer:
                    self.cam_0.render()

            # allow robot to reach the last waypoint
            for i in range(100):
                self.scene.step()
                if self.ray_tracer:
                    self.cam_0.render()


def main():
    # Initialize
    gs.init(backend=gs.gpu)

    # photo-realistic rendering
    ray_tracer = False

    # Initialize DualArm
    dual_arm = DualArm(ray_tracer)

    block = dual_arm.scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.08, 0.08, 0.08),
            pos=(-0.4, -0.5, 0.75),
        ),
        surface=gs.surfaces.Rough(
            color=(0.1, 0.6, 0.1),
        ),
        # material=gs.materials.Rigid(friction=0.1),
    )

    # bottle = dual_arm.scene.add_entity(
    #     material=gs.materials.Rigid(rho=300),
    #     morph=gs.morphs.URDF(
    #         file="urdf/3763/mobility_vhacd.urdf",
    #         scale=0.15,
    #         pos=(-0.4, -0.5, 0.75),
    #         euler=(0, 90, 90),
    #     ),
    #     # visualize_contact=True,
    # )


    # Build scene
    dual_arm.scene.build()

    # Set control gains
    dual_arm.set_control_gains(dual_arm.robot)

    # Move to start pose
    left_arm_joints = np.array([radians(161), radians(-60), radians(90), radians(-145), radians(50), radians(-15)])
    right_arm_joints = np.array([radians(-161), radians(-120), radians(-90), radians(-35), radians(-50), radians(-75)])
    init_qpos = np.zeros(34)
    init_qpos[dual_arm.left_arm] = left_arm_joints
    init_qpos[dual_arm.right_arm] = right_arm_joints

    time.sleep(5)

    dual_arm.motion_planning(dual_arm.robot, init_qpos, block, pose=False)

    # Pinch mode
    dual_arm.pinch_gripper(dual_arm.robot, True)

    # Move to pre-grasp pose
    pre_grasp_pose = np.array([-0.4, -0.5, 0.81, 0.5, -0.5, 0.5, 0.5])
    dual_arm.motion_planning(dual_arm.robot, pre_grasp_pose, block, left=True, holding=False, pose=True)

    # Move closer
    reach = pre_grasp_pose.copy()
    reach[2] -= 0.10
    dual_arm.motion_planning(dual_arm.robot, reach, block, left=True, holding=False, pose=True)

    # Grasp
    # dual_arm.control_gripper_contact(dual_arm.robot, left=True, pinch=True)
    dual_arm.close_gripper(dual_arm.robot, block,True)

    time.sleep(10)
    # Lift
    lift = pre_grasp_pose.copy()
    lift[0] -= 0.10
    lift[1] += 0.10
    lift[2] += 0.05

    dual_arm.motion_planning(dual_arm.robot, lift, block, left=True, holding=True, pose=True)


    for i in range(10000):
        dual_arm.scene.step()
        if dual_arm.ray_tracer:
            dual_arm.cam_0.render()


if __name__ == "__main__":
    main()
