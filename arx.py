import genesis as gs
import time
import numpy as np
from math import radians
import torch

# gs.init(backend=gs.gpu)

class ARX:
    def __init__(self):
        # Initialize a scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=40,
                max_FPS=200,
            ),
            show_viewer=True,
            show_FPS = False,
            sim_options=gs.options.SimOptions(
                dt=4e-3,
                substeps=10,
                gravity=(0, 0, -9.8)
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = False, # visualize the coordinate frame of `world` at its origin
                world_frame_size = 1.0, # length of the world frame in meter
                show_link_frame  = False, # do not visualize coordinate frames of entity links
                show_cameras     = True, # do not visualize mesh and frustum of the cameras added
                plane_reflection = True, # turn on plane reflection
                ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
            ),
        )

        # Define joints indices
        self.body = np.array([0, 1])
        self.head = np.array([2, 5])
        self.left_arm = np.array([3, 6, 8, 10, 12, 14, 16])
        self.right_arm = np.array([4, 7, 9, 11, 13, 15, 17])
        self.left_gripper = np.array([18, 19])
        self.right_gripper = np.array([20, 21])

        # Define control gains
        self.kp_body = np.array([3500, 3500])
        self.kd_body = np.array([350, 350])
        self.kp_head = np.array([4500, 4500])
        self.kd_head = np.array([450, 450])
        self.kp_arm = np.array([4500, 4500, 4500, 3500, 2000, 2000, 2000])
        self.kd_arm = np.array([450, 450, 450, 350, 200, 200, 200])
        self.kp_gripper = np.array([100, 100])
        self.kd_gripper = np.array([10, 10])
        self.force_rest = np.array([100] * 18)
        self.force_gripper = np.array([250] * 4)

        # add default entities (robot, plane, table)
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(),
        )

        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file='./X7S/urdf/X7S.urdf',
                fixed=True,
                merge_fixed_links=False,
                pos=(0.0, 0.0, 0.1)
            ),
        )

        self.table = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.762, 2.4384, 0.20),
                pos=(1.0, 0.0, 0.1),
            ),
        )

    def set_control_gains(self, robot):
        robot.set_dofs_kp(
            np.concatenate((self.kp_body, self.kp_head, self.kp_arm, self.kp_arm, self.kp_gripper, self.kp_gripper)),
            np.concatenate((self.body, self.head, self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )
        robot.set_dofs_kv(
            np.concatenate((self.kd_body, self.kd_head, self.kd_arm, self.kd_arm, self.kd_gripper, self.kd_gripper)),
            np.concatenate((self.body, self.head, self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )
        robot.set_dofs_force_range(
            -np.concatenate((self.force_rest, self.force_gripper)),
            np.concatenate((self.force_rest, self.force_gripper)),
            np.concatenate((self.body, self.head, self.left_arm, self.right_arm, self.left_gripper, self.right_gripper))
        )

    def open_gripper(self, robot, left=True, wide=True):
        """
        Parameters:
            robot: object
                The robot instance for which the motion planning is performed.
            left: bool, optional
                If using the left gripper, then True. If using the right gripper, then False.
            wide: bool, optional
                True then open the gripper in maximum width. False then open the gripper in suitable width.
        """
        gripper_dofs = self.left_gripper if left else self.right_gripper
        if wide:
            robot.control_dofs_position(np.array([0.025] * 2), gripper_dofs)
        else:
            robot.control_dofs_position(np.array([0.044] * 2), gripper_dofs)

        for i in range(100):
            self.scene.step()

    def close_gripper(self, robot, left=True):
        """
        Parameters:
            robot: object
                The robot instance for which the motion planning is performed.
            left: bool, optional
                If using the left gripper, then True. If using the right gripper, then False.
        """
        gripper_dofs = self.left_gripper if left else self.right_gripper
        robot.control_dofs_position(np.array([0.0] * 2), gripper_dofs)

        for i in range(100):
            self.scene.step()

    def motion_planning(self, robot, target, left=True, holding=False, pose=True):
        """
        Parameters:
            robot: object
                The robot instance for which the motion planning is performed.

            target: array-like
                If pose is True, it should be a 7-element array where the first 3 elements are the target position
                (in meters) and the next 4 elements are the target orientation as a
                normalized quaternion [w, x, y, z].
                If pose is False, it should be the target joint angles in radians.

            left: bool, optional
                If using the left arm, then True. If using the right arm, then False.

            holding: bool, optional
                If the robot is holding an object, True. Else False.

            pose: bool, optional
                If the target is a pose, True. Else False.
        """
        arm_dofs = self.left_arm if left else self.right_arm

        # target is end effector pose
        # TODO: change end effector
        if pose:
            end_effector = robot.get_link("link11") if left else robot.get_link("link20")
            qpos = robot.inverse_kinematics(
                link=end_effector,
                pos=target[:3],  # np array [x, y, z] (in meters)
                quat=target[3:],  # np array [w, x, y, z] (normalized quaternion)
                dofs_idx_local=arm_dofs,
            )
            qpos[arm_dofs] = qpos[arm_dofs]

            if holding:
                path = robot.plan_path(qpos_goal=qpos, planner="RRTConnect", ignore_collision=True)
            else:
                path = robot.plan_path(qpos_goal=qpos, planner="RRTConnect")

            # if holding an object then use force to gripper
            for waypoint in path:
                robot.control_dofs_position(waypoint)
                # if holding:
                #     contacts_info = robot.get_contacts(with_entity=object)
                self.scene.step()
            # allow robot to reach the last waypoint
            for i in range(100):
                self.scene.step()

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
                self.scene.step()

            # allow robot to reach the last waypoint
            for i in range(100):
                self.scene.step()

    def motion_planning_bimanual(self, robot, target_left, target_right):
        arm_dofs = np.concatenate((self.left_arm, self.right_arm))

        # target is end effector pose
        target_pos_left = target_left[:3]
        target_quat = target_left[3:]
        target_pos_right = target_right[:3]
        target_quat = target_right[3:]

        # TODO: change end effector
        left_ee = robot.get_link("link12")
        right_ee = robot.get_link("link21")

        qpos = robot.inverse_kinematics_multilink(
            links=[left_ee, right_ee],
            poss=[target_pos_left, target_pos_right],
            quats=[target_quat, target_quat],
            dofs_idx_local=arm_dofs,
        )

        path = robot.plan_path(qpos_goal=qpos, planner="RRTConnect")

        # if holding an object then use force to gripper
        for waypoint in path:
            robot.control_dofs_position(waypoint)
            self.scene.step()

        # allow robot to reach the last waypoint
        for i in range(100):
            self.scene.step()


def main():
    # Initialize
    gs.init(backend=gs.gpu)

    # Initialize ARX X7
    arx = ARX()

    # Build scene
    arx.scene.build()

    # Set control gains
    arx.set_control_gains(arx.robot)

    # Move to start pose
    body_joints = np.array([0.50, -0.00165])
    head_joints = np.array([0.0, 0.0])
    left_arm_joints = np.array([0.0, 0.0, 0.0, -0.00102, 0.0, 0.0, 0.0])
    right_arm_joints = np.array([0.0, 0.0, 0.0, 0.00102, 0.0, 0.0, 0.0])
    gripper_joints = np.array([0.00744, 0.00744])
    init_qpos = np.zeros(22)
    init_qpos[arx.body] = body_joints
    init_qpos[arx.head] = head_joints
    init_qpos[arx.left_arm] = left_arm_joints
    init_qpos[arx.right_arm] = right_arm_joints
    init_qpos[arx.left_gripper] = gripper_joints
    init_qpos[arx.right_gripper] = gripper_joints

    arx.motion_planning(arx.robot, init_qpos, holding=False, pose=False)

    # open the grippers
    arx.open_gripper(arx.robot, left=True, wide=False)
    arx.open_gripper(arx.robot, left=False, wide=False)

    # print the current end-effector pose (end-effector is temporary)
    print("left ee current pose: ", arx.robot.get_links_pos()[17], arx.robot.get_links_quat()[17])  # link11 (idx)
    print("right ee current pose: ", arx.robot.get_links_pos()[18], arx.robot.get_links_quat()[18])  # link20 (idx)

    # move arms

    target_left = np.array([-1.57, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0])  # modify this target pose
    target_right = np.array([0.5, -0.3, 0.7, 0.5, 0.5, 0.5, -0.5])  # modify this target pose

    init_qpos = np.zeros(22)
    init_qpos[arx.body] = body_joints
    init_qpos[arx.head] = head_joints
    init_qpos[arx.left_arm] = target_left
    init_qpos[arx.right_arm] = right_arm_joints
    init_qpos[arx.left_gripper] = gripper_joints
    init_qpos[arx.right_gripper] = gripper_joints

    # arx.motion_planning(arx.robot, init_qpos, left=True, holding=False, pose=False)
    arx.motion_planning(arx.robot, target_right, left=False, holding=False, pose=True)

    print("left ee current pose: ", arx.robot.get_links_pos()[17], arx.robot.get_links_quat()[17])  # link11 (idx)
    print("right ee current pose: ", arx.robot.get_links_pos()[18], arx.robot.get_links_quat()[18])  # link20 (idx)

    print("get right qpos:", arx.robot.get_dofs_position(dofs_idx_local=arx.right_arm))

    target_left = np.array([0.35, 0.15, 0.226, 1.0, 0.0, 0.0, 0.0])  # modify this target pose
    target_right = np.array([0.9, -0.3, 0.7, 1.0, 0.0, 0.0, 0.0])  # modify this target pose

    arx.motion_planning(arx.robot, target_left, left=True, holding=False, pose=True)
    # arx.motion_planning(arx.robot, target_right, left=False, holding=False, pose=True)

    # close the grippers
    # arx.close_gripper(arx.robot, left=False)
    # arx.close_gripper(arx.robot, left=True)

    for i in range(10000):
        arx.scene.step()


if __name__ == "__main__":
    main()
