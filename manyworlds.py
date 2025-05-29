import argparse
import numpy as np
import genesis as gs
import time
from math import radians
import torch


class SingleArm:
    def __init__(self):
        # Initialize a scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
                max_FPS=200,
            ),
            show_viewer=True,
            show_FPS = False,
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = True,
                plane_reflection = True,
                ambient_light    = (0.1, 0.1, 0.1),
            ),
        )

        # Define joints indices
        self.arm = np.arange(7)
        self.fingers = np.arange(7, 9)

        # Define control gains
        self.kp_arm = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000])
        self.kv_arm = np.array([450, 450, 350, 350, 200, 200, 200])
        self.kp_gripper = np.array([100] * 2)
        self.kv_gripper = np.array([10] * 2)
        self.force = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])

        # add default entities
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
            # visualize_contact=True,
        )

    def set_control_gains(self, robot):
        robot.set_dofs_kp(
            np.concatenate((self.kp_arm, self.kp_gripper)),
            np.concatenate((self.arm, self.fingers))
        )
        robot.set_dofs_kv(
            np.concatenate((self.kv_arm, self.kv_gripper)),
            np.concatenate((self.arm, self.fingers))
        )
        robot.set_dofs_force_range(
            -self.force,
            self.force,
            np.concatenate((self.arm, self.fingers))
        )

    def open_gripper(self, robot):
        robot.control_dofs_position(np.array([0.0] * 2), self.fingers)

        for i in range(100):
            self.scene.step()

    def close_gripper(self, robot):
        robot.control_dofs_position(np.array([-0.5,-0.5]), self.fingers)

        for i in range(100):
            self.scene.step()

    def motion_planning(self, robot, target, holding=False, pose=True):
        # target is end effector pose
        if pose:
            end_effector = robot.get_link("hand")
            qpos = robot.inverse_kinematics(
                link=end_effector,
                pos=target[:3],  # np array [x, y, z] (in meters)
                quat=target[3:],  # np array [x,y,z,w] (normalized quaternion)
            )
            qpos[self.arm] = qpos[self.arm]

            if holding:
                path = robot.plan_path(qpos_goal=qpos, planner="RRTConnect", ignore_collision=True)
            else:
                path = robot.plan_path(qpos_goal=qpos, planner="RRTConnect")

            for waypoint in path:
                robot.control_dofs_position(waypoint)
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
                self.scene.step()

            # allow robot to reach the last waypoint
            for i in range(100):
                self.scene.step()

    def motion_planning_sample(self, robot, target, holding=False, pose=True, num_samples=5):
        # target is end effector pose
        if pose:
            end_effector = robot.get_link("hand")
            qpos = robot.inverse_kinematics(
                link=end_effector,
                pos=target[:3],  # np array [x, y, z] (in meters)
                quat=target[3:],  # np array [x,y,z,w] (normalized quaternion)
            )
            qpos[self.arm] = qpos[self.arm]

            print("start sampling")
            if holding:
                paths = robot.sample_path(qpos_goal=qpos, planner="RRTConnect", ignore_collision=True, num_samples=num_samples)
            else:
                paths = robot.sample_path(qpos_goal=qpos, planner="RRTConnect", num_samples=num_samples)
            # print(paths)

        # target is joint angle radian
        else:
            if holding:
                paths = robot.sample_path(target, planner="RRTConnect", ignore_collision=True, num_samples=num_samples)
            else:
                paths = robot.sample_path(target, planner="RRTConnect", num_samples=num_samples)
            # print(paths)

        return paths

    def motion_path_vis(self, robot, paths):
        # TODO: paths의 각 path에 대해 end-effector pose의 trajectory를 계산 (forward kinematics)
        # 각 path는 tensor([-0.9371,  1.3369,  1.3841, -1.5457, -1.3416,  1.4469,  1.3112,  0.0101,0.0086], device='cuda:0') 이런 꼴, 맨 뒤의 값 두개는 finger의 config니까 end effector pose 구할때는 무시.


        # TODO: 각 end-effector trajectory 시각화


def main():
    # Initialize
    gs.init(backend=gs.gpu)
    franka = SingleArm()

    # additional objects
    cube = franka.scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.65, 0.0, 0.02),
        )
    )

    franka.scene.build()

    # Set control gains
    franka.set_control_gains(franka.robot)

    # move to pre-grasp pose
    pre_grasp_pose = np.array([0.65, 0.0, 0.24, 0, 1, 0, 0])
    franka.motion_planning_sample(franka.robot, pre_grasp_pose, holding=False, pose=True)

    '''
    # reach
    grasp_pose = pre_grasp_pose.copy()
    grasp_pose[2] -= 0.1
    franka.motion_planning(franka.robot, grasp_pose, holding=False, pose=True)

    # grasp
    franka.close_gripper(franka.robot)

    # lift
    lift_pose = grasp_pose.copy()
    lift_pose[2] += 0.15
    franka.motion_planning(franka.robot, lift_pose, holding=True, pose=True)

    for i in range(1000):
        franka.scene.step()
    '''


if __name__ == "__main__":
    main()
