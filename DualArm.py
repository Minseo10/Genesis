from mpmath import euler

import genesis as gs
import time
import numpy as np
from math import radians
import torch

# gs.init(backend=gs.gpu)


class DualArm:
    def __init__(self):
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
            #     # enable_self_collision=True,
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
            # visualize_contact=True,
            material=gs.materials.Rigid(coup_friction=1.0),
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

    # TODO: close gripper when pinch mode
    def close_gripper(self, robot, left=True):
        finger_dofs = self.left_fingers if left else self.right_fingers
        # robot.control_dofs_position(np.array([0.5, 0.5, -0.1]*3), finger_dofs)
        # robot.control_dofs_force(np.array([20, 20, 20]*3), finger_dofs)
        # if pinch:
        #     palm_dofs = self.left_palm if left else self.right_palm
        #     robot.control_dofs_position(np.array([-0.78, 0.78]), palm_dofs)

        for i in range(100):
            robot.control_dofs_position(np.array([0.2, 0.0, -0.2]*3), finger_dofs)
            self.scene.step()
            self.cam_0.render()

    def control_gripper(self, robot, left=True, max_iterations=1000, target_force=5.0, kp=100, ki=0.01, kd=10):
        finger_links = [32, 40, 46, 33, 41, 47, 34, 42, 48] if left else [36, 43, 49, 37, 44, 50, 38, 45, 51]
        finger_dofs = self.left_fingers if left else self.right_fingers
        iteration = 0

        # 초기화
        force_error_integral = torch.zeros(len(finger_links)).to('cuda:0')  # 적분 항 초기화
        previous_force_error = torch.zeros(len(finger_links)).to('cuda:0')  # 이전 힘 오차 저장

        while iteration < max_iterations:
            # 현재 힘 측정 (각 링크에 작용하는 힘/토크)
            links_force = robot.get_links_net_contact_force()  # (n_links, 3) 또는 (n_envs, n_links, 3)
            print("links force: ", links_force)
            finger_forces = torch.linalg.norm(links_force[finger_links], dim=-1)  # (n_fingers,)
            print("fingers forces size: ", finger_forces)

            # 힘 에러 계산
            force_error = target_force - finger_forces  # 목표 힘과의 차이

            # PID 제어: 힘 오차를 바탕으로 제어 신호 생성
            force_error_integral += force_error  # 적분 항
            force_error_derivative = force_error - previous_force_error  # 미분 항
            torque_signal = (
                    kp * force_error
                    + ki * force_error_integral
                    + kd * force_error_derivative
            )

            # 그리퍼에 토크 제어 신호 전달
            robot.control_dofs_force(torque_signal, finger_dofs)

            # 시뮬레이션 한 스텝 진행
            self.scene.step()

            # 이전 힘 오차 업데이트
            previous_force_error = force_error.clone()

            # 수렴 조건 체크
            if torch.all(torch.abs(force_error) < 0.01):  # 힘 에러가 충분히 작으면 종료
                print(f"Grasp stabilized in {iteration} iterations.")
                break

            iteration += 1

        if iteration == max_iterations:
            print("Failed to stabilize grasp within max iterations.")


    def pinch_gripper(self, robot, left=True):
        palm_dofs = self.left_palm if left else self.right_palm
        robot.control_dofs_position(np.array([-0.78, 0.78]), palm_dofs)  # 45 degree

        for i in range(100):
            self.scene.step()
            self.cam_0.render()

    def motion_planning(self, robot, target, left=True, holding=False, pose=True):
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

            qpos_start = joint_angle

            path = robot.plan_path(qpos_goal=qpos, qpos_start=joint_angle, planner="RRTConnect")
            # robot.control_dofs_position(qpos, arm_dofs)

            # if holding an object then use force to gripper
            # TODO: optimal force & which joints
            for waypoint in path:
                robot.control_dofs_position(waypoint)
                if holding:
                    robot.control_dofs_force(np.array([-20] * 11), gripper_dofs)
                self.scene.step()
                self.cam_0.render()
            # allow robot to reach the last waypoint
            for i in range(100):
                self.scene.step()
                self.cam_0.render()

        # target is joint angle radian
        else:
            path = robot.plan_path(target, planner="RRTConnect")
            for waypoint in path:
                robot.control_dofs_position(waypoint)
                if holding:
                    robot.control_dofs_force(np.array([20, 20, 20]), np.array([28, 29, 30]))
                self.scene.step()
                self.cam_0.render()

            # allow robot to reach the last waypoint
            for i in range(100):
                self.scene.step()
                self.cam_0.render()


def main():
    # Initialize
    gs.init(backend=gs.gpu)

    # Initialize DualArm
    dual_arm = DualArm()

    block = dual_arm.scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(-0.4, -0.5, 0.75),
        ),
        surface=gs.surfaces.Rough(
            color=(0.1, 0.6, 0.1),
        ),
    )


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

    dual_arm.motion_planning(dual_arm.robot, init_qpos, pose=False)

    # damping = dual_arm.robot.get_dofs_damping(dual_arm.left_arm)
    # stiffness = dual_arm.robot.get_dofs_stiffness(dual_arm.left_arm)
    # print("damping: ", damping)
    # print("stiffness: ", stiffness)
    # dual_arm.robot.set_dofs_stiffness(np.array([20.0] * 11), dual_arm.left_gripper)
    # dual_arm.robot.set_dofs_damping(np.array([20.0] * 11), dual_arm.left_gripper)

    # Pinch mode
    dual_arm.pinch_gripper(dual_arm.robot, True)

    print("robot base frame:", dual_arm.robot.get_pos(), dual_arm.robot.get_quat())
    print("world frame pose: ", dual_arm.robot.get_links_pos()[0], dual_arm.robot.get_links_quat()[0])
    print("stand frame pose: ", dual_arm.robot.get_links_pos()[1], dual_arm.robot.get_links_quat()[1])
    print("left ee current pose: ", dual_arm.robot.get_links_pos()[27], dual_arm.robot.get_links_quat()[27])
    print("right ee current pose: ", dual_arm.robot.get_links_pos()[31], dual_arm.robot.get_links_quat()[31])

    # Move to pre-grasp pose
    pre_grasp_pose = np.array([-0.4, -0.5, 0.83, 0.5, -0.5, 0.5, 0.5])
    dual_arm.motion_planning(dual_arm.robot, pre_grasp_pose, left=True, holding=False, pose=True)

    # Move closer
    reach = pre_grasp_pose.copy()
    reach[2] -= 0.10
    dual_arm.motion_planning(dual_arm.robot, reach, left=True, holding=False, pose=True)

    # Grasp
    dual_arm.close_gripper(dual_arm.robot, True)

    # Lift
    lift = pre_grasp_pose.copy()
    # dual_arm.motion_planning(dual_arm.robot, lift, left=True, holding=True, pose=True)
    end_effector = dual_arm.robot.get_link("left_gripper_tool0")
    joint_angle = dual_arm.robot.get_dofs_position()
    finger_pos = joint_angle[dual_arm.left_fingers]
    for i in range(250):
        qpos = dual_arm.robot.inverse_kinematics(
            link=end_effector,
            pos=np.array([lift[0], lift[1], lift[2]+0.10/250*i]),
            quat=lift[3:]
        )
        dual_arm.robot.control_dofs_position(qpos[dual_arm.left_arm], dual_arm.left_arm)
        dual_arm.robot.control_dofs_position(np.array([0.1, 0.0, -0.2] * 3), dual_arm.left_fingers)
        # dual_arm.robot.control_dofs_force(np.array([20, 20, 20] * 3), dual_arm.left_fingers)

        dual_arm.scene.step()
        dual_arm.cam_0.render()

    # Go to default pose (every joint angle is zero)
    # dual_arm.robot.control_dofs_position(np.array([0] *34), np.arange(34))

    for i in range(10000):
        dual_arm.robot.control_dofs_position(np.array([0.1, 0.0, -0.2] * 3), dual_arm.left_fingers)
        dual_arm.scene.step()
        dual_arm.cam_0.render()


if __name__ == "__main__":
    main()
