import genesis as gs
import time
import numpy as np
from math import radians
import torch

# gs.init(backend=gs.gpu)

def get_cloth_center(cloth):
    """천의 중심 좌표를 계산 (모든 입자의 평균)"""
    particles = cloth.get_particles()
    center_x = np.mean([p[0] for p in particles])
    center_y = np.mean([p[1] for p in particles])
    return center_x, center_y

def fold_cloth(cloth, fold_axis="x"):
    """옷을 중심을 기준으로 반으로 접는 함수"""
    particles = cloth.get_particles()
    center_x, center_y = get_cloth_center(cloth)

    if fold_axis == "x":
        fold_side = [p for p in particles if p[0] > center_x]  # 오른쪽 절반
        opposite_side = [p for p in particles if p[0] < center_x]  # 왼쪽 절반
    else:  # fold_axis == "y"
        fold_side = [p for p in particles if p[1] > center_y]  # 위쪽 절반
        opposite_side = [p for p in particles if p[1] < center_y]  # 아래쪽 절반

    if not fold_side or not opposite_side:
        return  # 접을 수 없는 경우 종료

    # 반대편의 가장 높은 z값 찾기
    max_z_opposite = max([p[2] for p in opposite_side])

    for i in range(len(particles)):
        x, y, z = particles[i]

        if fold_axis == "x" and x > center_x:
            # x축 기준으로 접기
            new_x = 2 * center_x - x
            new_z = max_z_opposite + 0.01
            cloth.set_particle_position(i, np.array([new_x, y, new_z]))

        elif fold_axis == "y" and y > center_y:
            # y축 기준으로 접기
            new_y = 2 * center_y - y
            new_z = max_z_opposite + 0.01
            cloth.set_particle_position(i, np.array([x, new_y, new_z]))

    for i in range(len(particles)):
        cloth.release_particle(i)

    # 회전 방지
    for i in range(len(particles)):
        cloth.set_particle_velocity(i, np.array([0, 0, 0]))

def move_cloth(cloth, move):
    particles = cloth.get_particles()
    for i in range(len(particles)):
        x, y, z = particles[i]
        new_x = x + move[0]
        new_y = y + move[1]
        new_z = z + move[2]
        cloth.set_particle_position(i, np.array([new_x, new_y, new_z]))



class ARX:
    def __init__(self, ray_tracer=False):
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
        # TODO: change for ARX X7
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
        # TODO: optimal control gains for ARX X7
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
                file='./X7S/urdf/X7S.urdf',
                fixed=True,
                merge_fixed_links=False,
                pos=(0.0, 0.0, 0.1)
            ),
            # vis_mode="collision",
            # visualize_contact=True,
        )

        # TODO: change size height
        self.table = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="genesis/assets/meshes/table/Desk_OBJ.obj",
                pos=(1.0, 0.0, 0.58),  # height 0.7
                euler=(90, 0, 90),
                scale=(0.004, 0.0029, 0.0025),
                fixed=True,
            ),
            # vis_mode = "collision",
        )

        self.cloth = self.scene.add_entity(
            material=gs.materials.PBD.Cloth(bending_relaxation=0.001),
            morph=gs.morphs.Mesh(
                file="meshes/cloth.obj",
                scale=(2.13, 1.52, 1),
                pos=(-2, 0, 0.1),  # if z is 0 then particles collide with the plane
                euler=(0.0, 0, 0.0),
            ),
            surface=gs.surfaces.Default(
                color=(0.2, 0.4, 0.8, 1.0),
                vis_mode="visual",
            ),
        )

    # TODO: change values for ARX X7
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

    # TODO: change values for ARX X7
    def open_gripper(self, robot, left=True):
        gripper_dofs = self.left_gripper if left else self.right_gripper
        robot.control_dofs_position(np.array([0.0] * 11), gripper_dofs)

        for i in range(100):
            self.scene.step()

    # TODO: change values for ARX X7
    def close_gripper(self, robot, object, left=True):
        finger_dofs = self.left_fingers if left else self.right_fingers
        gripper_dofs = self.left_gripper if left else self.right_gripper
        # robot.control_dofs_position(np.array([0.55, 0.0, -0.5236, 0.55, 0.0, -0.5323, 0.55, 0.0, -0.5323, -0.15708, 0.15708]), gripper_dofs)
        robot.control_dofs_position(np.array([0.78, 0.0, -0.5, 0.78, 0.0, -0.5, 0.78, 0.0, -0.5, -0.15708, 0.15708]), gripper_dofs)

        # robot.control_dofs_force(np.array([2, 2, 2] * 3 + [-2, 2]), gripper_dofs)

        for i in range(100):
            # contacts_info = robot.get_contacts(with_entity=object)
            # print(contacts_info)
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
                path = robot.plan_path(qpos_goal=qpos, planner="RRTConnect", ignore_collision=True)
            else:
                path = robot.plan_path(qpos_goal=qpos, planner="RRTConnect")

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
    arx = ARX(ray_tracer)

    # Build scene
    arx.scene.build()
    joints = arx.robot._joints
    links = arx.robot._links

    # Set control gains
    # arx.set_control_gains(arx.robot)

    for i in range(100):
        arx.scene.step()

    # fold the cloth 3 times
    particles = arx.cloth.get_particles()
    for i in range(len(particles)):
        arx.cloth.release_particle(i)

    for i in range(100):
        arx.scene.step()

    fold_cloth(arx.cloth, fold_axis="x")  # 가로 접기

    fold_cloth(arx.cloth, fold_axis="y")  # 세로 접기

    fold_cloth(arx.cloth, fold_axis="x")  # 다시 가로 접기


    # put cloth up on the table
    move_cloth(arx.cloth, move=(3.5, 0, 0.8))

    for i in range(100):
        arx.scene.step()

    for i in range(len(particles)):
        arx.cloth.release_particle(i)

    for i in range(100):
        arx.scene.step()

if __name__ == "__main__":
    main()
