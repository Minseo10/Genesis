import genesis as gs
import numpy as np

########################## init ##########################
gs.init()

########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
        gravity=(0, 0, -9.8)
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(1280, 720),
        max_FPS=60,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

cloth = scene.add_entity(
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

robot = scene.add_entity(
            morph=gs.morphs.URDF(
                file='../../X7S/urdf/X7S.urdf',
                fixed=True,
                merge_fixed_links=False,
                pos=(0.0, 0.0, 0.1)
            ),
            # vis_mode="collision",
            # visualize_contact=True,
        )

# TODO: change size height
table = scene.add_entity(
    morph=gs.morphs.Mesh(
        file="../../genesis/assets/meshes/table/Desk_OBJ.obj",
        pos=(-0.8, 0.0, 0.58),  # height 0.7
        euler=(90, 0, 90),
        scale=(0.004, 0.0029, 0.0025),
        fixed=True,
    ),
    # vis_mode = "collision",
)


########################## build ##########################
scene.build()


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

    # 🌟 속도 초기화 (회전 방지)
    for i in range(len(particles)):
        cloth.set_particle_velocity(i, np.array([0, 0, 0]))

horizon = 100
for i in range(horizon):
    scene.step()

particles = cloth.get_particles()
for i in range(len(particles)):
    cloth.release_particle(i)

# 🌟 세 번 접기
fold_cloth(cloth, fold_axis="x")  # 가로 접기

for i in range(horizon):
    scene.step()

fold_cloth(cloth, fold_axis="y")  # 세로 접기

for i in range(horizon):
    scene.step()

fold_cloth(cloth, fold_axis="x")  # 다시 가로 접기

horizon = 1000
for i in range(horizon):
    scene.step()
