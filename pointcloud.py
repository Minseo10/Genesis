import genesis as gs
import numpy as np
import open3d as o3d  # For creating and visualizing point clouds
import rospy  # ROS Python library
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2


class RealTimePointCloud:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics
        self.pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
        rospy.init_node('real_time_point_cloud_publisher', anonymous=True)

        # Open3D Visualizer Setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Real-Time Point Cloud", width=800, height=600)
        self.pointcloud = o3d.geometry.PointCloud()
        # self.vis.add_geometry(self.pointcloud)

    def generate_point_cloud(self, rgb, depth):
        """
        Generate a 3D point cloud from rgb and depth images.
        """
        h, w = depth.shape
        fx, fy, cx, cy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['cx'], self.intrinsics['cy']

        # Create a grid of pixel coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x.flatten()
        y = y.flatten()

        # Project depth into 3D space
        z = depth.flatten()
        x_3d = (x - cx) * z / fx
        y_3d = (y - cy) * z / fy

        # Mask out invalid points
        valid_mask = z > 0
        x_3d = x_3d[valid_mask]
        y_3d = y_3d[valid_mask]
        z_3d = z[valid_mask]
        rgb = rgb.reshape(-1, 3)[valid_mask]

        # Combine XYZ and RGB into a point cloud
        points = np.column_stack((x_3d, y_3d, z_3d, rgb))
        return points

    def publish_point_cloud(self, points):
        """
        Publish a point cloud to ROS topic.
        """
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_frame"

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.UINT8, 1),
            PointField('g', 13, PointField.UINT8, 1),
            PointField('b', 14, PointField.UINT8, 1),
        ]

        # Convert points to the correct format
        points_with_rgb = [
            (x, y, z, int(r), int(g), int(b))  # Ensure r, g, b are integers
            for x, y, z, r, g, b in points
        ]

        cloud_msg = point_cloud2.create_cloud(header, fields, points_with_rgb)
        self.pub.publish(cloud_msg)

    def update_visualization(self, points):
        """
        Update the Open3D visualization in real-time.
        """
        xyz = points[:, :3]
        rgb = points[:, 3:] / 255.0  # Normalize RGB to [0, 1]

        # Update the Open3D point cloud
        self.vis.add_geometry(self.pointcloud)
        self.pointcloud.points = o3d.utility.Vector3dVector(xyz)
        self.pointcloud.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.update_geometry(self.pointcloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self, rgb, depth):
        rate = rospy.Rate(30)  # Set to 30 FPS
        while not rospy.is_shutdown():
            # Generate point cloud
            points = self.generate_point_cloud(rgb, depth)

            # Publish point cloud
            self.publish_point_cloud(points)

            # Update visualization
            self.update_visualization(points)

            rate.sleep()

        self.vis.destroy_window()


if __name__ == '__main__':
    gs.init()

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(  # TODO: replace extrinsic with actual values
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
    cam.start_recording()

    # TODO: replace intrinsic with actual values
    cam_intrinsics = cam.intrinsics
    camera_intrinsics = {
        "fx": cam_intrinsics[0][0],  # Focal length in x
        "fy": cam_intrinsics[1][1],  # Focal length in y
        "cx": cam_intrinsics[0][2],  # Principal point x
        "cy": cam_intrinsics[1][2],  # Principal point y
    }

    # Run the real-time point cloud system
    point_cloud_system = RealTimePointCloud(camera_intrinsics)

    # Main simulation loop
    rate = rospy.Rate(100)  # Match FPS with ROS rate

    for i in range(10000):
        scene.step()
        cam.set_pose(
            pos=(3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
            lookat=(0, 0, 0.5),
        )
        rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True,
                                                      normal=True)

        # Generate point cloud and publish it
        points = point_cloud_system.generate_point_cloud(rgb, depth)
        point_cloud_system.publish_point_cloud(points)
        point_cloud_system.update_visualization(points)

        rate.sleep()

    # cam.stop_recording(save_to_filename="video.mp4", fps=60)
    point_cloud_system.vis.destroy_window()