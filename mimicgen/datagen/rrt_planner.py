import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# -------------------------------------------------------------------
#  CONFIGURABLE CONSTANTS
# -------------------------------------------------------------------
MAX_ITER = 10000  # Max number of RRT iterations
STEP_SIZE = 0.1  # Distance to move from a node toward the sample
GOAL_THRESHOLD = (
    0.5  # When we're "close enough" to the goal in Euclidean distance
)
ANGLE_THRESHOLD = 1.3  # When orientation is "close enough"
X_MIN, X_MAX = 0, 5  # Bounds for x
Y_MIN, Y_MAX = -6, 0  # Bounds for y
GOAL_SAMPLE_PROB = 0.1
# For theta, we might assume free rotation in [-pi, pi].


class Node:
    """
    A node in the RRT tree.
    """

    def __init__(self, x, y, theta, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent  # Index of the parent node in the tree


def distance_pose(pose1, pose2):
    """
    Euclidean distance in (x, y) plus an angular difference component.
    This is one possible distance metric. Adjust if needed.
    """
    dx = pose1[0] - pose2[0]
    dy = pose1[1] - pose2[1]
    # For rotation, you could do a wrap-around difference if needed.
    dtheta = abs(pose1[2] - pose2[2])
    return math.sqrt(dx * dx + dy * dy) + dtheta


def sample_random_pose():
    """
    Sample a random (x, y, theta) within some bounding box / angle range.
    """
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    theta = random.uniform(-math.pi, math.pi)
    return (x, y, theta)


def angle_wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def steer(from_pose, to_pose, step_size=STEP_SIZE):
    """
    Move from 'from_pose' toward 'to_pose' by at most step_size in linear space
    and proportionally in rotation. Returns the "intermediate" pose.
    """
    x0, y0, th0 = from_pose
    x1, y1, th1 = to_pose

    dx = x1 - x0
    dy = y1 - y0
    dist_xy = math.sqrt(dx * dx + dy * dy)

    if dist_xy < 1e-6:
        dist_xy = 0.0

    # Fraction of the distance to move
    frac = step_size / dist_xy if dist_xy > step_size else 1.0
    # Interpolate orientation linearly
    dtheta = th1 - th0

    new_x = x0 + frac * dx
    new_y = y0 + frac * dy

    # Interpolate orientation over the shortest angular distance
    dtheta = angle_wrap(th1 - th0)
    new_theta = angle_wrap(th0 + frac * dtheta)

    return (new_x, new_y, new_theta)


def rrt_planner(k_col, start, goal, step_size=STEP_SIZE, draw=False):
    """
    Builds an RRT from 'start' to 'goal', each is (x, y, theta).
    'k_col' is the collision-checking object.
    Returns a list of (x, y, theta) if successful, or None if not.
    """
    # Initialize tree with the start pose
    tree = [Node(start[0], start[1], start[2], parent=None)]
    best_dist = float("inf")
    best_idx = 0
    for iteration in range(MAX_ITER):
        # 1. Sample a random pose
        if np.random.rand() < GOAL_SAMPLE_PROB:
            rand_pose = goal
        else:
            rand_pose = sample_random_pose()

        # 2. Find nearest node in the tree
        nearest_idx = None
        min_dist = float("inf")
        for i, node in enumerate(tree):
            da = distance_pose((node.x, node.y, node.theta), rand_pose)

            if da < min_dist:
                min_dist = da
                nearest_idx = i

        nearest_node = tree[nearest_idx]
        nearest_pose = (nearest_node.x, nearest_node.y, nearest_node.theta)

        # 3. Steer toward that random pose
        new_pose = steer(nearest_pose, rand_pose, step_size)

        # 4. Check collision from nearest_pose -> new_pose
        if check_path_collision(k_col, nearest_pose, new_pose):
            # It's safe to add new node
            new_node = Node(*new_pose, parent=nearest_idx)
            tree.append(new_node)

            d_goal = distance_pose(new_pose, goal)
            if d_goal < best_dist:
                best_dist = d_goal
                best_idx = len(tree) - 1

            # 5. Check if we reached (or are close to) the goal
            if (
                np.linalg.norm(np.array(new_pose[:2]) - np.array(goal[:2]))
                < GOAL_THRESHOLD
            ):
                # Check orientation closeness if it matters
                if abs(angle_wrap(new_pose[2] - goal[2])) < ANGLE_THRESHOLD:

                    path = build_solution_path(tree, len(tree) - 1, goal)

                    a = time.time()
                    path_short = shortcut_path(path, k_col, n_iterations=200, steps=20)
                    print("shortcut time", time.time() - a)
                    print(
                        f"Reached goal in {iteration} iterations! Length of path {len(path)}...shortened to {len(path_short)}"
                    )
                    if draw:
                        draw_rrt(tree, start, goal, path=path_short, title="RRT")
                    return path_short

        if draw:
            draw_rrt(tree, start, goal, title="RRT", save_path=f"rrt_iter{iteration}.pdf")

    path = build_solution_path(
        tree, best_idx, (tree[best_idx].x, tree[best_idx].y, tree[best_idx].theta)
    )

    # "Force" the last waypoint to be the actual goal
    # NOTE: The step from path[-2] to path[-1] is not guaranteed collision-free.
    path[-1] = goal

    # Optionally do shortcut:
    path_short = shortcut_path(path, k_col, n_iterations=200, steps=20)

    # If we exit the loop, no path found
    print(
        "RRT failed to find a path within max iterations but trying best path anyways len:",
        len(path_short),
    )
    if draw:
        draw_rrt(tree, start, goal, path=path_short, title="RRT")
    # breakpoint()

    return path_short


def draw_rrt(tree, start, goal, path=None, title="RRT", save_path="my_rrt_plot.png"):
    """
    Draws (and saves) the RRT tree in 2D, ignoring orientation.

    Arguments:
      - tree:    list of Node objects
      - start:   (x, y, theta)
      - goal:    (x, y, theta)
      - path:    list of (x, y, theta) for the solution path (if found)
      - title:   string, plot title
      - save_path: file path or name to save the figure (PNG, PDF, etc.)
    """
    plt.figure(figsize=(8, 8))
    plt.title(title)

    # Plot the tree
    for i, node in enumerate(tree):
        # Node
        plt.plot(node.x, node.y, "ro", markersize=2)
        # Edge to parent
        if node.parent is not None:
            parent = tree[node.parent]
            plt.plot([node.x, parent.x], [node.y, parent.y], "k-", linewidth=0.5)

    # Mark start / goal
    plt.plot(start[0], start[1], "gs", markersize=8, label="Start")
    plt.plot(goal[0], goal[1], "bs", markersize=8, label="Goal")

    # If path exists, plot it in green
    if path is not None and len(path) > 1:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, "g-", linewidth=2, label="Path")
        plt.plot(path_x, path_y, "go", markersize=4)

    plt.xlim(X_MIN - 1, X_MAX + 1)
    plt.ylim(Y_MIN - 1, Y_MAX + 1)
    plt.gca().set_aspect("equal", "box")
    plt.legend()

    # Save the figure (no plt.show() so it doesn't pop up)
    plt.savefig(save_path, dpi=300)
    plt.close()  # close the figure so it doesn't remain in memory



def shortcut_path(path, k_col, n_iterations=100, steps=20):
    """
    Iterative shortcutting on an (x, y, theta) path.

    Args:
        path (list): A list of (x, y, theta) waypoints.
        k_col (object): Your collision checker, used by check_path_collision.
        n_iterations (int): How many attempts to try random shortcuts.
        steps (int): Number of interpolation steps when checking collision between two waypoints.

    Returns:
        list: A shortened/smoothed path.
    """
    if len(path) < 3:
        return path  # Nothing to shortcut if only start/goal or so

    path = list(path)  # Make a copy if needed

    for _ in range(n_iterations):
        # Pick two random indices i < j in the path
        i = random.randint(0, len(path) - 2)
        j = random.randint(i + 1, len(path) - 1)

        # If they're neighbors, no sense shortcutting
        if j <= i + 1:
            continue

        # Check collision from path[i] directly to path[j]
        if check_path_collision(k_col, path[i], path[j], steps=steps):
            # If collision-free, remove the in-between waypoints
            new_path = path[: i + 1] + path[j:]
            path = new_path

    return path


def build_solution_path(tree, goal_idx, goal_pose):
    """
    Reconstruct the path from the tree by walking from the goal node
    back to the start, then reversing the list.
    """
    path = []
    current_idx = goal_idx
    while current_idx is not None:
        node = tree[current_idx]
        path.append((node.x, node.y, node.theta))
        current_idx = node.parent

    path.reverse()
    # Optionally snap to the exact goal pose orientation if needed:
    path[-1] = goal_pose
    return path


def is_collision_free(k_col, poses):
    """
    Checks collision by calling your collision function k_col.compute_score.
    'poses' is an Nx3 array of [x, y, theta].
    We'll assume the function returns 0 for collision-free and >0 for collisions,
    or something similar. Adjust the check as needed.
    """
    poses_arr = np.array(poses).reshape(-1, 3)
    scores = ~k_col.compute_score(
        poses_arr, numpy=True
    )
    # If the function returns an array, we consider ANY score > 0 to be collision.
    return not np.any(scores > 0)


def check_path_collision(k_col, start_pose, end_pose, steps=10):
    """
    Check collision along the line (and interpolated angles)
    from start_pose to end_pose by discretizing into 'steps'.
    """
    path_poses = []
    for i in range(steps + 1):
        frac = i / float(steps)
        x = start_pose[0] + frac * (end_pose[0] - start_pose[0])
        y = start_pose[1] + frac * (end_pose[1] - start_pose[1])
        # Simple interpolation of theta
        th = angle_wrap(start_pose[2] + frac * angle_wrap(end_pose[2] - start_pose[2]))
        path_poses.append((x, y, th))

    return is_collision_free(k_col, path_poses)


class GTCollisionChecker:
    def __init__(self, env, vis=False):
        self.env = env
        self.vis = vis
        self.vis_images = []

    def _check_collision(self):
        sim = self.env.sim
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]

            # Get names of the two geoms in contact
            geom1 = sim.model.geom_id2name(contact.geom1)
            geom2 = sim.model.geom_id2name(contact.geom2)

            # Skip if names not found (safety check)
            if geom1 is None or geom2 is None:
                continue

            # Check if either geom belongs to the robot or mobile base
            is_robot1 = geom1.startswith("robot0") or geom1.startswith("mobilebase0")
            is_robot2 = geom2.startswith("robot0") or geom2.startswith("mobilebase0")

            # If one is robot and the other is not from robot/mobile base, it's a collision
            if is_robot1 != is_robot2:
                return True

        return False

    def compute_score(self, pose_arr, numpy=True, robot_size=0):
        assert numpy
        scores = []
        for pose in pose_arr:
            current_state = self.env.sim.get_state()
            set_robot_base_pose(self.env, pose)
            col_free = not self._check_collision()
            if self.vis:
                img = self.env.sim.render(camera_name="robot0_frontview", width=512, height=512)[::-1]
                info_dict = {
                    "pose": ", ".join([f"{p:.3f}" for p in pose]),
                    "col_free": "true" if col_free else "false"
                }
                vis_img = add_caption_to_img(img, info_dict)
                self.vis_images.append(vis_img)
            self.env.sim.set_state(current_state)
            self.env.sim.forward()
            # print(f"Pose {pose} is collision free? {col_free}")
            scores.append(col_free)
        return np.array(scores)


def get_base_pose(env, unwrapped=False):
    if not unwrapped:
        unwrapped_env = env.unwrapped.env
    else:
        unwrapped_env = env
    base_pos, base_mat = unwrapped_env.robots[0].composite_controller.get_controller_base_pose("right")
    heading = R.from_matrix(base_mat).as_euler("xyz")[-1]
    return np.array([base_pos[0], base_pos[1], heading])


def set_robot_base_pose(env, xy_heading):
    """
    Sets the robot's base pose to the specified absolute position and heading,
    considering joint offsets in the parent frame.

    Args:
        env: The simulation environment.
        xy_heading: A list or array [x, y, heading] specifying the absolute position (x, y)
                    and heading angle (yaw) in world coordinates.
    """
    # Extract x, y, and heading (yaw) from input
    x, y, heading = xy_heading

    # Get the parent body's world position and orientation (robot0_base)
    parent_body_name = "robot0_base"
    parent_body_id = env.sim.model.body_name2id(parent_body_name)
    parent_pos = env.sim.data.body_xpos[parent_body_id][:2]  # Parent position (x, y) in world frame
    parent_quat = env.sim.data.body_xquat[parent_body_id]    # Parent orientation (quaternion) in world frame

    # Adjust for the joint offset
    forward_joint_offset = env.sim.model.jnt_pos[env.sim.model.joint_name2id("mobilebase0_joint_mobile_forward")][:2]
    side_joint_offset = env.sim.model.jnt_pos[env.sim.model.joint_name2id("mobilebase0_joint_mobile_side")][:2]
    joint_offset = np.array([side_joint_offset[0], forward_joint_offset[1]])

    # Compute the relative position in the world frame
    parent_rot = R.from_quat(parent_quat[[1, 2, 3, 0]])  # Convert parent quaternion to rotation matrix
    joint_offset_rot_target = R.from_euler("z", heading).apply(np.array([joint_offset[0], joint_offset[1], 0]))[:2]
    joint_offset_rot_source = parent_rot.apply(np.array([joint_offset[0], joint_offset[1], 0]))[:2]
    abs_pos = np.array([x, y])  # Absolute position in world coordinates
    rel_pos_world = (abs_pos + joint_offset_rot_target) - (parent_pos + joint_offset_rot_source)  # Relative position in the world frame

    # Rotate the relative position into the parent body's coordinate system
    rel_pos = (R.from_euler('z', 90, degrees=True) * parent_rot).apply(np.array([rel_pos_world[0], -rel_pos_world[1], 0.0]))[:2]  # Transform to parent frame

    # Compute the relative heading (yaw)
    parent_yaw = R.from_quat(parent_quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=False)[2]  # Extract parent yaw
    rel_heading = heading - parent_yaw

    # Update the qpos values for the joints
    forward_joint_idx = env.sim.model.joint_name2id("mobilebase0_joint_mobile_forward")
    side_joint_idx = env.sim.model.joint_name2id("mobilebase0_joint_mobile_side")
    yaw_joint_idx = env.sim.model.joint_name2id("mobilebase0_joint_mobile_yaw")

    env.sim.data.qpos[forward_joint_idx] = rel_pos[0]  # Relative y position (forward joint)
    env.sim.data.qpos[side_joint_idx] = rel_pos[1]     # Relative x position (side joint)
    env.sim.data.qpos[yaw_joint_idx] = rel_heading     # Relative yaw angle (yaw joint)

    # Update the simulation state
    env.sim.forward()
    env.sim.step()

    realized_base_pose = get_base_pose(env, unwrapped=True)
    if not np.allclose(realized_base_pose, xy_heading, atol=1e-3):
        f"\033[1;31m[env_utils.py] Requested robot pose {xy_heading} is not met by realized robot position {realized_base_pose}\033[0m"
