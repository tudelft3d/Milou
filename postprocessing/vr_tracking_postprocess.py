import datetime
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from bs4 import BeautifulSoup


def parse_log_file(file_path):
    """Reads the VR trackers' log file into a dictionary

        Args:
            file_path: path to txt file.
        Returns:
            A dictionary with timestamps and position x, y, z and rotation x, y, z, w
    """
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expressions to extract timestamps, positions, and rotations
    timestamp_pattern = re.compile(r'\[(.*?)\]')
    position_pattern = re.compile(r'(Generic Tracker \d+) Position: x = ([\d.-]+), y = ([\d.-]+), z = ([\d.-]+)')
    rotation_pattern = re.compile(
        r'(Generic Tracker \d+) Rotation: x = ([\d.-]+(?:e[+-]?\d+)?), y = ([\d.-]+(?:e[+-]?\d+)?), '
        r'z = ([\d.-]+(?:e[+-]?\d+)?), w = ([\d.-]+(?:e[+-]?\d+)?)'
    )

    # Initialize tracker_data with keys for each required attribute
    tracker_data = defaultdict(lambda: {
        'timestamp': [],
        'position_x': [], 'position_y': [], 'position_z': [],
        'rotation_x': [], 'rotation_y': [], 'rotation_z': [], 'rotation_w': []
    })

    # Extract all timestamps
    timestamps = timestamp_pattern.findall(data)
    ts_idx = 0  # Index for tracking current timestamp

    current_timestamp = None
    for line in data.splitlines():
        # Update current timestamp for each new timestamp line
        if "[2024" in line:
            current_timestamp = pd.to_datetime(timestamps[ts_idx])
            ts_idx += 1
        elif "Generic Tracker" in line and "Position" in line:
            pos_match = position_pattern.search(line)
            if pos_match:
                tracker, x, y, z = pos_match.groups()
                tracker_data[tracker]['timestamp'].append(current_timestamp)
                tracker_data[tracker]['position_x'].append(float(x))
                tracker_data[tracker]['position_y'].append(float(y))
                tracker_data[tracker]['position_z'].append(float(z))
                # Placeholder None values for rotation components if not yet parsed
                tracker_data[tracker]['rotation_x'].append(None)
                tracker_data[tracker]['rotation_y'].append(None)
                tracker_data[tracker]['rotation_z'].append(None)
                tracker_data[tracker]['rotation_w'].append(None)
        elif "Generic Tracker" in line and "Rotation" in line:
            rot_match = rotation_pattern.search(line)
            if rot_match:
                tracker, x, y, z, w = rot_match.groups()
                # Add rotation data aligned with the latest position entry
                if len(tracker_data[tracker]['rotation_x']) > 0 and tracker_data[tracker]['rotation_x'][-1] is None:
                    tracker_data[tracker]['rotation_x'][-1] = float(x)
                    tracker_data[tracker]['rotation_y'][-1] = float(y)
                    tracker_data[tracker]['rotation_z'][-1] = float(z)
                    tracker_data[tracker]['rotation_w'][-1] = float(w)
                else:
                    # If rotation comes without a position, append all data with current timestamp
                    tracker_data[tracker]['timestamp'].append(current_timestamp)
                    tracker_data[tracker]['position_x'].append(None)
                    tracker_data[tracker]['position_y'].append(None)
                    tracker_data[tracker]['position_z'].append(None)
                    tracker_data[tracker]['rotation_x'].append(float(x))
                    tracker_data[tracker]['rotation_y'].append(float(y))
                    tracker_data[tracker]['rotation_z'].append(float(z))
                    tracker_data[tracker]['rotation_w'].append(float(w))

    return tracker_data


def read_xml_file(path):
    """Reads the dantec HSA's log file

        Args:
            file_path: path to XML file.
        Returns:
            start timestamp, velocity magnitude measurements and temperature measurements
    """
    # Load the XML file
    with open(path, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Parse the XML content with BeautifulSoup
    soup = BeautifulSoup(xml_content, 'xml')

    # Extracting header information
    header = soup.find('data0')
    # for some reason the date is YYYY-DD-MM which needs to be fixed to YYYY-MM-DD
    fix_data_start = (header['dateStart']).split("T")[0].split("-")[0] + "-" + \
                     (header['dateStart']).split("T")[0].split("-")[2] + "-" + \
                     (header['dateStart']).split("T")[0].split("-")[1] + " " + (header['dateStart']).split("T")[1]
    date_start = pd.to_datetime(fix_data_start)

    # Find all 'data0', 'data1', 'data2', etc. elements
    data_entries = soup.find_all(re.compile('data[0-9]+'))

    # Iterate through each data entry
    for data_entry in data_entries:
        entry_key = data_entry.get('ID')  # Use a unique identifier as the dictionary key
        time_values = []
        mean_v_values = []
        tu_values = []
        mean_t_values = []
        dr_values = []

        # Find all 'data' elements under the current data entry
        data_elements = data_entry.find_all('data')

        # Iterate through each 'data' element
        for data in data_elements:
            time = data.get('Time')
            mean_v = data.get('MeanV')
            tu = data.get('Tu')
            mean_t = data.get('MeanT')
            dr = data.get('DR')

            time_values.append(float(time))
            mean_v_values.append(float(mean_v))
            tu_values.append(float(tu))
            mean_t_values.append(float(mean_t))
            dr_values.append(float(dr))

        measurements = {'Time': np.array(time_values),
                        'Velocity magnitude': np.array(mean_v_values),
                        'Temperature': np.array(mean_t_values),
                        'Turbulence intensity': np.array(tu_values),
                        'Draught rate': np.array(dr_values)}

    return date_start, measurements["Velocity magnitude"], measurements["Temperature"]


def rotation_matrix_between(a, b):
    """Compute the rotation matrix that rotates vector a to vector b. Credit goes to Nail Ibrahimli

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    # Normalize the vectors
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # Compute the axis of rotation (cross product)
    v = np.cross(a, b)

    # Handle cases where `a` and `b` are parallel
    eps = 1e-6
    if np.sum(np.abs(v)) < eps:
        x = np.array([1.0, 0, 0]) if abs(a[0]) < eps else np.array([0, 1.0, 0])
        v = np.cross(a, x)

    v = v / np.linalg.norm(v)

    # Skew-symmetric matrix for cross product
    skew_sym_mat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Compute the angle between the vectors
    theta = np.arccos(np.clip(np.dot(a, b), -1, 1))

    # Rodrigues' rotation formula
    return np.eye(3) + np.sin(theta) * skew_sym_mat + (1 - np.cos(theta)) * (skew_sym_mat @ skew_sym_mat)


def fit_to_plane(points_3d):
    """Fits points into a 3D plane

        Args:
            points_3d: numpy array of 3d points.
        Returns:
            the normal of the plane and X, Y, Z of points on the plane for plotting
    """
    # Step 1: Fit a plane using PCA
    # Use Principal Component Analysis (PCA) to find the best-fit plane
    pca = PCA(n_components=2)
    pca.fit(points_3d)

    # The normal vector to the plane is the third component from PCA
    normal = np.cross(pca.components_[0], pca.components_[1])

    # Calculate the point on the plane closest to the origin (the mean of the points_3d)
    point_on_plane = np.mean(points_3d, axis=0)

    # Plane equation: ax + by + cz = d
    a, b, c = normal
    d = -point_on_plane.dot(normal)

    # Step 2: Create a meshgrid for the plane
    # Choose a range of x and y values around the points_3d for plotting the plane
    x_vals = np.linspace(np.min(points_3d[:, 0]), np.max(points_3d[:, 0]), 10)
    y_vals = np.linspace(np.min(points_3d[:, 1]), np.max(points_3d[:, 1]), 10)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = (-a * X - b * Y - d) / c  # Solve for Z based on plane equation

    return normal, X, Y, Z


def angle_between_vectors(u, v):
    """
    Calculate the angle between two vectors using the cross product.

    Args:
    u, v : array-like
        Input vectors.

    Returns:
    tuple
        The angle between the vectors in radians.
    """

    # Compute the dot product and magnitudes
    dot_product = np.dot(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)

    # Avoid division by zero
    if magnitude_u == 0 or magnitude_v == 0:
        raise ValueError("Vectors must have non-zero magnitude.")

    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_u * magnitude_v)

    # Clip the value to the range [-1, 1] to handle floating-point precision issues
    cos_theta = np.clip(cos_theta, -1, 1)

    # Compute and return the angle in radians
    return np.arccos(cos_theta)


def rotate_point_2d(center, reference, generic_point):
    """Compute the rotation matrix that rotates a generic 2d point to a 2d reference point
    from a 2d center point of rotation.

    Args:
    list
        generic_point: The 2d point to rotate.
        reference: The 2d point to rotate to.
        centre: The centre of rotation.
    Returns:
        The rotation matrix.
    """
    # Convert points to numpy arrays for easier vector math
    center = np.array(center)
    reference = np.array(reference)
    generic_point = np.array(generic_point)

    # Compute vectors
    vec_ref = reference - center
    vec_point = generic_point - center

    # Normalize the vectors
    vec_ref_norm = vec_ref / np.linalg.norm(vec_ref)
    vec_point_norm = vec_point / np.linalg.norm(vec_point)

    # Compute the angle between the two vectors
    # angle = np.arccos(np.clip(np.dot(vec_ref_norm, vec_point_norm), -1.0, 1.0))
    angle = np.radians(144)

    # Compute the cross product to determine the rotation direction
    cross = np.cross(vec_ref_norm, vec_point_norm)
    if cross > 0:
        angle = -angle

    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    # # Apply the rotation
    # rotated_vec = np.dot(rotation_matrix, vec_point)
    # rotated_point = rotated_vec + center

    return rotation_matrix


def fit_line(points):
    """Fit a line through points.

        Args:
        np.array
            points: 2d points to fit a line through.
        Returns:
            The slope and intercept of the line.
    """
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_  # Slope and intercept


def middle_line(cluster1, cluster2):
    """Finds the middle line between 2 clusters of points

            Args:
            np.array
                points: array of 2d points.
            Returns:
                The slope and intercept of the middle line.
        """
    slope1, intercept1 = fit_line(cluster1)
    slope2, intercept2 = fit_line(cluster2)
    middle_slope = (slope1 + slope2) / 2
    middle_intercept = (intercept1 + intercept2) / 2
    return middle_slope, middle_intercept


def compute_position_accuracy(points):
    """Computes the positional and angular accuracy of measured points

        Args:
        dict
            points: dictionary of points with values of 2 trackers.
        Returns:
            2 dictionaries for positional and angular accuracy
    """
    position_errors = {}
    angular_errors = {}

    for point_name, rounds in points.items():
        tracker_1_positions = np.array([round["tracker_1"] for round in rounds])
        tracker_2_positions = np.array([round["tracker_2"] for round in rounds])

        # Compute the mean position for Tracker 1 and Tracker 2
        mean_tracker_1 = tracker_1_positions.mean(axis=0)
        mean_tracker_2 = tracker_2_positions.mean(axis=0)

        # Compute position errors (Euclidean distance to the mean)
        tracker_1_errors = np.linalg.norm(tracker_1_positions - mean_tracker_1, axis=1)
        tracker_2_errors = np.linalg.norm(tracker_2_positions - mean_tracker_2, axis=1)

        # Store position accuracy as the mean error for Tracker 1 and Tracker 2
        position_errors[point_name] = {
            "tracker_1_mean_error": np.round(tracker_1_errors.mean(), 4),
            "tracker_2_mean_error": np.round(tracker_2_errors.mean(), 4),
            "tracker_1_std_error": np.round(tracker_1_errors.std(), 4),
            "tracker_2_std_error": np.round(tracker_2_errors.std(), 4),
        }

        # Compute angular accuracy
        vectors = tracker_2_positions - tracker_1_positions
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # Angle of each vector
        mean_angle = np.arctan2(
            np.mean(np.sin(angles)), np.mean(np.cos(angles))
        )  # Mean angle using circular statistics
        angular_deviations = np.arctan2(
            np.sin(angles - mean_angle), np.cos(angles - mean_angle)
        )  # Angular difference to the mean angle

        # Store angular accuracy as the mean angular deviation (in degrees)
        angular_errors[point_name] = {
            "mean_angle": np.round(np.degrees(mean_angle), 1),
            "mean_angular_error": np.round(np.degrees(np.abs(angular_deviations).mean()), 1),
            "angular_error_std": np.round(np.degrees(angular_deviations.std()), 1),
        }

    return position_errors, angular_errors


def fixPlot(fontname='Times New Roman', thickness=1.5, fontsize=20, markersize=8, labelsize=15, texuse=True, tickSize=15):
    '''
        This plot sets the default plot parameters - Courtesy of Akshay Patil
    INPUT
        thickness:      [float] Default thickness of the axes lines
        fontsize:       [integer] Default fontsize of the axes labels
        markersize:     [integer] Default markersize
        labelsize:      [integer] Default label size
    OUTPUT
        None
    '''
    # Set the font of the text
    plt.rcParams['font.family'] = fontname
    # Set the thickness of plot axes
    plt.rcParams['axes.linewidth'] = thickness
    # Set the default fontsize
    plt.rcParams['font.size'] = fontsize
    # Set the default markersize
    plt.rcParams['lines.markersize'] = markersize
    # Set the axes label size
    plt.rcParams['axes.labelsize'] = labelsize
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = texuse
    # Tick size
    plt.rcParams['xtick.major.size'] = tickSize
    plt.rcParams['ytick.major.size'] = tickSize
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


if __name__ == "__main__":
    new_path = "/usr/local/texlive/2023/bin/universal-darwin"
    os.environ["PATH"] = os.environ.get("PATH", "") + f":{new_path}" if new_path not in os.environ.get("PATH", "") else \
        os.environ["PATH"]
    sns.set_palette("colorblind")
    # points on track
    name_points = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    vr_data_path = 'data/setup_1_repositioning_accuracy/vr_tracking_data_2024_11_01_cleaned.log'
    trackers_data = parse_log_file(vr_data_path)

    dantec_data_path = 'data/setup_1_repositioning_accuracy/dantec'
    dict_points = {}
    for filename in os.listdir(dantec_data_path):
        if filename.endswith(".xml"):
            path_to_filename = os.path.join(dantec_data_path, filename)
            time_dantec, U_dantec, T_dantec = read_xml_file(path_to_filename)
            dict_points[time_dantec] = [U_dantec, T_dantec]

    sorted_points = sorted(dict_points.keys())[:30]  # removing the last round where tracker 2 turned off

    total_number_of_points = len(name_points)
    rounds_number = 3

    points_location = []

    for i in range(total_number_of_points):
        position_point_x = []
        position_point_y = []
        position_point_z = []
        for j in range(rounds_number):
            time_start = sorted_points[i + (j*total_number_of_points)]
            end_time_interval = time_start + datetime.timedelta(minutes=5)
            U_point, T_point = dict_points[time_start]
            print("Mean U is ", np.mean(U_point))

            for tracker, t_data in trackers_data.items():
                time_stamps_tracker = np.array(t_data['timestamp'])
                indices = np.where((time_stamps_tracker >= time_start) & (time_stamps_tracker <= end_time_interval))[0]
                measurement_time = time_stamps_tracker[indices]
                position_x = np.array(t_data['position_x'])[indices]
                position_y = np.array(t_data['position_y'])[indices]
                position_z = np.array(t_data['position_z'])[indices]
                rotation_x = np.array(t_data['rotation_x'])[indices]
                rotation_y = np.array(t_data['rotation_y'])[indices]
                rotation_z = np.array(t_data['rotation_z'])[indices]
                rotation_w = np.array(t_data['rotation_w'])[indices]

                mean_position_x = np.mean(position_x)
                mean_position_y = np.mean(position_y)
                mean_position_z = np.mean(position_z)
                mean_rotation_x = np.mean(rotation_x)
                mean_rotation_y = np.mean(rotation_y)
                mean_rotation_z = np.mean(rotation_z)
                mean_rotation_w = np.mean(rotation_w)

                std_position_x = np.std(position_x)
                std_position_y = np.std(position_y)
                std_position_z = np.std(position_z)
                std_rotation_x = np.std(rotation_x)
                std_rotation_y = np.std(rotation_y)
                std_rotation_z = np.std(rotation_z)
                std_rotation_w = np.std(rotation_w)

                position_point_x.append(mean_position_x)
                position_point_y.append(mean_position_y)
                position_point_z.append(mean_position_z)
                points_location.append([mean_position_x, -mean_position_z, mean_position_y])

        print("position x 1 ", np.mean([position_point_x[0], position_point_x[2], position_point_x[4]]))
        print("position y 1 ", np.mean([position_point_y[0], position_point_y[2], position_point_y[4]]))
        print("position z 1 ", np.mean([position_point_z[0], position_point_z[2], position_point_z[4]]))
        print("position x 2 ", np.mean([position_point_x[1], position_point_x[3], position_point_x[5]]))
        print("position y 2 ", np.mean([position_point_y[1], position_point_y[3], position_point_y[5]]))
        print("position z 2 ", np.mean([position_point_z[1], position_point_z[3], position_point_z[5]]))

        print("Repositioning accuracy x 1 ", np.std([position_point_x[0], position_point_x[2], position_point_x[4]]))
        print("Repositioning accuracy y 1 ", np.std([position_point_y[0], position_point_y[2], position_point_y[4]]))
        print("Repositioning accuracy z 1 ", np.std([position_point_z[0], position_point_z[2], position_point_z[4]]))
        print("Repositioning accuracy x 2 ", np.std([position_point_x[1], position_point_x[3], position_point_x[5]]))
        print("Repositioning accuracy y 2 ", np.std([position_point_y[1], position_point_y[3], position_point_y[5]]))
        print("Repositioning accuracy z 2 ", np.std([position_point_z[1], position_point_z[3], position_point_z[5]]))
        test_0 = (position_point_x[0] - position_point_x[1]) ** 2
        test_1 = (position_point_y[0] - position_point_y[1]) ** 2
        test_2 = (position_point_z[0] - position_point_z[1]) ** 2

        print("Distance ", np.sqrt(test_0 + test_1 + test_2))

    fixPlot(thickness=1.2, fontsize=12, markersize=6, labelsize=12, texuse=True, tickSize=5)

    array_points_location = np.array(points_location)
    normal_plane, X, Y, Z = fit_to_plane(array_points_location)

    # Rotate points
    vertical_vector = np.array([0, 0, 1])

    rotated_points = []
    rotation_matrix = rotation_matrix_between(normal_plane, vertical_vector)

    for point in array_points_location:
        rotated_points.append(rotation_matrix.dot(point))

    rotated_points = np.array(rotated_points)
    normal_plane_rotated, X0, Y0, Z0 = fit_to_plane(rotated_points)

    # move points to the room's coordinate system
    median_rotated_points = np.median(rotated_points, axis=0)
    translated_points = rotated_points - median_rotated_points

    one_translated_point = translated_points[0][0:2]

    rotation_matrix_2d = rotate_point_2d([0, 0], [0.15, 0.41], one_translated_point)

    aligned_points = []
    for translated_point in translated_points:
        rotated_point_intermediate = np.dot(rotation_matrix_2d, translated_point)
        move_intermediate_point = rotated_point_intermediate
        aligned_points.append(move_intermediate_point)

    aligned_points = np.array(aligned_points)

    # get the defined path from the points
    clusters = {
        "top_1": np.array([aligned_points[0], aligned_points[2], aligned_points[4],
                           aligned_points[6], aligned_points[8], aligned_points[10]]),
        "top_2": np.array([aligned_points[1], aligned_points[3], aligned_points[5],
                           aligned_points[7], aligned_points[9], aligned_points[11]]),
        "left_1": np.array([aligned_points[12], aligned_points[14], aligned_points[16],
                            aligned_points[18], aligned_points[20], aligned_points[22],
                            aligned_points[24], aligned_points[26], aligned_points[28]]),
        "left_2": np.array([aligned_points[13], aligned_points[15], aligned_points[17],
                            aligned_points[19], aligned_points[21], aligned_points[23],
                            aligned_points[25], aligned_points[27], aligned_points[29]]),
        "bottom_1": np.array([aligned_points[30], aligned_points[32], aligned_points[34],
                              aligned_points[36], aligned_points[38], aligned_points[40]]),
        "bottom_2": np.array([aligned_points[31], aligned_points[33], aligned_points[35],
                              aligned_points[37], aligned_points[39], aligned_points[41]]),
        "right_1": np.array([aligned_points[42], aligned_points[44], aligned_points[46],
                             aligned_points[48], aligned_points[50], aligned_points[52],
                             aligned_points[54], aligned_points[56], aligned_points[58]]),
        "right_2": np.array([aligned_points[43], aligned_points[45], aligned_points[47],
                             aligned_points[49], aligned_points[51], aligned_points[53],
                             aligned_points[55], aligned_points[57], aligned_points[59]])
    }  # the 4 cluster of points

    # Define sides using the middle line between two clusters
    middle_lines = {
        "top": middle_line(clusters["top_1"], clusters["top_2"]),
        "left": middle_line(clusters["left_1"], clusters["left_2"]),
        "bottom": middle_line(clusters["bottom_1"], clusters["bottom_2"]),
        "right": middle_line(clusters["right_1"], clusters["right_2"]),
    }

    # Hardcoded
    vertices = [
        np.array([-0.425, 0.48]),
        np.array([0.42, 0.49]),
        np.array([0.41, -0.46]),
        np.array([-0.435, -0.45])
    ]

    plt.figure(figsize=(6, 4))

    n_points = 10  # Select the first 10 points
    rounds = 3
    trackers = 2
    tracker_names = ["tracker_1", "tracker_2"]

    # Reshape the data for easier handling
    reshaped_data = aligned_points[:, 0:2].reshape(-1, trackers * rounds, 2)
    round_colors = []

    reshaped_data_for_statistics = {}

    for point_idx, point_data in enumerate(reshaped_data):
        reshaped_data_for_statistics[name_points[point_idx]] = []
        for round_idx in range(rounds):
            round_lap = {}
            if len(round_colors) < 3:
                round_colors.append(sns.color_palette()[round_idx])
            for tracker_idx in range(trackers):
                # Extract the [x, y] position
                idx = round_idx * trackers + tracker_idx
                x, y = point_data[idx]
                if tracker_idx % 2 == 0:
                    round_lap[tracker_names[0]] = [x, y]
                else:
                    round_lap[tracker_names[1]] = [x, y]
                plt.scatter(
                    x, y,
                    facecolors='none' if tracker_idx == 0 else sns.color_palette()[round_idx],
                    # Transparent fill only for 'o'
                    edgecolors=sns.color_palette()[round_idx],  # Edge color for all markers
                    marker='o' if tracker_idx == 0 else 'x'  # Marker depends on tracker_idx
                )
                if round_idx == 2 and point_idx in [0, 1, 2, 3, 4] and tracker_idx == 1:
                    plt.text(
                        x - 0.02, y + 0.03,
                        f"${name_points[point_idx]}_{tracker_idx+1}$",  # Label format
                        fontsize=9,
                        verticalalignment='bottom',
                        horizontalalignment='right'
                    )
                elif round_idx == 2 and point_idx in [5, 6] and tracker_idx == 1:
                    plt.text(
                        x, y - 0.07,
                        f"${name_points[point_idx]}_{tracker_idx+1}$",  # Label format
                        fontsize=9,
                        verticalalignment='bottom',
                        horizontalalignment='right'
                    )
                elif round_idx == 2 and point_idx in [7, 8, 9] and tracker_idx == 1:
                    plt.text(
                        x + 0.1, y - 0.05,
                        f"${name_points[point_idx]}_{tracker_idx+1}$",  # Label format
                        fontsize=9,
                        verticalalignment='bottom',
                        horizontalalignment='right'
                    )
                elif round_idx == 2 and point_idx in [0, 1] and tracker_idx == 0:
                    plt.text(
                        x + 0.08, y - 0.07,
                        f"${name_points[point_idx]}_{tracker_idx+1}$",  # Label format
                        fontsize=9,
                        verticalalignment='bottom',
                        horizontalalignment='right'
                    )
                elif round_idx == 2 and point_idx in [2, 3, 4] and tracker_idx == 0:
                    plt.text(
                        x + 0.09, y + 0.02,
                        f"${name_points[point_idx]}_{tracker_idx+1}$",  # Label format
                        fontsize=9,
                        verticalalignment='bottom',
                        horizontalalignment='right'
                    )
                elif round_idx == 2 and point_idx not in [0, 1, 2, 3, 4] and tracker_idx == 0:
                    plt.text(
                        x - 0.02, y + 0.03,
                        f"${name_points[point_idx]}_{tracker_idx+1}$",  # Label format
                        fontsize=9,
                        verticalalignment='bottom',
                        horizontalalignment='right'
                    )
            reshaped_data_for_statistics[name_points[point_idx]].append(round_lap)

    # Compute the accuracy metrics
    position_accuracy, angular_accuracy = compute_position_accuracy(reshaped_data_for_statistics)

    # Prepare data for table
    statistical_data = []
    for point in position_accuracy.keys():
        position_metrics = position_accuracy[point]
        angular_metrics = angular_accuracy[point]
        statistical_data.append({
            "Point": point,
            "Tracker 1 Mean Error (Pos)": position_metrics["tracker_1_mean_error"],
            "Tracker 2 Mean Error (Pos)": position_metrics["tracker_2_mean_error"],
            "Tracker 1 Std Error (Pos)": position_metrics["tracker_1_std_error"],
            "Tracker 2 Std Error (Pos)": position_metrics["tracker_2_std_error"],
            # "Mean Angle (°)": angular_metrics["mean_angle"],
            "Mean Angular Error (°)": angular_metrics["mean_angular_error"],
            "Angular Error Std (°)": angular_metrics["angular_error_std"],
        })

    # Create a pandas DataFrame
    df = pd.DataFrame(statistical_data)

    # Convert the DataFrame to LaTeX table format
    latex_table = df.head(10).to_latex(index=False, escape=False, float_format="%.3f")

    # Print the LaTeX table
    print(latex_table)

    # Prepare for a rectangular legend
    handles = [
        Patch(facecolor=round_colors[0], label='Lap 1'),
        Patch(facecolor=round_colors[1], label='Lap 2'),
        Patch(facecolor=round_colors[2], label='Lap 3'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='black', label='Tracker 1', linestyle='None'),
        plt.Line2D([0], [0], marker='x', color='black', label='Tracker 2', linestyle='None'),
        plt.Line2D([0], [0], linestyle='--', color='grey', label="Milou's path")
    ]

    # Add dummy points for legend
    for round_idx, color in enumerate(round_colors):
        plt.scatter([], [], color=color, label=f'Round {round_idx + 1}')

    plt.scatter([], [], color='black', label='Tracker 1', marker='o')
    plt.scatter([], [], color='black', label='Tracker 2', marker='x')

    # Connect the vertices to form a rectangle
    vertices = np.vstack([vertices, vertices[0]])  # Close the rectangle
    plt.plot(vertices[:, 0], vertices[:, 1], color='grey', linestyle='--', label="path")

    # Customise the plot
    plt.title("VR trackers positioning measurements")
    plt.xlabel("X Coordinate [m]")
    plt.ylabel("Y Coordinate [m]")
    # Set the same tick intervals for both axes
    tick_interval = 0.2  # Change this as needed
    plt.xticks(np.arange(-0.7, 0.7, tick_interval))
    plt.yticks(np.arange(-0.7, 0.7, tick_interval))
    plt.gca().set_xlim([-0.7, 0.7])
    plt.gca().set_ylim([-0.7, 0.7])
    plt.gca().set_aspect('equal', adjustable='box')
    # Hide the top and right spines
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Define original tick positions
    original_ticks = np.arange(-0.7, 0.75, 0.2)

    # Convert original tick positions to new labels
    new_labels = [str(round(t + 0.7, 2)) for t in original_ticks]  # Shift by +0.7
    # Remove duplicate "0" at the origin by setting the first label to an empty string
    new_labels_y = [""] + new_labels[1:] # Remove "0" from one axis

    # Set new tick labels
    plt.gca().set_xticks(original_ticks)
    plt.gca().set_xticklabels(new_labels)
    plt.gca().set_yticks(original_ticks)
    plt.gca().set_yticklabels(new_labels_y)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    # plt.savefig(f"output/vr_path_annotated_01.pdf", bbox_inches='tight', dpi=300)
    plt.show()

