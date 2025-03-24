from datetime import timedelta, datetime
from bs4 import BeautifulSoup
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
import re
import scipy.stats as stats


def read_xlsx_hobos(path_hobos):
    """Reads the HOBO log file

        Args:
            file_path: path to xlsx file.
        Returns:
            dictionary of fields timestamp, temperature, relative humidity, and CO2 concentration
    """
    sns.set_palette("colorblind")
    hobos_num_to_symbol = {"20930890": "H1", "20965420": "H2", "20967180": "H3", "20930886": "H4", "20971954": "H5",
                           "20967179": "H6"}
    hobos_dict = {}
    for folder_hobo in os.listdir(path_hobos):
        if os.path.isdir(os.path.join(path_hobos, folder_hobo)):
            for hobo_file in os.listdir(os.path.join(path_hobos, folder_hobo)):
                if hobo_file.endswith(".xlsx"):
                    hobo_number = hobo_file.split()[0]
                    hobo_filename = os.path.join(path_hobos, folder_hobo, hobo_file)
                    dataframe = pd.read_excel(hobo_filename)
                    data_dict = dataframe.to_dict(orient='list')
                    hobos_dict[hobos_num_to_symbol[hobo_number]] = data_dict
    return hobos_dict


def read_xml_file(path, probe_to_tag):
    """Reads the dantec HSA's log file

        Args:
            file_path: path to XML file.
        Returns:
            start timestamp, dictionnary of fields "Time" as the time step, "Velocity magnitude", "Temperature",
            "Turbulence intensity", "Draught rate"
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
    all_measurements = {}

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

        entry_tag = probe_to_tag[entry_key]
        measurements = {'Time': np.array(time_values),
                        'Velocity magnitude': np.array(mean_v_values),
                        'Temperature': np.array(mean_t_values),
                        'Turbulence intensity': np.array(tu_values),
                        'Draught rate': np.array(dr_values)}
        all_measurements[entry_tag] = measurements

    return date_start, all_measurements


def detect_outliers(data):
    """Detects extreme outliers from the measurements

        Args:
        array-like
            data: values of a measured field.
        Returns:
            list of indices of outliers in the dataset
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 4.0 * IQR
    upper_bound = Q3 + 4.0 * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]


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


def extract_point_data(path_to_files):
    """Extract data per point

        Args:
        string
            path_to_files: path to xml files from HSAs.
        Returns:
            dictionary of points and their respective fields
    """
    dict_points_to_extract = {}
    for filename in os.listdir(path_to_files):
        if filename.endswith(".xml"):
            path_to_filename = os.path.join(path_to_files, filename)
            time_dantec, measured_dantec = read_xml_file(path_to_filename, probes_dict)
            dict_points_to_extract[time_dantec] = measured_dantec
    return dict_points_to_extract


if __name__ == "__main__":
    sns.set_palette("colorblind")
    new_path = "/usr/local/texlive/2023/bin/universal-darwin"
    os.environ["PATH"] = os.environ.get("PATH", "") + f":{new_path}" if new_path not in os.environ.get("PATH", "") \
        else os.environ["PATH"]

    probes_dict = {'54T33:849': 'probe 1',
                   '54T33:850': 'probe 2',
                   '54T33:851': 'probe 3'}

    test_path = "data/setup_2_HSA_no_fans_01"

    dict_points = extract_point_data(test_path)

    hobos_dict_temperature = {"milou_lights": [],
                              "no_lights": [],
                              "lights": []}

    path_to_hobos = "data/hobos_chamber"
    hobos_data = read_xlsx_hobos(path_to_hobos)

    # fixPlot(thickness=1.2, fontsize=18, markersize=8, labelsize=18, texuse=True, tickSize=5)
    # plt.figure(figsize=(10, 5))

    for hobo_data, hobo_fields in hobos_data.items():
        print(hobo_data)
        # Define your start and end timestamps
        start_timestamp_milou_lights = np.datetime64('2024-11-25T07:00:00')
        end_timestamp_milou_lights = np.datetime64('2024-11-25T15:57:00')  # 13:05
        start_timestamp_no_milou_no_lights = np.datetime64('2025-01-14T08:05:00')
        end_timestamp_no_milou_no_lights = np.datetime64('2025-01-14T17:02:00')
        start_timestamp_no_milou_lights = np.datetime64('2025-01-15T08:05:00')
        end_timestamp_no_milou_lights = np.datetime64('2025-01-15T17:02:00')  # 14:05
        # time_step = (np.array(hobo_fields['Date-Time (CET)']) + datetime.timedelta(days=224, hours=16, minutes=30))
        time_step = np.array(hobo_fields['Date-Time (CET)'])

        # Use np.where to filter between the two timestamps
        indices_milou_lights = np.where((time_step >= start_timestamp_milou_lights) &
                                        (time_step <= end_timestamp_milou_lights))
        time_step_milou_lights = time_step[indices_milou_lights]
        indices_no_milou_no_lights = np.where((time_step >= start_timestamp_no_milou_no_lights) &
                                              (time_step <= end_timestamp_no_milou_no_lights))
        time_step_no_milou_no_lights = time_step[indices_no_milou_no_lights]
        indices_no_milou_lights = np.where((time_step >= start_timestamp_no_milou_lights) &
                                           (time_step <= end_timestamp_no_milou_lights))
        time_step_no_milou_lights = time_step[indices_no_milou_lights]
        if 'Temperature (°C) ' in hobo_fields.keys():
            hobos_dict_temperature["milou_lights"].append(np.array(hobo_fields['Temperature (°C) '])
                                                          [indices_milou_lights])
            hobos_dict_temperature["no_lights"].append(np.array(hobo_fields['Temperature (°C) '])
                                                       [indices_no_milou_no_lights])
            hobos_dict_temperature["lights"].append(np.array(hobo_fields['Temperature (°C) '])
                                                    [indices_no_milou_lights])
        elif 'Ch:1 - Temperature   (°C)' in hobo_fields.keys():
            hobos_dict_temperature["milou_lights"].append(np.array(hobo_fields['Ch:1 - Temperature   (°C)'])
                                                          [indices_milou_lights])
            hobos_dict_temperature["no_lights"].append(np.array(hobo_fields['Ch:1 - Temperature   (°C)'])
                                                       [indices_no_milou_no_lights])
            hobos_dict_temperature["lights"].append(np.array(hobo_fields['Ch:1 - Temperature   (°C)'])
                                                    [indices_no_milou_lights])

    mean_milou_lights = np.mean(hobos_dict_temperature["milou_lights"], axis=0)
    mean_no_lights = np.mean(hobos_dict_temperature["no_lights"], axis=0)
    mean_lights = np.mean(hobos_dict_temperature["lights"], axis=0)

    plot_type = "lights_on_off"  # "lights_on_off", "milou_no_milou", "hsa_hobos"

    if plot_type == "lights_on_off":
        # plots hobo temperature measurements from lights on and off
        fixPlot(thickness=1.2, fontsize=14, markersize=8, labelsize=14, texuse=True, tickSize=5)
        plt.figure(figsize=(10, 5))
        time_range = range(0, 540, 5)  # Time in minutes past 8:05

        # Generate time labels
        start_time = datetime.strptime("08:00", "%H:%M")
        hour_intervals = [i for i in time_range if (start_time + timedelta(minutes=i)).minute == 0]
        time_labels = [start_time + timedelta(minutes=i) for i in hour_intervals]
        time_labels_formatted = [t.strftime("%H:%M") for t in time_labels]

        # plt.plot(time_range, mean_milou_lights, label="Milou - lights on")
        plt.plot(time_range, mean_no_lights, label="Lights off", linestyle="dashed", color=sns.color_palette()[0])
        plt.plot(time_range, mean_lights, label="Lights on", linestyle="dashdot", color=sns.color_palette()[1])
        plt.xlabel("Time [minutes]")
        plt.ylabel("Mean temperature [°C]")
        plt.xticks(hour_intervals, time_labels_formatted, rotation=45)
        plt.xlabel("Time of Day")
        # Extracting handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Sorting by label
        sorted_labels_handles = sorted(zip(labels, handles), key=lambda x: x[0][-1])

        # Unzip the sorted tuples
        sorted_labels, sorted_handles = zip(*sorted_labels_handles)

        # Add the sorted legend
        plt.title("Impact of lights on temperature")
        plt.legend(sorted_handles, sorted_labels, loc="upper center", bbox_to_anchor=(0.1, -0.2), frameon=False)
        plt.tight_layout()
        # plt.savefig(f"output/lights_on_off.pdf", bbox_inches='tight', dpi=300)
        plt.show()

    elif plot_type == "milou_no_milou":
        # Plots hobo temperature measurements between milou on and off
        fixPlot(thickness=1.2, fontsize=18, markersize=8, labelsize=18, texuse=True, tickSize=5)
        plt.figure(figsize=(10, 5))
        time_range = range(0, 305, 5)  # Time in minutes past 8:05
        mean_milou_lights_subset = mean_milou_lights[:61]
        mean_lights_subset = mean_lights[:61]

        # normalise the means
        min_mean_milou_lights = np.min(mean_milou_lights_subset)
        min_mean_lights = np.min(mean_lights_subset)
        max_mean_milou_lights = np.max(mean_milou_lights_subset)
        max_mean_lights = np.max(mean_lights_subset)

        normalised_mean_milou_lights = (mean_milou_lights_subset - np.mean(mean_milou_lights_subset)) / \
                                       (max_mean_milou_lights - min_mean_milou_lights)
        normalised_mean_lights = (mean_lights_subset - np.mean(mean_lights_subset)) / (max_mean_lights - min_mean_lights)

        # print(np.var(normalised_mean_milou_lights), np.var(normalised_mean_lights))

        normality_check_1 = stats.kstest(normalised_mean_milou_lights, 'norm')
        normality_check_2 = stats.kstest(normalised_mean_lights, 'norm')

        stat_w1, p_w1 = stats.wilcoxon(normalised_mean_milou_lights, normalised_mean_lights)
        print("p_w1 ", p_w1)

        plt.plot(time_range, normalised_mean_milou_lights, label="Milou - lights on", color=sns.color_palette()[2])
        # plt.plot(time_range, mean_no_lights, label="Lights off", linestyle="dashed")
        plt.plot(time_range, normalised_mean_lights, label="no Milou - lights on", linestyle="dashdot", color=sns.color_palette()[1])
        plt.xlabel("Time [minutes]")
        plt.ylabel("Normalised mean temperature [°C]")
        # Extracting handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Sorting by label
        sorted_labels_handles = sorted(zip(labels, handles), key=lambda x: x[0][-1])

        # Unzip the sorted tuples
        sorted_labels, sorted_handles = zip(*sorted_labels_handles)

        ax = plt.gca()  # Get current axis

        # Align the x=0 and y=0
        ax.spines['left'].set_position(('data', 0))  # Move y-axis to cross x-axis at 0
        ax.spines['right'].set_position(('data', 300))

        # Clip the spines to the axis limits to avoid overflow
        ax.spines['bottom'].set_bounds([0, 300])  # Limit bottom spine
        ax.spines['top'].set_bounds([0, 300])  # Limit top spine

        # Set limits so grid does not extend beyond the spines
        plt.xlim(left=0, right=300)
        plt.ylim(bottom=-1, top=1)


        # Add the sorted legend
        plt.title("Normalised temperature over time")
        plt.legend(sorted_handles, sorted_labels, loc="upper center", bbox_to_anchor=(0.1, -0.2), frameon=False)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        # plt.savefig(f"output/milou_no_milou.pdf", bbox_inches='tight', dpi=300)
        plt.show()

    elif plot_type == "hsa_hobos":
        # Plots hsas and hobo measurements when Milou is operating in the chamber and compares it to when Milou is not
        # in the chamber
        fixPlot(thickness=1.2, fontsize=18, markersize=8, labelsize=18, texuse=True, tickSize=5)
        plt.figure(figsize=(10, 5))
        time_difference = start_timestamp_no_milou_lights - start_timestamp_milou_lights
        time_difference_seconds = time_difference.astype('timedelta64[s]').astype(int)
        shift = timedelta(seconds=int(time_difference_seconds))
        shifted_timestamps = [(ts - shift) for ts in time_step_no_milou_lights]

        # Prepare HSA data for plotting
        interval = timedelta(seconds=0.5)
        timestamps_hsa = []
        seconds_hsa = []
        dict_temperature_hsas = {'probe 1': [],
                                 'probe 2': [],
                                 'probe 3': []}

        start_datetime = datetime.strptime("2024-11-25 00:00:00", "%Y-%m-%d %H:%M:%S")  # random start date just to avoid big numbers

        for start_time, probes in dict_points.items():
            time_marked = 0
            for probe, probe_values in probes.items():
                if time_marked == 0:
                    round_timestamps = [start_time + i * interval for i in range(len(probe_values["Time"]))]
                    timestamps_hsa.extend(round_timestamps)
                    seconds_hsa.extend([(ts - start_datetime).total_seconds() for ts in round_timestamps])
                    time_marked = 1
                U_outliers = detect_outliers(probe_values["Velocity magnitude"])
                T_outliers = detect_outliers(probe_values["Temperature"])
                outlier_indices = np.union1d(U_outliers, T_outliers)
                U_with_nan = probe_values["Velocity magnitude"].copy()
                T_with_nan = probe_values["Temperature"].copy()
                U_with_nan[outlier_indices] = np.nan
                T_with_nan[outlier_indices] = np.nan
                dict_temperature_hsas[probe].extend(T_with_nan)

        # Remove NaN values across all probes
        hsas_datasets = [seconds_hsa, dict_temperature_hsas['probe 1'], dict_temperature_hsas['probe 2'],
                         dict_temperature_hsas['probe 3']]
        valid_mask = ~np.isnan(hsas_datasets).any(axis=0)
        hsas_datasets_cleaned = []

        hsas_datasets_cleaned = np.array([np.array(data)[valid_mask] for data in hsas_datasets])

        normalised_probe_1 = ((hsas_datasets_cleaned[1] - np.mean(hsas_datasets_cleaned[1])) /
                              (np.max(hsas_datasets_cleaned[1]) - np.min(hsas_datasets_cleaned[1])))
        normalised_probe_2 = ((hsas_datasets_cleaned[2] - np.mean(hsas_datasets_cleaned[2])) /
                              (np.max(hsas_datasets_cleaned[2]) - np.min(hsas_datasets_cleaned[2])))
        normalised_probe_3 = ((hsas_datasets_cleaned[3] - np.mean(hsas_datasets_cleaned[3])) /
                              (np.max(hsas_datasets_cleaned[3]) - np.min(hsas_datasets_cleaned[3])))

        seconds_time_hobos = [(th - start_datetime).total_seconds() for th in time_step_milou_lights[:61]]

        # normalise all data
        common_indices_hsa_hobos = np.where((seconds_time_hobos >= np.min(seconds_hsa)) &
                                            (seconds_time_hobos <= np.max(seconds_hsa)))[0]

        subset_milou_lights = mean_milou_lights[:61][common_indices_hsa_hobos]
        min_subset_milou_lights = np.min(subset_milou_lights)
        max_subset_milou_lights = np.max(subset_milou_lights)
        subset_no_milou_lights = mean_lights[:61][common_indices_hsa_hobos]
        min_subset_no_milou_lights = np.min(subset_no_milou_lights)
        max_subset_no_milou_lights = np.max(subset_no_milou_lights)
        test_0 = subset_milou_lights - np.mean(subset_milou_lights)
        test_1 = subset_no_milou_lights - np.mean(subset_no_milou_lights)
        test_2 = np.max(subset_milou_lights) - np.min(subset_milou_lights)
        test_3 = np.max(subset_no_milou_lights) - np.min(subset_no_milou_lights)

        normalised_milou_lights = test_0 / test_2
        normalised_no_milou_lights = test_1 / test_3

        # Fit polynomial regression for lights on with Milou
        model_lights_milou = np.poly1d(np.polyfit(np.array(seconds_time_hobos)[common_indices_hsa_hobos],
                                                  normalised_milou_lights, 2))  # 3rd degree polynomial

        # Fit polynomial regression for lights on without Milou
        model_lights_no_milou = np.poly1d(np.polyfit(np.array(seconds_time_hobos)[common_indices_hsa_hobos],
                                                     normalised_no_milou_lights, 2))  # 3rd degree polynomial

        # Fit polynomial regression for the probes
        model_probe_1 = np.poly1d(
            np.polyfit(hsas_datasets_cleaned[0], normalised_probe_1, 2))  # 3rd degree polynomial
        model_probe_2 = np.poly1d(
            np.polyfit(hsas_datasets_cleaned[0], normalised_probe_2, 2))  # 3rd degree polynomial
        model_probe_3 = np.poly1d(
            np.polyfit(hsas_datasets_cleaned[0], normalised_probe_3, 2))  # 3rd degree polynomial

        # Generate smooth curves for plotting
        x_hobos_smooth = np.linspace(min(np.array(seconds_time_hobos)[common_indices_hsa_hobos]),
                                     max(np.array(seconds_time_hobos)[common_indices_hsa_hobos]), 500)
        x_hsa_smooth = np.linspace(min(hsas_datasets_cleaned[0]), max(hsas_datasets_cleaned[0]), 500)
        y_hobos_smooth_lights_milou = model_lights_milou(x_hsa_smooth)
        y_hobos_smooth_lights_no_milou = model_lights_no_milou(x_hsa_smooth)

        x_hsa_hours = (x_hsa_smooth - x_hsa_smooth[0]) / 3600
        y_hsa_smooth_1 = model_probe_1(x_hsa_smooth)
        y_hsa_smooth_2 = model_probe_2(x_hsa_smooth)
        y_hsa_smooth_3 = model_probe_3(x_hsa_smooth)

        x_hobos_hours = (x_hobos_smooth - x_hsa_smooth[0]) / 3600

        stat_w1, p_w1 = stats.wilcoxon(y_hsa_smooth_1, y_hobos_smooth_lights_no_milou)
        print("p_w1 ", p_w1)

        stat_w2, p_w2 = stats.wilcoxon(y_hsa_smooth_2, y_hobos_smooth_lights_no_milou)
        print("p_w2 ", p_w2)

        stat_w3, p_w3 = stats.wilcoxon(y_hsa_smooth_3, y_hobos_smooth_lights_no_milou)
        print("p_w3 ", p_w3)

        # Dataset1
        plt.plot(x_hsa_hours, y_hobos_smooth_lights_milou, label="HOBOs with Milou",
                 color=sns.color_palette()[2])

        # Dataset2
        plt.plot(x_hsa_hours, y_hobos_smooth_lights_no_milou, label="HOBOs no Milou",
                 linestyle="dashdot", color=sns.color_palette()[1])

        # Dataset3
        plt.plot(x_hsa_hours, y_hsa_smooth_1, label="Probe 1", color=sns.color_palette()[7],
                 linestyle=(0, (5, 2, 2, 2, 2, 2)))
        plt.plot(x_hsa_hours, y_hsa_smooth_2, label="Probe 2", color=sns.color_palette()[7],
                 linestyle=(0, (3, 5, 1, 5)))
        plt.plot(x_hsa_hours, y_hsa_smooth_3, label="Probe 3", color=sns.color_palette()[7],
                 linestyle=":")

        plt.xlabel("Time [hours]")
        plt.ylabel("$T^*$")
        plt.title("Polynomial fit")
        plt.xlim([0, 3.6])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig(f"output/T_polynomial_mean_norm_font.pdf", bbox_inches='tight', dpi=300)
        plt.show()
