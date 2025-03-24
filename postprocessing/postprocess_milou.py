import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from collections import defaultdict
from scipy.stats import median_abs_deviation

from bs4 import BeautifulSoup


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
    new_path = "/usr/local/texlive/2023/bin/universal-darwin"
    os.environ["PATH"] = os.environ.get("PATH", "") + f":{new_path}" if new_path not in os.environ.get("PATH", "") else \
    os.environ["PATH"]
    name_points = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    sns.set_palette("colorblind")
    linestyle_list = ["-", "--", "-.", ":"]
    probes_dict = {'54T33:849': 'probe 1',
                   '54T33:850': 'probe 2',
                   '54T33:851': 'probe 3'}

    # Uncomment and comment "test_speed_path" for the different Setups
    test_path = "data/setup_2_HSA_no_fans_01"  # dataset for Setup 2 in paper
    # test_path = "data/setup_3_HSA_2_fans"  # dataset for Setup 3 in paper

    dict_points = extract_point_data(test_path)

    # Sort points chronologically
    sorted_points = sorted(dict_points.keys())

    total_number_of_points = len(name_points)
    rounds_number = 4

    # Choose which field to postprocess, "U" -> velocity magnitude, "T" -> temperature
    field = "T"  # "U", "T"

    dict_measurements = {}

    for probe in probes_dict.values():
        dict_measurements[probe] = {}
        print(probe)
        field_10_points = []
        for i in range(total_number_of_points):
            dict_measurements[probe][name_points[i]] = {"U": [], "T": []}
            for j in range(rounds_number):
                U_dantec = dict_points[sorted_points[i + (j * total_number_of_points)]][probe]["Velocity magnitude"]
                T_dantec = dict_points[sorted_points[i + (j * total_number_of_points)]][probe]["Temperature"]
                U_outliers = detect_outliers(U_dantec)
                T_outliers = detect_outliers(T_dantec)
                outlier_indices = np.union1d(U_outliers, T_outliers)
                U_with_nan = U_dantec.copy()
                T_with_nan = T_dantec.copy()
                U_with_nan[outlier_indices] = np.nan
                T_with_nan[outlier_indices] = np.nan
                dict_measurements[probe][name_points[i]]["U"].append(U_with_nan)
                dict_measurements[probe][name_points[i]]["T"].append(T_with_nan)
            if field == "T":
                data_00 = np.round(dict_measurements[probe][name_points[i]][field][0], 2)
                data_01 = np.round(dict_measurements[probe][name_points[i]][field][1], 2)
                data_02 = np.round(dict_measurements[probe][name_points[i]][field][2], 2)
                data_03 = np.round(dict_measurements[probe][name_points[i]][field][3], 2)
                datasets = [data_00, data_01, data_02, data_03]

                # Find valid indices
                valid_mask = ~np.isnan(datasets).any(axis=0)

                # Remove NaN values from all dataset
                datasets_cleaned = np.array([data[valid_mask] for data in datasets])
                flattened_datatsets_cleaned = datasets_cleaned.flatten()
                field_10_points.append(flattened_datatsets_cleaned)
            elif field == "U":
                # plot the U repeatability test
                fixPlot(thickness=1.2, fontsize=18, markersize=8, labelsize=18, texuse=True, tickSize=5)

                data_00 = np.round(dict_measurements[probe][name_points[i]][field][0], 2)
                data_01 = np.round(dict_measurements[probe][name_points[i]][field][1], 2)
                data_02 = np.round(dict_measurements[probe][name_points[i]][field][2], 2)
                data_03 = np.round(dict_measurements[probe][name_points[i]][field][3], 2)
                datasets = [data_00, data_01, data_02, data_03]

                # Find valid indices
                valid_mask = ~np.isnan(datasets).any(axis=0)

                # Remove NaN values from all dataset
                datasets_cleaned = np.array([data[valid_mask] for data in datasets])
                flattened_datatsets_cleaned = datasets_cleaned.flatten()
                field_10_points.append(flattened_datatsets_cleaned)

                normality_check_0 = stats.kstest(datasets_cleaned[0], 'norm')
                normality_check_1 = stats.kstest(datasets_cleaned[1], 'norm')
                normality_check_2 = stats.kstest(datasets_cleaned[2], 'norm')
                normality_check_3 = stats.kstest(datasets_cleaned[3], 'norm')
                if (normality_check_0[1] > 0.01 or normality_check_1[1] > 0.01 or normality_check_2[1] > 0.01
                        or normality_check_3[1] > 0.01):
                    print("Bingpot")

                stat_f, p_f = stats.friedmanchisquare(datasets_cleaned[0], datasets_cleaned[1],
                                                      datasets_cleaned[2], datasets_cleaned[3])
                print("Friedman ", p_f)

                mean_cleaned_0 = np.mean(datasets_cleaned[0])
                median_cleaned_0 = np.median(datasets_cleaned[0])
                std_cleaned_0 = np.std(datasets_cleaned[0])
                mad_cleaned_0 = median_abs_deviation(datasets_cleaned[0])
                print("Round 1 ", mean_cleaned_0, std_cleaned_0, median_cleaned_0)
                normalised_dataset_0 = ((datasets_cleaned[0] - mean_cleaned_0) /
                                        (np.max(datasets_cleaned[0]) - np.min(datasets_cleaned[0])))

                mean_cleaned_1 = np.mean(datasets_cleaned[1])
                median_cleaned_1 = np.median(datasets_cleaned[1])
                std_cleaned_1 = np.std(datasets_cleaned[1])
                mad_cleaned_1 = median_abs_deviation(datasets_cleaned[1])
                print("Round 2 ", mean_cleaned_1, std_cleaned_1, median_cleaned_1)
                normalised_dataset_1 = ((datasets_cleaned[1] - mean_cleaned_1) /
                                        (np.max(datasets_cleaned[1]) - np.min(datasets_cleaned[1])))

                mean_cleaned_2 = np.mean(datasets_cleaned[2])
                median_cleaned_2 = np.median(datasets_cleaned[2])
                std_cleaned_2 = np.std(datasets_cleaned[2])
                mad_cleaned_2 = median_abs_deviation(datasets_cleaned[2])
                print("Round 3 ", mean_cleaned_2, std_cleaned_2, median_cleaned_2)
                normalised_dataset_2 = ((datasets_cleaned[2] - mean_cleaned_2) /
                                        (np.max(datasets_cleaned[2]) - np.min(datasets_cleaned[2])))

                mean_cleaned_3 = np.mean(datasets_cleaned[3])
                median_cleaned_3 = np.median(datasets_cleaned[3])
                std_cleaned_3 = np.std(datasets_cleaned[3])
                mad_cleaned_3 = median_abs_deviation(datasets_cleaned[3])
                print("Round 4 ", mean_cleaned_3, std_cleaned_3, median_cleaned_3)
                normalised_dataset_3 = ((datasets_cleaned[3] - mean_cleaned_3) /
                                        (np.max(datasets_cleaned[3]) - np.min(datasets_cleaned[3])))

                normalised_normality_check_0 = stats.kstest(normalised_dataset_0, 'norm')
                normalised_normality_check_1 = stats.kstest(normalised_dataset_1, 'norm')
                normalised_normality_check_2 = stats.kstest(normalised_dataset_2, 'norm')
                normalised_normality_check_3 = stats.kstest(normalised_dataset_3, 'norm')
                if (normalised_normality_check_0[1] > 0.05 or normalised_normality_check_1[1] > 0.05 or
                        normalised_normality_check_2[1] > 0.05 or normalised_normality_check_3[1] > 0.05):
                    print("Here")

                stat_f_norm, p_f_norm = stats.friedmanchisquare(np.round(normalised_dataset_0, 2),
                                                                np.round(normalised_dataset_1, 2),
                                                                np.round(normalised_dataset_2, 2),
                                                                np.round(normalised_dataset_3, 2))

                palette = sns.color_palette("colorblind", 4)
                plt.figure(figsize=(5, 5))
                box = plt.boxplot([normalised_dataset_0, normalised_dataset_1, normalised_dataset_2,
                                   normalised_dataset_3], patch_artist=True, showmeans=True,  # Enable mean display
                                   meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black",
                                              "markersize": 8})  # Enable colored boxes
                # Apply colorblind-friendly colors
                for patch, color in zip(box['boxes'], palette):
                    patch.set_facecolor((*color, 0.7))  # Convert to RGBA by adding alpha

                for whisker, cap, median in zip(box['whiskers'], box['caps'], box['medians']):
                    whisker.set(color="black", linewidth=1.5)
                    cap.set(color="black", linewidth=1.5)
                    median.set(color="black", linewidth=2)

                if p_f_norm < 0.01:
                    plt.text(3.0, -0.74, f"p = {np.round(p_f_norm, 2)}", color="red")
                else:
                    plt.text(3.0, -0.74, f"p = {np.round(p_f_norm, 2)}")

                # Customize labels
                plt.xlabel("Lap")
                plt.ylabel("$U^*$")
                plt.title(f"{probe} {name_points[i]}")

                # Show grid for better readability
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.ylim([-0.8, 0.9])
                plt.tight_layout()

                # Show the plot
                # plt.savefig(f"output/U_{probe}_{name_points[i]}_box_plot_fans_high_paper_no_fan_smaller.svg", bbox_inches='tight',
                #             dpi=300)
                plt.show()

        if field == "U":
            # plot the non-normalised values of U
            palette = sns.color_palette("rocket", 10)
            plt.figure(figsize=(7, 5))
            box = plt.boxplot(field_10_points, patch_artist=True, showmeans=True,  # Enable mean display
                              meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black",
                                         "markersize": 6})  # Enable colored boxes
            # Apply colorblind-friendly colors
            for patch, color in zip(box['boxes'], palette):
                patch.set_facecolor((*color, 0.7))  # Convert to RGBA by adding alpha

            for whisker, cap, median in zip(box['whiskers'], box['caps'], box['medians']):
                whisker.set(color="black", linewidth=1.5)
                cap.set(color="black", linewidth=1.5)
                median.set(color="black", linewidth=1.5)

            # Customize labels
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name_points)
            plt.xlabel("Points")
            plt.ylabel("U [m/s]")
            plt.title(f"{probe}")

            # Show grid for better readability
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            # plt.ylim([0, 5.5])  # For Setup 3
            plt.ylim([-0.05, 0.35])  # For Setup 2
            plt.tight_layout()

            # Show the plot
            # plt.savefig(f"output/U_{probe}_box_plot_all_points_no_fans.svg", bbox_inches='tight', dpi=300)
            plt.show()

        if field == "T":
            # plot the non-normalised values of T
            fixPlot(thickness=1.2, fontsize=18, markersize=8, labelsize=18, texuse=True, tickSize=5)
            palette = sns.color_palette("rocket", 10)
            plt.figure(figsize=(7, 5))
            box = plt.boxplot(field_10_points, patch_artist=True, showmeans=True,  # Enable mean display
                              meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black",
                                         "markersize": 6})  # Enable colored boxes
            # Apply colorblind-friendly colors
            for patch, color in zip(box['boxes'], palette):
                patch.set_facecolor((*color, 0.7))  # Convert to RGBA by adding alpha

            for whisker, cap, median in zip(box['whiskers'], box['caps'], box['medians']):
                whisker.set(color="black", linewidth=1.5)
                cap.set(color="black", linewidth=1.5)
                median.set(color="black", linewidth=1.5)

            # Customize labels
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name_points)
            plt.xlabel("Points")
            plt.ylabel("T [$^\circ$C]")
            plt.title(f"{probe}")

            # Show grid for better readability
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.ylim([21.2, 22.3])
            # plt.ylim([-2.1, 2.1])
            plt.tight_layout()

            # Show the plot
            # plt.savefig(f"output/T_{probe}_box_plot_all_points_buoyancy_smaller_font.svg", bbox_inches='tight', dpi=300)
            plt.show()
    if field == "T" and test_path == "data/setup_2_HSA_no_fans_01":
        # plot repeatability of T over 2 days
        # additional datasets for repeatability test
        dataset2 = "data/setup_2_HSA_no_fans_00"

        hsa_data_rounds = [dict_points, extract_point_data(dataset2)]

        # New data structure
        dict_normalised_temperature_hsas = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Reorganize data
        for round_index, timestamps in enumerate(hsa_data_rounds):
            sorted_start_time = sorted(timestamps.keys())
            counter = 0
            for idx_point, start_time in enumerate(sorted_start_time):
                if idx_point == 25 and round_index == 1:
                    pass
                elif idx_point > 25 and round_index == 1:
                    probes = timestamps[start_time]
                    point_index = (idx_point - 1) % 10  # Compute point index
                    for probe, value in probes.items():
                        U_outliers = detect_outliers(value["Velocity magnitude"])
                        T_outliers = detect_outliers(value["Temperature"])
                        outlier_indices = np.union1d(U_outliers, T_outliers)
                        U_with_nan = value["Velocity magnitude"].copy()
                        T_with_nan = value["Temperature"].copy()
                        U_with_nan[outlier_indices] = np.nan
                        T_with_nan[outlier_indices] = np.nan
                        dict_normalised_temperature_hsas[probe][name_points[point_index]][round_index].extend(
                            T_with_nan)
                else:
                    probes = timestamps[start_time]
                    point_index = idx_point % 10  # Compute point index
                    for probe, value in probes.items():
                        U_outliers = detect_outliers(value["Velocity magnitude"])
                        T_outliers = detect_outliers(value["Temperature"])
                        outlier_indices = np.union1d(U_outliers, T_outliers)
                        U_with_nan = value["Velocity magnitude"].copy()
                        T_with_nan = value["Temperature"].copy()
                        U_with_nan[outlier_indices] = np.nan
                        T_with_nan[outlier_indices] = np.nan
                        dict_normalised_temperature_hsas[probe][name_points[point_index]][round_index].extend(
                            T_with_nan)

        # Convert defaultdict to a normal dict
        dict_normalised_temperature_hsas = {probe: dict(points)
                                            for probe, points in dict_normalised_temperature_hsas.items()}

        test_00 = np.nanmean(dict_normalised_temperature_hsas['probe 3']['D'][0])
        test_01 = np.nanmean(dict_normalised_temperature_hsas['probe 3']['D'][1])

        for probe, points in dict_normalised_temperature_hsas.items():
            for point, rounds in points.items():
                probes_data = [rounds[0], rounds[1]]
                # clean data from NaN values
                valid_mask = ~np.isnan(probes_data).any(axis=0)
                probes_data_cleaned = np.array([np.array(data)[valid_mask] for data in probes_data])
                # normalise datatsets
                normalised_round_0 = ((probes_data_cleaned[0] - np.mean(probes_data_cleaned[0])) /
                                      (np.max(probes_data_cleaned[0]) - np.min(probes_data_cleaned[0])))
                normalised_round_1 = ((probes_data_cleaned[1] - np.mean(probes_data_cleaned[1])) /
                                      (np.max(probes_data_cleaned[1]) - np.min(probes_data_cleaned[1])))

                print(probe, point)
                difference2 = np.round((normalised_round_0 - normalised_round_1), 1)
                stat_w1, p_w1 = stats.wilcoxon(difference2)
                print("p_w1 ", p_w1)

                palette = sns.color_palette("colorblind", 6)
                plt.figure(figsize=(4, 5))
                box = plt.boxplot([normalised_round_0, normalised_round_1], patch_artist=True, showmeans=True,
                                  # Enable mean display
                                  meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black",
                                             "markersize": 8}, widths=0.3)  # Enable colored boxes
                selected_colors = [palette[4], palette[5]]
                # Apply colorblind-friendly colors
                for patch, color in zip(box['boxes'], selected_colors):
                    patch.set_facecolor((*color, 0.7))  # Convert to RGBA by adding alpha

                for whisker, cap, median in zip(box['whiskers'], box['caps'], box['medians']):
                    whisker.set(color="black", linewidth=1.5)
                    cap.set(color="black", linewidth=1.5)
                    median.set(color="black", linewidth=2)

                if p_w1 < 0.01:
                    plt.text(1.15, -0.67, f"p = {np.round(p_w1, 2)}", color="red")
                else:
                    plt.text(1.15, -0.67, f"p = {np.round(p_w1, 2)}")

                # Customize labels
                plt.xlabel("Day")
                plt.ylabel("$T^*$")
                plt.title(f"{probe} {point}")

                # Show grid for better readability
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.ylim([-0.7, 0.6])
                plt.tight_layout()

                # Show the plot
                # plt.savefig(f"output/T_{probe}_{point}_box_plot_repeatability.svg", bbox_inches='tight', dpi=300)
                plt.show()
