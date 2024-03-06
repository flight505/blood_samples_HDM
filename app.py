import os

import pandas as pd
import streamlit as st

# Check if the user is authenticated
from auth import check_password, initialize_session_state
from chart_utils import create_component_plot, create_radar_chart

# Initialize session state at the very beginning
initialize_session_state()

# Move the set_page_config call here, ensuring it's the first Streamlit command
if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="Events Detection",
        page_icon="🔁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state["page_config_set"] = True

# Check if the user is authenticated
if not check_password():
    st.stop()


@st.cache_data
def load_csv_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)

        if "nohpo_nr" in data.columns:
            data["nohpo_nr"] = data["nohpo_nr"].astype(int)

        date_columns = ["diagdate", "sample_time", "birthdate", "mtx_inf_datetime"]
        for col in date_columns:
            if col in data.columns:
                if col == "birthdate":
                    data[col] = pd.to_datetime(data[col], format="%d-%m-%Y")
                elif col == "sample_time":
                    data[col] = pd.to_datetime(data[col], format="%Y-%m-%d %H:%M:%S")
                elif col == "diagdate":
                    try:
                        data[col] = pd.to_datetime(data[col], format="%Y-%m-%d")
                    except ValueError:
                        data[col] = pd.to_datetime(
                            data[col], format="%Y-%m-%d %H:%M:%S"
                        )
                else:
                    data[col] = pd.to_datetime(data[col])

        return data
    else:
        print("The file does not exist")


def main():

    # df_all = load_csv_data("data/df_all_filtered.csv")
    df_mtx = load_csv_data("data/df_mtx_filtered.csv")
    df_blood = load_csv_data("data/df_blood_filtered.csv")

    # def display_dataframe(df_dict):
    #     selected_df_name = st.selectbox("Select a DataFrame", list(df_dict.keys()))
    #     st.dataframe(df_dict[selected_df_name].head(2000), height=200)

    # df_dict = {"df_all": df_all, "df_mtx": df_mtx, "df_blood": df_blood}
    # with st.expander("Display DataFrames"):
    #     display_dataframe(df_dict)

    def interpolate_crossing_time(ts1, val1, ts2, val2, threshold):
        """Interpolates the exact timestamp when the value crosses the threshold."""
        if val1 == val2:
            return ts1 if val1 <= threshold else ts2

        fraction = (threshold - val1) / (val2 - val1)
        crossing_time = ts1 + (ts2 - ts1) * fraction
        return crossing_time

    def find_cycles_with_interpolation(series, infno_series, threshold, mode):
        """
        Identifies cycles in a time series where the values cross a specified threshold.

        Parameters:
        series (Series): Pandas Series containing the time series data. The index should be datetime.
        threshold (float, optional): The threshold value to identify the start and end of a cycle. If None, all values are considered.
        mode (str, optional): The mode to identify cycles. Can be "drop_below" (default) or "rise_above". In "drop_below" mode, a cycle starts when the value drops below the threshold and ends when it rises above. In "rise_above" mode, a cycle starts when the value rises above the threshold and ends when it drops below.

        Returns:
        list: A list of dictionaries, where each dictionary represents a cycle and contains the start time ('t_start'), end time ('t_end'), time of extreme value ('t_extreme'), duration ('duration'), and degree of change ('doc').
        """

        cycles = []
        in_cycle = False
        start_time, end_time, min_time, max_time = None, None, None, None
        min_val, max_val = float("inf"), float("-inf")
        start_infno = None  # Initialize start_infno here

        prev_time, prev_val = None, None

        # If the series has only one data point, skip processing
        if len(series) <= 1:
            return cycles

        # Initializing edge case handling based on mode
        if mode == "drop_below" and series.iloc[0] < threshold:
            start_time = series.index[0]
            in_cycle = True
        elif mode == "rise_above" and series.iloc[0] > threshold:
            start_time = series.index[0]
            in_cycle = True

        for idx, (time, value) in enumerate(series.items()):
            infno = infno_series.iloc[idx]
            # Update next_time and next_val for the current iteration
            if idx + 1 < len(series):
                next_time, next_val = series.index[idx + 1], series.iloc[idx + 1]
            else:
                next_time, next_val = None, None

            # Handling values exactly on the threshold
            if value == threshold:
                if prev_val is not None and next_val is not None:
                    # Determine the direction of crossing based on previous and next values
                    if mode == "drop_below":
                        if prev_val > threshold and next_val < threshold:
                            value = (
                                threshold - 0.0001
                            )  # Adjust value slightly below threshold
                        elif prev_val < threshold and next_val > threshold:
                            value = (
                                threshold + 0.0001
                            )  # Adjust value slightly above threshold
                    elif mode == "rise_above":
                        if prev_val < threshold and next_val > threshold:
                            value = (
                                threshold + 0.0001
                            )  # Adjust value slightly above threshold
                        elif prev_val > threshold and next_val < threshold:
                            value = (
                                threshold - 0.0001
                            )  # Adjust value slightly below threshold

            if prev_time is not None:
                if mode == "drop_below":
                    crossing_below = prev_val > threshold and value < threshold
                    crossing_above = prev_val < threshold and value > threshold
                else:  # mode == "rise_above"
                    crossing_below = prev_val < threshold and value > threshold
                    crossing_above = prev_val > threshold and value < threshold

                if crossing_below:
                    interpolated_time = interpolate_crossing_time(
                        prev_time, prev_val, time, value, threshold
                    )
                    if not in_cycle:
                        start_time = interpolated_time
                        in_cycle = True
                        min_val = value
                        min_time = time
                        start_infno = infno

                elif crossing_above:
                    interpolated_time = interpolate_crossing_time(
                        prev_time, prev_val, time, value, threshold
                    )
                    if in_cycle:
                        end_time = interpolated_time
                        if mode == "drop_below":
                            extreme_val, extreme_time = min_val, min_time
                        else:  # mode == "rise_above"
                            extreme_val, extreme_time = max_val, max_time

                        cycles.append(
                            {
                                "t_start": start_time,
                                "t_end": end_time,
                                "t_extreme": extreme_time,
                                "duration": (end_time - start_time).total_seconds()
                                / 3600,  # in hours
                                "doc": extreme_val - threshold,
                                "start_infno": start_infno,
                                "end_infno": infno,
                            }
                        )
                        in_cycle = False
                        start_time, end_time, min_time, max_time = (
                            None,
                            None,
                            None,
                            None,
                        )
                        min_val, max_val = float("inf"), float("-inf")

            if in_cycle:
                if mode == "drop_below" and value < min_val:
                    min_val = value
                    min_time = time
                elif mode == "rise_above" and value > max_val:
                    max_val = value
                    max_time = time

            prev_time, prev_val = time, value

        # Handle edge case where series ends in a cycle
        if in_cycle:
            end_time = series.index[-1]
            if mode == "drop_below":
                extreme_val, extreme_time = min_val, min_time
            else:  # mode == "rise_above"
                extreme_val, extreme_time = max_val, max_time

            cycles.append(
                {
                    "t_start": start_time,
                    "t_end": end_time,
                    "t_extreme": extreme_time,
                    "duration": (end_time - start_time).total_seconds()
                    / 3600,  # in hours
                    "doc": extreme_val - threshold,
                    "start_infno": start_infno,
                    "end_infno": infno,
                }
            )

        return cycles

    def identify_cycles(
        df: pd.DataFrame, component=None, threshold=None, mode="drop_below"
    ) -> pd.DataFrame:
        """
        Identifies cycles in the given DataFrame based on a specified component and threshold.

        Parameters:
        df (DataFrame): DataFrame containing the data. It should have columns 'component', 'nopho_nr', 'sample_time', and 'reply_num'.
        component (str, optional): The name of the component to identify cycles for. If None, all components are considered.
        threshold (float, optional): The threshold value to identify the start and end of a cycle. If None, all values are considered.
        mode (str, optional): The mode to identify cycles. Can be "drop_below" (default) or "rise_above". In "drop_below" mode, a cycle starts when the value drops below the threshold and ends when it rises above. In "rise_above" mode, a cycle starts when the value rises above the threshold and ends when it drops below.

        Returns:
        DataFrame: A DataFrame with the identified cycles. Each row represents a cycle and contains the 'nopho_nr', start time ('t_start'), end time ('t_end'), duration ('duration'), and degree of change ('doc').
        """
        df_component = df[df["component"] == component]

        all_cycles_interpolated = []
        for nopho_nr, group in df_component.groupby("nopho_nr"):
            cycles = find_cycles_with_interpolation(
                group.set_index("sample_time")["reply_num"],
                group.set_index("sample_time")["infno"],  # Pass the infno series
                threshold,
                mode,
            )
            for cycle in cycles:
                cycle["nopho_nr"] = nopho_nr
            all_cycles_interpolated.extend(cycles)

        return pd.DataFrame(all_cycles_interpolated)

    def select_nopho_nr(df_blood):
        unique_nopho_nr = df_blood["nopho_nr"].unique()
        selected_nopho_nr = st.selectbox("Select Patient ID", unique_nopho_nr)
        return selected_nopho_nr

    def select_component(df_blood):
        components = df_blood["component"].unique().tolist()
        selected_component = st.selectbox(
            "Select a component",
            components,
            index=components.index("Neutrophilocytes;B"),
        )
        return selected_component

    def get_threshold():
        threshold = st.text_input("Enter a threshold value", "0.5")
        return threshold

    def get_toggle():
        st.markdown(
            "<p style='text-align: left; color: black; font-size: 14px;'>Select the mode to identify cycles</p>",
            unsafe_allow_html=True,
        )
        toggle = st.toggle(
            "detection mode",
            key="mode_toggle",
            value=False,
            help="The mode to identify cycles. Can be 'drop_below' (default) or 'rise_above'. In 'drop_below' mode, a cycle starts when the value drops below the threshold and ends when it rises above. In 'rise_above' mode, a cycle starts when the value rises above the threshold and ends when it drops below.",
        )
        if toggle:
            st.markdown(
                "<p style='text-align: left; color: green; font-size: 10px;'>rise_above threshold activated</p>",
                unsafe_allow_html=True,
            )
            return "rise_above"
        else:
            st.markdown(
                "<p style='text-align: left; color: green; font-size: 10px;'>drop_below threshold activated</p>",
                unsafe_allow_html=True,
            )
            return "drop_below"

    with st.form(key="my_form"):
        select_container = st.container()
        with select_container:
            select_nopho_col, select_comp_col, threshold_col, toggle_col = st.columns(4)
            with select_nopho_col:
                selected_nopho_nr = select_nopho_nr(df_blood)
            with select_comp_col:
                selected_component = select_component(df_blood)
            with threshold_col:
                threshold_value = get_threshold()
            with toggle_col:
                mode = get_toggle()

        submit_button = st.form_submit_button(label="Submit")
        if submit_button:
            selected_nopho_nr = int(selected_nopho_nr)  # Convert to int
            selected_component = str(selected_component)  # Convert to str
            threshold_value = float(threshold_value)  # Convert to float
            mode = str(mode)
            df_cycles = identify_cycles(
                df_blood,
                component=selected_component,
                threshold=threshold_value,
                mode=mode,
            )
            cycles_data_nopho_nr = df_cycles[df_cycles["nopho_nr"] == selected_nopho_nr]
            df_blood_component = df_blood[df_blood["component"] == selected_component]
            df_blood_component_nopho_nr = df_blood_component[
                df_blood_component["nopho_nr"] == selected_nopho_nr
            ]
        else:
            st.stop()

    # Disply the plots in a container side by side
    data_container = st.container(border=True)
    comp_plot, radar_plot = data_container.columns(2)

    with comp_plot:
        create_component_plot(
            data=df_blood_component_nopho_nr,
            cycles_data=cycles_data_nopho_nr,
            df_mtx=df_mtx,
            component=selected_component,
            threshold=threshold_value,
            selected_nopho_nr=selected_nopho_nr,
        )
    with radar_plot:
        r_values_all_patients, single_patient_infno_counts = create_radar_chart(
            df_cycles=df_cycles,
            cycles_data_nopho_nr=cycles_data_nopho_nr,
        )

    # Calculate statistics for the single patient
    single_patient_cycles = len(cycles_data_nopho_nr)
    single_patient_avg_duration = cycles_data_nopho_nr["duration"].mean()
    single_patient_avg_doc = cycles_data_nopho_nr["doc"].mean()

    # Calculate statistics for all patients
    all_patients_cycles = len(df_cycles)
    all_patients_avg_duration = df_cycles["duration"].mean()
    all_patients_avg_doc = df_cycles["doc"].mean()

    # Create a DataFrame that contains these statistics
    data = {
        "No. Events": [single_patient_cycles, all_patients_cycles],
        "Avg. Duration": [single_patient_avg_duration, all_patients_avg_duration],
        "Avg. DOC": [single_patient_avg_doc, all_patients_avg_doc],
    }

    # Add r values for each infno
    for i, r_value in enumerate(r_values_all_patients):
        data[f"{'inf' + str(i) if i != 0 else 'Induction'}"] = [
            single_patient_infno_counts[i],
            r_value,
        ]
    statistics_df = pd.DataFrame(
        data, index=[f"Patient {selected_nopho_nr}", "All Patients"]
    )
    # Print the DataFrame
    st.dataframe(statistics_df)
    with st.expander("What is DOC?"):
        st.write(
            """
            In the context of the provided code, `doc` stands for "Degree of Change". It represents the difference between the extreme value (`extreme_val`) and the threshold. 

            If the mode is "drop_below", the extreme value is the minimum value in the cycle, and `doc` represents how much this minimum value is below the threshold. 

            If the mode is "rise_above", the extreme value is the maximum value in the cycle, and `doc` represents how much this maximum value is above the threshold. 

            This measure can be useful to understand the magnitude of the change in the data during each cycle.
            """
        )


if __name__ == "__main__":
    main()
