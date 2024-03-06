import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def create_component_plot(
    data: pd.DataFrame,
    cycles_data: pd.DataFrame,
    df_mtx: pd.DataFrame,
    component,
    threshold,
    selected_nopho_nr,
) -> None:
    """
    Creates a Plotly scatter plot of components data.

    Parameters:
    data (DataFrame): DataFrame containing the original data. It should have columns 'sample_time' and 'reply_num'.
    cycles_data (DataFrame): DataFrame containing the cycle start and end times. It should have columns 't_start', 't_end', and 'duration'.
    df_mtx (DataFrame): DataFrame containing the mtx data. It should have columns 'nopho_nr' and 'mtx_inf_datetime'.
    component (str): The name of the component to be displayed in the y-axis title.
    threshold (float, optional): The threshold value to be displayed as a horizontal line in the plot. Defaults to 0.5.
    select_nopho_nr (function, optional): Function to select a 'nopho_nr'. Defaults to select_nopho_nr function.

    Returns:
    None: The function displays the plot in a Streamlit app and does not return anything.
    """
    fig = go.Figure()

    # Add the original data to the plot
    fig.add_trace(
        go.Scatter(
            x=data["sample_time"],
            y=data["reply_num"],
            mode="lines+markers",
            name=f"{component} Data",  # Update name here
            line=dict(),
        )
    )

    # Add cycle start and end points for nopho_nr 2011122 from df_cycles_neutrophilocytes
    fig.add_trace(
        go.Scatter(
            x=cycles_data["t_start"],
            y=[threshold] * len(cycles_data),
            mode="markers",
            name=f"{component} Cycle Start Points",  # Update name here
            marker=dict(size=10, symbol="triangle-up", color="green"),
            text=cycles_data["duration"].apply(lambda x: f"Duration: {x:.2f} hours"),
            hoverinfo="text+x+y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cycles_data["t_end"],
            y=[threshold] * len(cycles_data),
            mode="markers",
            name=f"{component} Cycle End Points",  # Update name here
            marker=dict(size=10, symbol="triangle-down", color="red"),
            text=cycles_data["duration"].apply(lambda x: f"Duration: {x:.2f} hours"),
            hoverinfo="text+x+y",
        )
    )

    # Add the threshold line to the plot
    fig.add_shape(
        type="line",
        x0=data["sample_time"].min(),
        x1=data["sample_time"].max(),
        y0=threshold,
        y1=threshold,
        line=dict(color="orange", dash="dash"),
    )

    # Filter df_mtx for the selected patient
    df_mtx_filtered = df_mtx[df_mtx["nopho_nr"] == selected_nopho_nr]

    # Add infno annotations
    for _, row in df_mtx_filtered.iterrows():
        fig.add_annotation(
            x=row["mtx_inf_datetime"],
            y=threshold - (0.3 * threshold),
            text=f"inf: {row['infno']}",
            showarrow=False,
            font=dict(size=10),
            bgcolor="white",
            bordercolor="black",
            hovertext=f"{row['mtx_inf_datetime']}",
        )

    # Get diagnosis date for the patient
    diag_date = data[data["nopho_nr"] == selected_nopho_nr]["diagdate"].iloc[0]

    # Add diagnosis date annotation as a text box
    fig.add_annotation(
        x=diag_date,
        y=threshold - (0.3 * threshold),
        text="diag",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
        bordercolor="black",
        hovertext=f"{diag_date}",
    )

    # Update legend to show custom names
    fig.update_layout(
        legend=dict(
            traceorder="normal",
            itemsizing="constant",
            font=dict(
                size=12,
            ),
            y=-0.2,  # Add this line
            x=0.5,  # Add this line
            xanchor="center",  # Add this line
            yanchor="top",  # Add this line
        ),
    )

    # Adjust the layout and display settings
    fig.update_layout(
        title=f"{component} conc. for ID {selected_nopho_nr}, threshold ({threshold})",
        # xaxis_title="Sample Time",
        yaxis_title=f"{component} conc.",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, range=[0, max(data["reply_num"].max(), 1)]),
        height=500,
    )

    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def create_radar_chart(df_cycles: pd.DataFrame, cycles_data_nopho_nr):
    """
    Creates a radar chart of patient events.

    This function takes two dataframes as input, processes them to count the number of events
    for each infusion number (infno) for all patients and a single patient, and then creates
    a radar chart to visualize these counts.

    Parameters:
    df_cycles (pd.DataFrame): A dataframe containing the cycle data for all patients.
    cycles_data_nopho_nr (pd.DataFrame): A dataframe containing the cycle data for a single patient.

    Returns:
    r_values_all_patients (list): A list of average counts of events for each infno for all patients.
    single_patient_infno_counts (pd.Series): A series of counts of events for each infno for a single patient.
    """
    # Convert 'start_infno' and 'end_infno' to integers, with NaNs replaced by 0
    df_cycles = df_cycles.astype({"start_infno": "Int64", "end_infno": "Int64"})
    df_cycles["start_infno"].fillna(0, inplace=True)
    df_cycles["end_infno"].fillna(0, inplace=True)
    cycles_data_nopho_nr = cycles_data_nopho_nr.astype(
        {"start_infno": "Int64", "end_infno": "Int64"}
    )
    cycles_data_nopho_nr["start_infno"].fillna(0, inplace=True)
    cycles_data_nopho_nr["end_infno"].fillna(0, inplace=True)

    # Initialize counts for all infnos for all patients and single patient
    all_patient_infno_counts = pd.Series(
        0, index=np.arange(0, 10)
    )  # Include 0 for before infusion 1
    single_patient_infno_counts = pd.Series(0, index=np.arange(0, 10))

    # Increment counts for all_patient_infno_counts
    for _, row in df_cycles.iterrows():
        # Convert to int to avoid indexing issues
        start_infno = int(row["start_infno"])
        end_infno = int(row["end_infno"])
        for idx in range(start_infno, end_infno + 1):
            all_patient_infno_counts.loc[idx] += 1
    # Calculate the average per infno across all patients
    avg_cycles_all = all_patient_infno_counts[1:] / len(
        df_cycles[df_cycles["start_infno"] > 0]["nopho_nr"].unique()
    )  # Exclude 0 for the average

    # Increment counts for single_patient_infno_counts
    for _, row in cycles_data_nopho_nr.iterrows():
        start_infno = int(row["start_infno"])
        end_infno = int(row["end_infno"])
        single_patient_infno_counts.loc[start_infno : end_infno + 1] += 1

    # Create the theta categories including a category for 'Before Infno 1'
    theta_categories = ["Before Infno 1"] + [f"Infno {i}" for i in range(1, 10)]

    # Since the 'Before Infno 1' category is included in theta_categories,
    # we need to prepend a value to avg_cycles_all for the radar chart to match.
    # We'll prepend the average count of cycles before the first infno.
    before_infno_1_count = all_patient_infno_counts[0] / len(
        df_cycles["nopho_nr"].unique()
    )
    r_values_all_patients = [before_infno_1_count] + avg_cycles_all.tolist()

    # Ensure that r_values_all_patients has the same length as theta_categories
    if len(r_values_all_patients) < len(theta_categories):
        # This should not normally happen, but we add zeros just in case
        r_values_all_patients += [0] * (
            len(theta_categories) - len(r_values_all_patients)
        )

    # Initialize the radar chart
    fig = go.Figure()

    # Now use r_values_all_patients for the 'All Patients' trace
    fig.add_trace(
        go.Scatterpolar(
            r=r_values_all_patients,
            theta=theta_categories,
            fill="toself",
            name="Average Events - All Patients",
        )
    )

    # Add a trace for 'Single Patient' with their specific cycle count per infno
    fig.add_trace(
        go.Scatterpolar(
            r=single_patient_infno_counts.tolist(),  # Include the count for 'Before Infno 1'
            theta=theta_categories,
            fill="toself",
            name="Number of Events - Single Patient",
        )
    )

    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[
                    0,
                    max(
                        avg_cycles_all.max(),
                        single_patient_infno_counts.max(),
                    ),
                ],
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta_categories,
                ticktext=theta_categories,
            ),
        ),
        height=500,
        legend=dict(y=-0.7, x=0.3),  # Add this line
        title="Chart of Patient Events",
        title_x=0.5,
    )
    # Display the figure
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    return r_values_all_patients, single_patient_infno_counts
