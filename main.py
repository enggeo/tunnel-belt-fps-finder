import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Initialize variables
RPM = st.sidebar.slider('RPM', min_value=1.0, max_value=10.0, value=5.15)
buckets = 6
bucket_pass_length = 0.5
expected_ratio = bucket_pass_length/(1 - bucket_pass_length)  # Calculate expected_ratio
evaluation_time = 1200
fps = st.sidebar.slider('FPS', min_value=1, max_value=30, value=2)
image_capture_rate = 1/fps

# Initialize arrays for fps values and condition counts
fps_values = np.arange(1.0, 3, 0.01)
bucket_pass_counts = np.zeros_like(fps_values)
gap_counts = np.zeros_like(fps_values)

# Initialize arrays for RPM values and condition ratios
RPM_values = np.arange(4.5, 6, 0.01)
condition_ratios = np.zeros((len(RPM_values), len(fps_values)))

# Calculate events per minute and length of event
events_per_minute = RPM * buckets
length_of_event = 60 / events_per_minute

# Calculate length of sub-events
length_sub_event_bucket_pass = bucket_pass_length * length_of_event
length_sub_event_gap = 1 - bucket_pass_length * length_of_event

# Initialize time and condition arrays
time_array = np.arange(0, evaluation_time, image_capture_rate)
condition_array = np.zeros_like(time_array)

# Determine condition at each time point
for i, time in enumerate(time_array):
    if time % length_of_event < length_sub_event_bucket_pass:
        condition_array[i] = 1

# Calculate the total counts and ratio
total_bucket_pass = np.sum(condition_array)
total_gap = len(condition_array) - total_bucket_pass
ratio_gap_to_bucket_pass = total_gap / total_bucket_pass

# Initialize the new variable
fps_for_full_capture = 20
image_capture_rate_for_full_capture = 1/fps_for_full_capture

# Initialize the new time and condition arrays
time_array_for_full_capture = np.arange(0, evaluation_time, image_capture_rate_for_full_capture)
condition_array_for_full_capture = np.zeros_like(time_array_for_full_capture)

# Determine condition at each time point for the new array
for i, time in enumerate(time_array_for_full_capture):
    if time % length_of_event < length_sub_event_bucket_pass:
        condition_array_for_full_capture[i] = 1

# Print total of each sampled condition and their ratio
bucket_pass_total = np.sum(condition_array == 1)
gap_total = np.sum(condition_array == 0)
st.write(f"Total bucket_pass: {bucket_pass_total}")
st.write(f"Total gap: {gap_total}")
st.write(f"Ratio gap to bucket_pass: {gap_total / bucket_pass_total}")

# Determine condition counts for each fps value
for i, fps in enumerate(fps_values):
    image_capture_rate = 1 / fps
    time_array = np.arange(0, evaluation_time, image_capture_rate)
    condition_array = np.zeros_like(time_array)
    for j, time in enumerate(time_array):
        if time % length_of_event < length_sub_event_bucket_pass:
            condition_array[j] = 1
    bucket_pass_counts[i] = np.sum(condition_array == 1)
    gap_counts[i] = np.sum(condition_array == 0)

# Determine condition ratios for each combination of RPM and fps values
for i, RPM in enumerate(RPM_values):
    events_per_minute = RPM * buckets
    length_of_event = 60 / events_per_minute
    length_sub_event_bucket_pass = bucket_pass_length * length_of_event
    length_sub_event_gap = 1 - bucket_pass_length * length_of_event
    for j, fps in enumerate(fps_values):
        image_capture_rate = 1 / fps
        time_array = np.arange(0, evaluation_time, image_capture_rate)
        condition_array = np.zeros_like(time_array)
        for k, time in enumerate(time_array):
            if time % length_of_event < length_sub_event_bucket_pass:
                condition_array[k] = 1
        bucket_pass_total = np.sum(condition_array == 1)
        gap_total = np.sum(condition_array == 0)
        # Add a small epsilon to the denominator to prevent divide by zero error
        epsilon = 1e-7
        condition_ratios[i, j] = gap_total / (bucket_pass_total + epsilon)  # Reversed the ratio

# Plot condition ratios as a surface plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(fps_values, RPM_values)
surf = ax.plot_surface(X, Y, condition_ratios, cmap=cm.coolwarm)
fig.colorbar(surf, shrink=0.5, aspect=5)  # Added color bar
ax.set_xlabel('image_capture_rate (fps)')
ax.set_ylabel('Cutterhead revolutions per minute (RPM)')
ax.set_zlabel('Ratio bucket pass to gap')
plt.title('Ratio of belt conditions (bucket_pass vs. gap) vs. FPS and cutterhead RPM')
plt.show()
