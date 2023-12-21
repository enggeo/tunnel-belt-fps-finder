import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Initialize variables
RPM = 5.15
buckets = 6
bucket_pass_length = 0.5
expected_ratio = bucket_pass_length/(1 - bucket_pass_length)  # Calculate expected_ratio
evaluation_time = 1200
fps = 2
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

# Plot condition over time
plt.figure(figsize=(25, 6))  # Made the plot wider
plt.plot(time_array, condition_array, 'b-', label='fps')  # Blue line for 'fps' dataset
plt.plot(time_array, condition_array, 'bo')  # Blue circle markers for 'fps' dataset
plt.plot(time_array_for_full_capture, condition_array_for_full_capture, color='lightgrey', label='actual bucket passes')
plt.xlabel('Time (s)')
plt.ylabel('Condition (1 = bucket pass  0 = gap)')
plt.title(f'Belt Bucket Passes sampled at {fps} fps , RPM: {RPM}, Buckets: {buckets}')  # Include 'fps', 'RPM', 'buckets', and 'fps_for_full_capture' in the title
plt.legend()

# Add text box
plt.text(0.05, 0.95, f'Total bucket_pass: {total_bucket_pass}\nTotal gap: {total_gap}\nRatio gap to bucket_pass: {ratio_gap_to_bucket_pass}', transform=plt.gca().transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))

plt.show()

# Print total of each sampled condition and their ratio
bucket_pass_total = np.sum(condition_array == 1)
gap_total = np.sum(condition_array == 0)
print(f"Total bucket_pass: {bucket_pass_total}")
print(f"Total gap: {gap_total}")
print(f"Ratio gap to bucket_pass: {gap_total / bucket_pass_total}")


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

# Plot total sampled conditions over fps
plt.figure(figsize=(10, 6))
plt.plot(fps_values, bucket_pass_counts, label='bucket_pass')
plt.plot(fps_values, gap_counts, label='gap')
plt.xlabel('Image capture rate (fps)')
plt.ylabel('Total Sampled Conditions')
plt.title('Total Sampled Conditions over FPS')
plt.legend()
plt.show()

# Plot ratio of conditions over fps
plt.figure(figsize=(10, 6))
plt.plot(fps_values, gap_counts / bucket_pass_counts, label='Ratio')  # Reversed the ratio and added label

# Add bucket_pass_length as a dashed black line
plt.axhline(y=expected_ratio, color='black', linestyle='--', label='Expected')

plt.xlabel('Image capture rate (fps)')
plt.ylabel('Ratio of bucket pass/empty belt conditions captured')
plt.title('Ratio of bucket pass/empty belt conditions vs. belt image capture rate (FPS)')
plt.legend()
plt.title(f'Ratio of bucket pass/empty belt conditions captured vs. belt image capture rate RPM: {RPM}, Buckets: {buckets}')  # Include 'fps', 'RPM' and 'buckets' in the title
plt.show()

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
