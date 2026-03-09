#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


import sys

if len(sys.argv) < 2:
    print("Usage: python3 plot_wall_error.py <bag_folder>")
    sys.exit(1)

BAG_PATH = sys.argv[1]

TOPIC_NAME = "/wall_error"

TRIM_START_SEC = 2.0    # ignore first 2 seconds
TRIM_END_SEC = 6.5       # example: 15.0 to stop at 15 sec, or None for full bag


def read_wall_error(bag_path, topic_name):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    if topic_name not in type_map:
        raise ValueError(f"Topic {topic_name} not found in bag")

    msg_type = get_message(type_map[topic_name])

    times_ns = []
    values = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)

        # assumes /wall_error message has a numeric .data field
        values.append(float(msg.data))
        times_ns.append(t)

    if len(values) == 0:
        raise ValueError(f"No messages found on {topic_name}")

    times_ns = np.array(times_ns)
    values = np.array(values)

    times_sec = (times_ns - times_ns[0]) * 1e-9
    return times_sec, values


def trim_by_time(times, values, start_sec=None, end_sec=None):
    mask = np.ones(len(times), dtype=bool)

    if start_sec is not None:
        mask &= times >= start_sec
    if end_sec is not None:
        mask &= times <= end_sec

    return times[mask], values[mask]


def main():
    times, values = read_wall_error(BAG_PATH, TOPIC_NAME)

    # trim off the bad start / end
    times, values = trim_by_time(times, values, TRIM_START_SEC, TRIM_END_SEC)

    if len(values) == 0:
        raise ValueError("No data left after trimming")

    # re-zero time so trimmed plot starts at 0
    times = times - times[0]

    mae = np.mean(np.abs(values))
    std = np.std(values)
    max_error = np.max(np.abs(values))

    print(f"Samples used: {len(values)}")
    print(f"MAE: {mae:.6f}")
    print(f"Standard deviation: {std:.6f}")
    print(f"Max error: {max_error:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(times, values, color="purple",linewidth=3)
    plt.xlabel("Time (s)")
    plt.ylabel("Wall Error (m)")
    plt.title("Wall Tracking Error vs Time")
    plt.grid(True)
    plt.savefig("straight_speed_1")


if __name__ == "__main__":
    main()
