import re
import math
import time

import numpy as np

def parse_best_solution(file_path):
    routes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('Route #'):
                match = re.match(r'Route #\d+: \[(.*)\]', line)
                if match:
                    nodes_str = match.group(1)
                    nodes = [int(n.strip())+1 for n in nodes_str.split(',')]
                    routes.append(nodes)
    return routes

def parse_vrp_instance(file_path):
    node_coords = {}
    time_windows = {}
    service_time = None
    with open(file_path, 'r') as f:
        lines = f.readlines()
        parsing_coords = False
        parsing_time_windows = False

        for line in lines:
            line = line.strip()

            if line.startswith('SERVICE_TIME'):
                service_time = int(line.split(':')[1].strip())

            elif line == 'NODE_COORD_SECTION':
                parsing_coords = True
                parsing_time_windows = False
                parsing_demand = False
                continue

            elif line == 'TIME_WINDOW_SECTION':
                parsing_time_windows = True
                parsing_coords = False
                parsing_demand = False
                continue

            elif line in ['DEPOT_SECTION', 'EOF']:
                parsing_coords = False
                parsing_time_windows = False
                parsing_demand = False
                continue


            if parsing_coords:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    node_coords[node_id] = (x, y)

            if parsing_time_windows:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    left = float(parts[1])
                    right = float(parts[2])
                    time_windows[node_id] = (left, right)


    if service_time is None:
        service_time = 0

    return node_coords, service_time, time_windows

def euclidean_distance(coord1, coord2):
    dist = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
    return dist

def calculate_route_time(route_nodes, node_coords, time_windows, service_time, depot_node=1):
    if not route_nodes:
        return 0.0
    start_time = time.perf_counter()

    full_path = [depot_node] + route_nodes + [depot_node]
    total_travel_time = 0.0
    total_wait_time = 0.0
    delta_TU=np.inf
    delta_TL=0.0

    for i in range(len(full_path) - 1):
        from_node = full_path[i]
        to_node = full_path[i + 1]

        distance = euclidean_distance(node_coords[from_node], node_coords[to_node])
        total_travel_time += distance


        left_tw, right_tw = time_windows[to_node]
        left_tw=left_tw
        right_tw=right_tw

        delta_TL = max(delta_TL, total_wait_time + left_tw - total_travel_time)
        delta_TU = min(delta_TU, total_wait_time + right_tw - total_travel_time)


        wait_time = max(0.0, left_tw - total_travel_time)
        total_travel_time = max(total_travel_time,left_tw)
        total_wait_time += wait_time

        if to_node!=1:
            total_travel_time += service_time

    runtime = time.perf_counter() - start_time
    original_total_travel_time=total_travel_time
    original_total_wait_time=total_wait_time


    total_travel_time = delta_TU
    total_wait_time = 0.0
    Distance=0

    for i in range(len(full_path) - 1):
        from_node = full_path[i]
        to_node = full_path[i + 1]

        distance = euclidean_distance(node_coords[from_node], node_coords[to_node])
        Distance += distance
        total_travel_time += distance

        left_tw, right_tw = time_windows[to_node]
        left_tw=left_tw
        right_tw=right_tw

        wait_time = max(0.0, left_tw - total_travel_time)
        total_travel_time = max(total_travel_time, left_tw)
        if from_node == 1:
            first_wait_time = wait_time
        else:
            total_wait_time += wait_time


        if to_node != 1:
            total_travel_time += service_time


    return original_total_travel_time, original_total_wait_time, (total_travel_time-delta_TU), total_wait_time, delta_TL, delta_TU, Distance, runtime

def ODT_main(best_solution_file, vrp_instance_file, txt):

    file_txt = f'experimental_results/exp3_ODT/details_revision/BKS_{txt}.txt'

    routes = parse_best_solution(best_solution_file)
    node_coords, service_time, time_windows = parse_vrp_instance(vrp_instance_file)

    total_travel_time = 0.0
    total_wait_time = 0.0
    total_original_travel_time=0.0
    total_original_wait_time=0.0

    timetable= {}
    i=0
    total_distance=0

    for route in routes:
        i = i + 1
        original_route_time, original_wait_time, route_time, wait_time, delta_TL, delta_TU, distance, runtime = calculate_route_time(route, node_coords, time_windows, service_time)
        total_travel_time += route_time
        total_wait_time += wait_time
        total_original_travel_time += original_route_time
        total_original_wait_time += original_wait_time
        timetable[i]=delta_TU
        total_distance += distance
        with open(file_txt, 'a') as f:
            f.write(f'Route for Vehicle {i}:\n')
            f.write(f'  Customers: '+' -> '.join(['0']+[str(x-1) for x in route]+['0']))
            f.write('\n')
            f.write(f'  Optimal Distance: {distance:.1f}\n')
            f.write(f'  Optimal Departure Time Interval: [{max(min(delta_TL, delta_TU), 0):.1f}, {delta_TU:.1f}]\n')
            f.write(f'  Duration: {original_route_time:.1f} -> Optimal Duration: {route_time:.1f}\n' )
            f.write(f'  Waiting time: {original_wait_time:.1f} -> Optimal Waiting time: {wait_time:.1f}\n')
            f.write(f'  Runtime: {runtime:.5f} seconds\n')
            f.write(f'\n')
    with open(file_txt, 'a') as f:
        f.write(f'Optimal Vehicle Number: {i}\n')
        f.write(f"Optimal Total Distance: {total_distance:.1f}\n")
        f.write(f"Optimal Total Duration: {total_travel_time:.1f}\n")
        f.write(f"Optimal Total Waiting time: {total_wait_time:.1f}\n")

    return total_original_wait_time, total_wait_time
