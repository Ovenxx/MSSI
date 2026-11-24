import numpy as np
import sys
import time
import tempfile
import os
import re

from pyvrp import Model, read, Solution
from pyvrp.stop import MaxRuntime
from typing import List, Dict, Tuple
from math import atan2, degrees
import matplotlib.pyplot as plt
import io
import random
from sklearn.cluster import KMeans
import math
from sklearn.preprocessing import StandardScaler

class IterativeVRPTWSolver:
    def __init__(self, instance_path: str, time_delimiter: list, max_time: float = 1800, num_each_problem: int = 100, decomposition: str = 'C3D', algorithm = "MSSI"):
        self._t0 = time.time()
        self.instance_path=instance_path
        '''
        'round' rounds the values to the nearest integer;
        'trunc' truncates the values to an integer;
        'dimacs' scales by 10 and truncates the values to an integer; 
        'exact' scales by 1000 and rounds to the nearest integer.
        'none' does no rounding. This is the default.
        '''

        self.precision_str = "exact"
        self.num_each_problem = num_each_problem
        self.max_time = max_time
        self.decomposition = decomposition
        self.time_delimiter = time_delimiter
        self.algorithm = algorithm

        if self.precision_str == "round":
            self.precision = 1
        elif self.precision_str == "dimacs":
            self.precision = 10
        elif self.precision_str == "exact":
            self.precision = 1000

        self.problem_scale, self.vehicle_capacity, self.service_time, self.demands, self.pickup, self.time_window_earliest, self.time_window_latest, self.depot_latest_time = self.extract_vrp_data(
            instance_path)


        self.problem_scale = self.problem_scale - 1
        self.instance = read(instance_path, round_func=self.precision_str)


        self.min_clusters = max(int(self.problem_scale / self.num_each_problem), 1)


        self.max_clusters = self.min_clusters
        self.num_clusters = self.min_clusters

        self.vehicle_priority = 100000.0


        self.sweep_angle = -3 / np.sqrt(2) * np.sqrt(self.problem_scale) + 120
        if self.sweep_angle < 15:
            self.sweep_angle = 15
        if self.sweep_angle > 120:
            self.sweep_angle = 120
        self.original_sweep_angle = self.sweep_angle


        self.num_locations = self.instance.num_locations
        self.coords = np.array([
            (self.instance.location(i).x, self.instance.location(i).y)
            for i in range(self.num_locations)
        ])
        self.depot_coord = self.coords[0]


        self.customer_coords = self.coords[1:]
        self.num_customers = len(self.customer_coords)


        self.customer_angles = {}
        for idx, coord in enumerate(self.customer_coords, start=1):
            dx = coord[0] - self.depot_coord[0]
            dy = coord[1] - self.depot_coord[1]
            self.customer_angles[idx] = degrees(atan2(dy, dx)) % 360


        self.best_solution = None
        self.best_cost = float('inf')
        self.best_vehicles = float('inf')
        self.best_distance = float('inf')
        self.current_solution = None


        self.current_sweep_angle = 0.0
        self.last_selected_routes = set()


        self.iterations = 0
        self.runtimes = []
        self.costs = []
        self.vehicle_counts = []
        self.sweep_angles = []
        self.subproblem_sizes = []


        SUM=0
        for idx in range(self.num_customers):
            SUM += self.instance.location(idx+1).delivery[0] / self.precision

        self.VN_LB = np.ceil(SUM/self.vehicle_capacity)
        self.stagnation = 0

        self.nb_inferior_accept = 0
        self.max_nb_inferior_accept = 5
        self.max_stagnation = 3

    def extract_vrp_data(self, instance_path):

        dimension = None
        capacity = None
        service_time = None
        time_window_earliest = []
        time_window_latest = []
        demands = []
        pickup = []

        with open(instance_path, 'r') as file:
            time_window_section_found = False
            demands_section_found = False
            backhaul_section_found = False
            for line in file:
                line = line.strip()

                if not line or line.startswith('COMMENT') or line.startswith('NAME') or line.startswith('TYPE'):
                    continue

                if line.startswith('DIMENSION'):
                    dimension = int(line.split(':')[-1].strip())

                elif line.startswith('CAPACITY'):
                    capacity = int(line.split(':')[-1].strip())

                elif line.startswith('SERVICE_TIME'):
                    service_time = int(line.split(':')[-1].strip())

                elif line.startswith('DEMAND_SECTION'):
                    demands_section_found = True
                    continue

                elif line.startswith('BACKHAUL_SECTION'):
                    backhaul_section_found = True
                    continue

                elif line.startswith('TIME_WINDOW_SECTION'):
                    time_window_section_found = True
                    continue

                if demands_section_found and not time_window_section_found and not backhaul_section_found:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            demands.append(float(parts[1]))
                        except (ValueError, IndexError):
                            continue

                if backhaul_section_found and not time_window_section_found:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pickup.append(float(parts[1]))
                        except (ValueError, IndexError):
                            continue

                if time_window_section_found:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            time_window_earliest.append(float(parts[1]))
                            time_window_latest.append(float(parts[2]))
                        except (ValueError, IndexError):
                            continue
                    if not backhaul_section_found:
                        pickup.append(0)

        depot_latest_time = time_window_latest[0]

        return dimension, capacity, service_time, demands, pickup, time_window_earliest, time_window_latest, depot_latest_time

    def boolean_matrix_power_series(self, P, N):
        n = P.shape[0]


        current_power = P.copy()


        numerator = np.zeros((n, n), dtype=float)
        denominator = np.zeros((n, n), dtype=float)


        bool_current_power = (current_power>0).astype(float)

        for k in range(1, N + 1):

            numerator += k * bool_current_power
            denominator += bool_current_power


            if np.all(bool_current_power.astype(int) == 0):
                break


            next_power = current_power @ P
            bool_next_power = ((next_power) > 0).astype(float)


            if np.array_equal(bool_current_power, bool_next_power):
                break

            current_power = next_power
            bool_current_power = bool_next_power


        S = np.zeros((n, n))
        mask = denominator > 0
        S[mask] = numerator[mask] / denominator[mask]
        S[denominator == 0] = 10 ** 3 * np.max(S)

        return S, k, 'max_iterations'

    def transitive_closure_warshall(self, adj_matrix):
        n = adj_matrix.shape[0]

        R = adj_matrix.copy().astype(int)


        for i in range(n):
            R[i, i] = 1


        for k in range(n):
            for i in range(n):
                for j in range(n):

                    if R[i, j] == 0 and R[i, k] == 1 and R[k, j] == 1:
                        R[i, j] = 1

        return R

    def spatial_decomposition(self, num_clusters: int, random_state: int = None) -> Dict[int, List[int]]:

        customer_coords = self.coords[1:]
        num_customers = len(customer_coords)
        if num_customers < num_clusters:
            num_clusters = max(1, num_customers)

        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        kmeans.fit(customer_coords)

        clusters = {}
        for idx, label in zip(range(1, self.num_locations), kmeans.labels_):
            clusters.setdefault(label, []).append(idx)

        for cluster_id in clusters:
            if 0 not in clusters[cluster_id]:
                clusters[cluster_id].insert(0, 0)

        return clusters

    def random_decomposition(self, num_clusters: int, random_state: int = None) -> Dict[int, List[int]]:

        customer_coords = self.coords[1:]
        num_customers = len(customer_coords)


        if num_customers < num_clusters:
            num_clusters = max(1, num_customers)


        if random_state is not None:
            np.random.seed(random_state)


        customer_indices = list(range(1, self.num_locations))


        np.random.shuffle(customer_indices)


        base_size = num_customers // num_clusters
        remainder = num_customers % num_clusters


        clusters = {}
        start_idx = 0

        for cluster_id in range(num_clusters):

            cluster_size = base_size
            if cluster_id < remainder:
                cluster_size += 1


            if cluster_size > 0:
                end_idx = start_idx + cluster_size
                clusters[cluster_id] = customer_indices[start_idx:end_idx]
                start_idx = end_idx


        for cluster_id in clusters:
            if 0 not in clusters[cluster_id]:
                clusters[cluster_id].insert(0, 0)

        return clusters

    def sweep_decomposition(self, num_clusters: int, random_state: int = None) -> Dict[int, List[int]]:


        depot_coords = self.coords[0]


        angles = []
        customer_indices = []

        for idx in range(1, self.num_locations):
            customer_coords = self.coords[idx]

            dx = customer_coords[0] - depot_coords[0]
            dy = customer_coords[1] - depot_coords[1]


            angle = math.atan2(dy, dx)


            angle = (2 * math.pi - angle) % (2 * math.pi)

            angles.append(angle)
            customer_indices.append(idx)


        sorted_pairs = sorted(zip(angles, customer_indices))


        sorted_customer_indices = [idx for _, idx in sorted_pairs]


        num_customers = len(sorted_customer_indices)

        if num_customers < num_clusters:
            num_clusters = max(1, num_customers)


        base_size = num_customers // num_clusters
        remainder = num_customers % num_clusters


        clusters = {}
        start_idx = 0

        for cluster_id in range(num_clusters):

            cluster_size = base_size
            if cluster_id < remainder:
                cluster_size += 1


            if cluster_size > 0:
                end_idx = start_idx + cluster_size
                clusters[cluster_id] = sorted_customer_indices[start_idx:end_idx]
                start_idx = end_idx


        for cluster_id in clusters:
            if 0 not in clusters[cluster_id]:
                clusters[cluster_id].insert(0, 0)

        return clusters

    def C3D_decomposition(self, num_clusters: int, random_state: int = None) -> Dict[int, List[int]]:

        depot_coords = self.coords[0]


        angles = []
        customer_indices = []

        for idx in range(1, self.num_locations):
            customer_coords = self.coords[idx]

            dx = customer_coords[0] - depot_coords[0]
            dy = customer_coords[1] - depot_coords[1]


            angle = math.atan2(dy, dx)


            angle = (2 * math.pi - angle) % (2 * math.pi)

            angles.append(angle)
            customer_indices.append(idx)


        angles_array = np.array(angles).reshape(-1, 1)


        num_customers = len(angles_array)


        if num_customers < num_clusters:
            num_clusters = max(1, num_customers)


        if num_clusters == 1:
            return {0: [0] + list(range(1, self.num_locations))}


        angle_points = np.column_stack([np.cos(angles_array), np.sin(angles_array)])

        if random_state is not None:
            kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        else:
            kmeans = KMeans(n_clusters=num_clusters)

        kmeans.fit(angle_points)


        clusters = {}
        for idx, label in enumerate(kmeans.labels_):

            customer_idx = customer_indices[idx]
            clusters.setdefault(label, []).append(customer_idx)


        for cluster_id in clusters:
            if 0 not in clusters[cluster_id]:
                clusters[cluster_id].insert(0, 0)

        return clusters

    def feature_cluster_decomposition(self, num_clusters: int, random_state: int = None) -> Dict[int, List[int]]:

        depot_coords = self.coords[0]


        features = []
        customer_indices = []

        for idx in range(1, self.num_locations):
            customer_coords = self.coords[idx]

            dx = customer_coords[0] - depot_coords[0]
            dy = customer_coords[1] - depot_coords[1]


            angle = math.atan2(dy, dx)


            angle = (2 * math.pi - angle) % (2 * math.pi)


            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)


            r = math.sqrt(dx ** 2 + dy ** 2)


            feature_vector = [
                cos_angle,
                sin_angle,
                r
            ]

            features.append(feature_vector)
            customer_indices.append(idx)


        features_array = np.array(features)


        num_customers = len(features_array)


        if num_customers < num_clusters:
            num_clusters = max(1, num_customers)


        if num_clusters == 1:
            return {0: [0] + list(range(1, self.num_locations))}


        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_array)


        if random_state is not None:
            kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        else:
            kmeans = KMeans(n_clusters=num_clusters)

        kmeans.fit(scaled_features)


        clusters = {}
        for idx, label in enumerate(kmeans.labels_):

            customer_idx = customer_indices[idx]
            clusters.setdefault(label, []).append(customer_idx)


        for cluster_id in clusters:
            if 0 not in clusters[cluster_id]:
                clusters[cluster_id].insert(0, 0)

        return clusters

    def _merge_clusters(self, clusters: Dict[int, List[int]], target_num: int) -> Dict[int, List[int]]:
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]))

        while len(sorted_clusters) > target_num:
            smallest_id, smallest_nodes = sorted_clusters[0]
            second_id, second_nodes = sorted_clusters[1]

            merged_nodes = smallest_nodes + second_nodes
            new_cluster_id = min(smallest_id, second_id)

            sorted_clusters = sorted_clusters[2:]
            sorted_clusters.append((new_cluster_id, merged_nodes))
            sorted_clusters = sorted(sorted_clusters, key=lambda x: len(x[1]))

        return dict(sorted_clusters)

    def proba_transform(self, D):
        lambda_val = 0.9
        N = D.shape[0]
        epsilon = 1e-10
        H = np.zeros((N, N))

        for n1 in range(N):

            if n1 == 0:
                Dslace = D[n1, 1:]
            elif n1 == N - 1:
                Dslace = D[n1, :n1]
            else:
                Dslace = np.concatenate([D[n1, :n1], D[n1, n1 + 1:]])


            sorted_indices = np.argsort(Dslace)

            R1 = np.zeros_like(sorted_indices)
            for rank, idx in enumerate(sorted_indices):
                R1[idx] = rank + 1


            if n1 == 0:
                R = np.concatenate([[0], R1])
            elif n1 == N - 1:
                R = np.concatenate([R1, [0]])
            else:
                R = np.concatenate([R1[:n1], [0], R1[n1:]])

            maxDslace = np.max(Dslace)


            for n2 in range(N):
                if n2 != n1:

                    H[n1, n2] = lambda_val ** (R[n2]-1) * (maxDslace - D[n1, n2] + epsilon)
                else:
                    H[n1, n2] = 0


            row_sum = np.sum(H[n1, :])
            if row_sum > 0:
                H[n1, :] = H[n1, :] / row_sum

        return H

    def space_time_load_compatibility(self):
        N = self.problem_scale + 1
        D = np.zeros((N, N))
        Vmax = 1
        Vmin = 1
        delta1 = 10 ** (-1)
        delta2 = 0
        epsilon = 10 ** (-10)
        si = self.service_time
        TC = np.zeros_like(D)
        CC = np.zeros_like(D)

        for i in range(N):
            for j in range(N):

                ci = self.coords[i] / self.precision
                cj = self.coords[j] / self.precision
                D[i][j] = np.sqrt(np.sum((ci - cj) ** 2))


                di = self.demands[i]
                dj = self.demands[j]
                pi = self.pickup[i]
                pj = self.pickup[j]
                if np.max([di + dj, pi+dj, pi+pj]) <= self.vehicle_capacity:
                    CC[i][j] = 1


            ei = self.time_window_earliest[i]
            li = self.time_window_latest[i]
            if i == 0:
                for j in range(i + 1, N):
                    ej = self.time_window_earliest[j]
                    lj = self.time_window_latest[j]
                    TC[i, j] = (max(min(D[i, j]/Vmin, lj), ej) - ej) / (
                            lj - ej) - delta1 * (ej - D[i, j] / Vmin)

                tc_slice = TC[0, 1:]
                min_val = np.min(tc_slice)
                max_val = np.max(tc_slice)
                TC[0, 1:] = (tc_slice - min_val + epsilon) / (max_val - min_val + epsilon)
            else:
                for j in range(N):
                    if j != i:
                        ej = self.time_window_earliest[j]
                        lj = self.time_window_latest[j]
                        if ei + si + D[i, j]/Vmax > lj:
                            TC[i, j] = delta2
                        elif li + si + D[i, j]/Vmin >= ej:
                            TC[i, j] = 1
                        else:
                            TC[i, j] = 1 / (1 + ej - (li + si + D[i, j] / Vmin))

            row_sum = np.sum(TC[i, :])
            if row_sum != 0:
                TC[i, :] = TC[i, :] / row_sum
        SC = self.proba_transform(D)

        return SC, TC * CC, D

    def space_time_load_compatibility_paper7(self):
        N = self.problem_scale + 1
        D = np.zeros((N, N))
        Vmax = 1
        Vmin = 1
        delta1 = 10 ** (-1)
        delta2 = 0
        epsilon = 10 ** (-10)
        si = self.service_time
        TC = np.zeros_like(D)
        CC = np.zeros_like(D)

        for i in range(N):
            for j in range(N):

                ci = self.coords[i] / self.precision
                cj = self.coords[j] / self.precision
                D[i][j] = np.sqrt(np.sum((ci - cj) ** 2))


                di = self.demands[i]
                dj = self.demands[j]
                pi = self.pickup[i]
                pj = self.pickup[j]
                if np.max([di + dj, pi+dj, pi+pj]) <= self.vehicle_capacity:
                    CC[i][j] = 1


            ei = self.time_window_earliest[i]
            li = self.time_window_latest[i]
            if i == 0:
                for j in range(i + 1, N):
                    ej = self.time_window_earliest[j]
                    lj = self.time_window_latest[j]
                    TC[i, j] = (max(min(D[i, j]/Vmin, lj), ej) - ej) / (
                            lj - ej) - delta1 * (ej - D[i, j] / Vmin)

                tc_slice = TC[0, 1:]
                min_val = np.min(tc_slice)
                max_val = np.max(tc_slice)
                TC[0, 1:] = (tc_slice - min_val + epsilon) / (max_val - min_val + epsilon)
            else:
                for j in range(N):
                    if j != i:
                        ej = self.time_window_earliest[j]
                        lj = self.time_window_latest[j]
                        if ei + si + D[i, j]/Vmax > lj:
                            TC[i, j] = delta2
                        elif li + si + D[i, j]/Vmin >= ej:
                            TC[i, j] = 1
                        else:
                            TC[i, j] = 1 / (1 + ej - (li + si + D[i, j] / Vmin))

        SC = self.proba_transform(D)
        P = SC * TC * CC

        return P, D

    def roulette_wheel_selection(self, P):

        P = P / np.sum(P)

        C = np.cumsum(P)

        r = np.random.rand()
        i = np.sum(r > C)
        return i

    def individual_construction(self, PTC, D):
        P = PTC.copy()
        P[:,0] = 0
        individual = []
        individual.append(0)
        flag = [False] * (self.problem_scale+1)
        flag[0] = True
        sum_d = 0
        sum_p = 0
        mixed_d_p_min = 0
        mixed_d_p = 0
        current_route_costumers = []
        arrival_time = 0
        while not np.all(flag):
            i = individual[-1]
            if np.sum(P[i, :]) == 0:

                j = 0
                individual.append(j)
                sum_d = 0
                sum_p = 0
                mixed_d_p_min = 0
                mixed_d_p = 0
                current_route_costumers = []
                arrival_time = 0
                continue
            if len(current_route_costumers)==0:
                j = int(np.argmax(P[i,:]))

            else:
                x_mean = np.mean(self.coords[current_route_costumers, 0])
                y_mean = np.mean(self.coords[current_route_costumers, 1])
                current_center = [x_mean, y_mean]
                P_modify = []
                for ii in range(self.problem_scale+1):
                    if P[i, ii]==0:
                        P_modify.append(0)
                    else:
                        mid1 = 1/np.sqrt(np.sum((np.array(self.coords[ii]) - np.array(current_center)) ** 2)) * P[i, ii]
                        P_modify.append(mid1)


                j = int(np.argmax(P_modify))


            if i==0:
                st = 0
            else:
                st = self.service_time
            if arrival_time + st + D[i][j] > self.time_window_latest[j] or self.demands[j] + sum_d + mixed_d_p_min > self.vehicle_capacity or self.pickup[j] + sum_p > self.vehicle_capacity:


                j = 0
                individual.append(j)
                sum_d = 0
                sum_p = 0
                mixed_d_p_min = 0
                mixed_d_p = 0
                current_route_costumers = []
                arrival_time = 0
            else:
                individual.append(j)
                P[:, j] = 0
                sum_d += self.demands[j]
                sum_p += self.pickup[j]
                mixed_d_p += self.pickup[j] - self.demands[j]
                if mixed_d_p <= mixed_d_p_min:
                    mixed_d_p_min = mixed_d_p
                current_route_costumers.append(j)
                flag[j] = True
                arrival_time = np.max([arrival_time + st + D[i][j], self.time_window_earliest[j]])
        return individual

    def individual_construction_paper7(self, P, D):
        P = P.copy()
        P[:,0] = 0
        individual = []
        individual.append(0)
        flag = [False] * (self.problem_scale+1)
        flag[0] = True
        sum_d = 0
        sum_p = 0
        mixed_d_p_min = 0
        mixed_d_p = 0
        current_route_costumers = []
        arrival_time = 0
        while not np.all(flag):
            i = individual[-1]
            if np.sum(P[i, :]) == 0:

                j = 0
                individual.append(j)
                sum_d = 0
                sum_p = 0
                mixed_d_p_min = 0
                mixed_d_p = 0
                current_route_costumers = []
                arrival_time = 0
                continue
            else:
                j = int(np.argmax(P[i,:]))


            if i==0:
                st = 0
            else:
                st = self.service_time

            if arrival_time + st + D[i][j] > self.time_window_latest[j] or self.demands[j] + sum_d + mixed_d_p_min > self.vehicle_capacity or self.pickup[j] + sum_p > self.vehicle_capacity:

                j = 0
                individual.append(j)
                sum_d = 0
                sum_p = 0
                mixed_d_p_min = 0
                mixed_d_p = 0
                current_route_costumers = []
                arrival_time = 0
            else:
                individual.append(j)
                P[:, j] = 0
                sum_d += self.demands[j]
                sum_p += self.pickup[j]
                mixed_d_p += self.pickup[j] - self.demands[j]
                if mixed_d_p <= mixed_d_p_min:
                    mixed_d_p_min = mixed_d_p
                current_route_costumers.append(j)
                flag[j] = True
                arrival_time = np.max([arrival_time + st + D[i][j], self.time_window_earliest[j]])
        return individual

    def cluster_routes_centers(self, routes_center_coords, routes_customer_number):
        num_clusters = self.num_clusters


        if not routes_center_coords:
            return {}


        route_indices = list(routes_center_coords.keys())
        center_points = np.array([routes_center_coords[idx] for idx in route_indices])


        if num_clusters >= len(route_indices):
            cluster_dict = {}
            for i, route_idx in enumerate(route_indices):
                cluster_dict[i] = [route_idx]
            return cluster_dict


        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(center_points)


        cluster_dict = {}
        for route_idx, cluster_label in zip(route_indices, cluster_labels):
            if cluster_label not in cluster_dict:
                cluster_dict[cluster_label] = []
            cluster_dict[cluster_label].append(route_idx)

        return cluster_dict

    def sweep_routes_centers(self, routes_center_coords, routes_customer_number):


        if not routes_center_coords:
            return {}


        route_indices = list(routes_center_coords.keys())


        if not route_indices:
            return {}


        max_customers_per_cluster = np.ceil(self.problem_scale / self.num_clusters)


        angles = []
        for idx in route_indices:
            x, y = routes_center_coords[idx]

            angle = np.arctan2(y, x)

            if angle < 0:
                angle += 2 * np.pi
            angles.append(angle)


        sorted_indices = [idx for _, idx in sorted(zip(angles, route_indices))]


        cluster_dict = {}
        current_cluster = 0
        current_customer_count = 0


        for route_idx in sorted_indices:

            customer_count = routes_customer_number[route_idx]


            if current_customer_count > max_customers_per_cluster:
                current_cluster += 1
                current_customer_count = 0


            if current_cluster not in cluster_dict:
                cluster_dict[current_cluster] = []

            cluster_dict[current_cluster].append(route_idx)
            current_customer_count += customer_count


        return cluster_dict

    def compatibility_based_decomposition(self, num_clusters):


        P, D = self.space_time_load_compatibility_paper7()
        individual = self.individual_construction_paper7(P, D)


        routes = {}
        n = -1
        for i in range(len(individual)):
            if individual[i] == 0:
                n += 1
                routes[n] = []
            else:
                routes[n].append(individual[i])
        routes_center_coords = {}
        routes_customer_number = {}
        for n in range(len(routes)):
            customers = routes[n]
            center_x = float(np.mean(self.coords[customers, 0])/self.precision)
            center_y = float(np.mean(self.coords[customers, 1])/self.precision)
            routes_customer_number[n] = len(customers)
            routes_center_coords[n] = (center_x, center_y)

        routes_center_clusters = self.cluster_routes_centers(routes_center_coords, routes_customer_number)


        clusters = {}
        for k in range(len(routes_center_clusters)):
            clusters[k] = []
            clusters[k].append(0)
            routes_subset = routes_center_clusters[k]
            for kk in range(len(routes_subset)):
                for customer in routes[routes_subset[kk]]:
                    clusters[k].append(customer)

        return clusters

    def cake_spatial_decomposition(self, num_clusters: int, random_state: int = None) -> Dict[int, List[int]]:

        if self.num_customers < num_clusters:
            num_clusters = max(1, self.num_customers)


        if num_clusters == 1:
            return {0: [0] + list(range(1, self.num_customers))}
        if self.num_customers == 0:
            return {0: [0]}


        sector_width = 360.0 / num_clusters


        if random_state is not None:
            random.seed(random_state)
        random_start = random.uniform(0, 360)


        clusters = {i: [] for i in range(num_clusters)}


        for idx, angle in self.customer_angles.items():

            rotated_angle = (angle + random_start) % 360


            sector_id = int(rotated_angle // sector_width)


            clusters[sector_id].append(idx)


        for sector_id in clusters:
            clusters[sector_id].insert(0, 0)

        return clusters

    def generate_subproblem(self, node_indices: List[int], original_vehicle_number: int, process_flag: int):

        buffer = io.StringIO()
        num_nodes = len(node_indices)


        orig_to_sub = {}
        sub_to_orig = {}
        for new_idx, orig_idx in enumerate(node_indices):
            orig_to_sub[orig_idx] = new_idx
            sub_to_orig[new_idx] = orig_idx


        buffer.write(f"NAME : SUBPROBLEM_{num_nodes}\n")
        buffer.write("TYPE : VRPTW\n")
        buffer.write(f"DIMENSION : {num_nodes}\n")
        if process_flag == 0:
            buffer.write(f"VEHICLES : {self.instance.num_vehicles}\n")
        else:
            buffer.write(f"VEHICLES : {original_vehicle_number}\n")

        buffer.write(f"CAPACITY : {self.vehicle_capacity}\n")
        buffer.write(f"SERVICE_TIME : {int(self.service_time)}\n")
        buffer.write("EDGE_WEIGHT_TYPE : EUC_2D\n")

        buffer.write("NODE_COORD_SECTION\n")
        for new_idx, orig_idx in enumerate(node_indices, start=1):
            loc = self.instance.location(orig_idx)
            buffer.write(f"{new_idx}\t{int(loc.x / self.precision)}\t{int(loc.y / self.precision)}\n")

        buffer.write("DEMAND_SECTION\n")
        for new_idx, orig_idx in enumerate(node_indices, start=1):
            loc = self.instance.location(orig_idx)
            if orig_idx == 0:
                buffer.write(f"1 0\n")
            else:
                demand = loc.delivery[0]
                buffer.write(f"{new_idx}\t{int(demand / self.precision)}\n")

        buffer.write("BACKHAUL_SECTION\n")
        for new_idx, orig_idx in enumerate(node_indices, start=1):
            loc = self.instance.location(orig_idx)
            if orig_idx == 0:
                buffer.write(f"1 0\n")
            else:
                pickup = loc.pickup[0]
                buffer.write(f"{new_idx}\t{int(pickup / self.precision)}\n")

        buffer.write("TIME_WINDOW_SECTION\n")
        for new_idx, orig_idx in enumerate(node_indices, start=1):
            loc = self.instance.location(orig_idx)
            if orig_idx == 0:
                buffer.write(f"1 0 {self.depot_latest_time}\n")
            else:
                buffer.write(f"{new_idx}\t{int(loc.tw_early / self.precision)}\t{int(loc.tw_late / self.precision)}\n")

        buffer.write("DEPOT_SECTION\n")
        buffer.write("1\n")
        buffer.write("-1\n")
        buffer.write("EOF\n")


        fd, path = tempfile.mkstemp(suffix=".vrp")
        with os.fdopen(fd, 'w') as tmp_file:
            tmp_file.write(buffer.getvalue())
        buffer.close()

        return path, orig_to_sub, sub_to_orig

    def solve_subproblem(self, subproblem_path: str, solve_time: int, previous_solution: Solution = None) -> Solution:

        try:
            sub_instance = read(subproblem_path, round_func=self.precision_str)

            model = Model.from_data(sub_instance)

            if previous_solution is None:
                result = model.solve(stop=MaxRuntime(solve_time), seed=np.random.randint(0, 10000), display=False)

            else:
                result = model.solve(stop=MaxRuntime(solve_time), seed=np.random.randint(0, 10000), display=False, previous_best=previous_solution)

            return result.best
        except Exception as e:
            print(f"Failed to solve subproblem: {str(e)}")
            return Solution(self.instance, [])

    def evaluate_solution(self, solution: Solution):

        try:
            num_vehicles = solution.num_routes()
            total_distance = solution.distance() / self.precision
        except AttributeError:
            return float('inf'), float('inf'), float('inf')

        weighted_cost = num_vehicles * self.vehicle_priority + total_distance
        return weighted_cost, num_vehicles, total_distance

    def combine_solutions(self, solutions: List[Solution], clusters: Dict[int, List[int]]) -> Solution:

        all_routes = []
        covered_nodes = set([0])

        for cluster_id, solution in zip(clusters.keys(), solutions):
            if solution.num_routes() == 0:
                continue

            for route in solution.routes():
                orig_route = []
                for node in route.visits():
                    if node < len(clusters[cluster_id]):
                        orig_node = clusters[cluster_id][node]
                        if orig_node not in covered_nodes:
                            orig_route.append(orig_node)
                            covered_nodes.add(orig_node)
                if orig_route:
                    all_routes.append(orig_route)

        return Solution(self.instance, all_routes)

    def select_subproblem_by_rotating_sweep(self, solution: Solution, better_flag: int, stage_number: int,
                                            num_vehicles: int) -> Tuple[List[int], int, List[List[int]]]:

        if not hasattr(self, 'last_selected_routes') or self.last_selected_routes is None:
            self.last_selected_routes = set()

        depot_x, depot_y = self.depot_coord

        route_angles = []
        for idx, route in enumerate(solution.routes()):
            if len(route.visits()) == 0:
                continue


            angles = []
            for node in route.visits():

                if node in self.customer_angles:
                    angles.append(self.customer_angles[node])
            if not angles:
                continue


            avg_angle = self._circular_mean(angles)
            route_angles.append((idx, route, avg_angle))

        if not route_angles:
            return [0], 0, []


        if better_flag == 0 and stage_number == 1:
            self.sweep_angle = self.sweep_angle + 45
            print(f"Update sweep angle: {self.sweep_angle}")
        if better_flag == 1 and stage_number == 1:
            self.sweep_angle = self.sweep_angle + 10
            print(f"Update sweep angle: {self.sweep_angle}")
        if better_flag == 0 and stage_number == 2:
            self.sweep_angle = self.sweep_angle + 15
        if better_flag == 1 and stage_number == 2:
            self.sweep_angle = self.sweep_angle + 0

        if stage_number == 1 and num_vehicles == self.VN_LB + 2:
            self.sweep_angle = max(270, self.sweep_angle)
        elif stage_number == 1 and num_vehicles == self.VN_LB + 1:
            self.sweep_angle = 360

        print(f"better_flag: {better_flag}")
        print(f"stage_number: {stage_number}")
        print(f"Current sweep angle: {self.sweep_angle}")

        if stage_number == 1:
            if self.sweep_angle > 360:
                self.sweep_angle = 360
        else:
            if self.sweep_angle > 360:
                self.sweep_angle = self.original_sweep_angle
            if self.sweep_angle < 20:
                self.sweep_angle = self.original_sweep_angle

        selected_routes = []
        while len(selected_routes) == 0:
            angle_low = self.current_sweep_angle
            angle_high = (self.current_sweep_angle + self.sweep_angle) % 360
            for idx, route, angle in route_angles:
                if self.sweep_angle >= 360:
                    selected_routes.append((idx, route))
                else:
                    if angle_low <= angle_high:
                        if angle_low <= angle <= angle_high:
                            selected_routes.append((idx, route))
                    else:
                        if angle >= angle_low or angle <= angle_high:
                            selected_routes.append((idx, route))

            if len(selected_routes) > 0:
                if self.last_selected_routes:
                    current_selected = set(idx for idx, _ in selected_routes)
                    current_selected = set(int(x) for x in current_selected)
                    self.last_selected_routes = set(int(x) for x in self.last_selected_routes)

                    overlap = current_selected & self.last_selected_routes

                    if not overlap:
                        min_diff = 360
                        best_route = None
                        for idx in self.last_selected_routes:
                            for route_idx, route, angle in route_angles:
                                if route_idx == idx:
                                    diff = min(abs(angle - self.current_sweep_angle),
                                               360 - abs(angle - self.current_sweep_angle))
                                    if diff < min_diff:
                                        min_diff = diff
                                        best_route = (route_idx, route)
                                    break

                        if best_route:
                            selected_routes.append(best_route)
                            print(f"Added overlapping route {best_route[0]} to ensure continuity")
            else:
                if self.sweep_angle >= 360:
                    print('Without customers!')
                    break

            self.last_selected_routes = set(int(idx) for idx, _ in selected_routes)

            selected_customers = set()
            partial_routes = []
            for _, route in selected_routes:
                visits = list(route.visits())
                selected_customers.update(visits)

                partial_routes.append(visits)

            original_vehicle_number = len(selected_routes)
            print(f"The vehicle number of the current sub problem: {original_vehicle_number}")

            self.sweep_angles.append(self.current_sweep_angle)


            sweep_step = self.sweep_angle / 2
            self.current_sweep_angle = (self.current_sweep_angle + sweep_step) % 360

        return [0] + list(selected_customers), original_vehicle_number, partial_routes

    def _circular_mean(self, angles: List[float]) -> float:

        sin_sum = 0.0
        cos_sum = 0.0

        for angle in angles:
            rad = np.radians(angle)
            sin_sum += np.sin(rad)
            cos_sum += np.cos(rad)

        mean_rad = atan2(sin_sum / len(angles), cos_sum / len(angles))
        mean_deg = degrees(mean_rad) % 360
        return mean_deg

    def merge_solutions(self, full_solution: Solution, subproblem_nodes: List[int],sub_solution: Solution) -> Solution:

        sub_customers = set(subproblem_nodes) - {0}

        new_routes = []

        for route in full_solution.routes():
            if all(node not in sub_customers for node in route.visits()):
                new_routes.append(list(route.visits()))

        for route in sub_solution.routes():
            orig_route = []
            for node in route.visits():
                if node < len(subproblem_nodes):
                    orig_node = subproblem_nodes[node]
                    if orig_node != 0:
                        orig_route.append(orig_node)
            if orig_route:
                new_routes.append(orig_route)

        return Solution(self.instance, new_routes)

    def iterative_optimization(self):
        start_time = self._t0
        best_iteration = 0

        num_clusters = random.randint(self.min_clusters, self.max_clusters)

        random_seed = random.randint(0, 10000)
        if self.decomposition == 'CD':
            print("Generating initial solution with cluster decomposition...")
            clusters = self.spatial_decomposition(num_clusters, random_seed)
        elif self.decomposition == 'SD':
            print("Generating initial solution with sweep decomposition...")
            clusters = self.sweep_decomposition(num_clusters, random_seed)
        elif self.decomposition == 'RD':
            print("Generating initial solution with random decomposition...")
            clusters = self.random_decomposition(num_clusters, random_seed)
        elif self.decomposition == 'C3D':
            print("Generating initial solution with Connectivity-Based Cyclic Clustering Decomposition (C3D)...")
            clusters = self.C3D_decomposition(num_clusters, random_seed)

        solutions = []
        temp_files = []
        for cluster_id, node_indices in clusters.items():
            if len(node_indices) <= 1:
                continue
            sub_file, orig_to_sub, sub_to_orig = self.generate_subproblem(node_indices, self.instance.num_vehicles, 0)
            temp_files.append(sub_file)
            sub_time = len(node_indices) / 10
            solution = self.solve_subproblem(sub_file, sub_time)
            solutions.append(solution)

        initial_solution = self.combine_solutions(solutions, clusters)
        self.current_solution = initial_solution

        cost, num_vehicles, distance = self.evaluate_solution(initial_solution)
        if cost < self.best_cost:
            self.best_solution = initial_solution
            self.best_cost = cost
            self.best_vehicles = num_vehicles
            self.best_distance = distance
            best_iteration = 0
            print(f"Initial solution: Vehicles: {num_vehicles}, Distance: {distance}")

        self.current_sweep_angle = 0.0
        self.last_selected_routes = set()

        if num_clusters > 1:

            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting temp file: {str(e)}")

        print("Starting local optimization with rotating sweep...")
        print(f"Vehicle number lower bound: {self.VN_LB}")
        better_flag = 1
        stage_number = 1

        process_data_VN = []
        process_data_TD = []
        td = 0
        J = True

        best_vehicles_list = []
        best_distance_list = []

        total_time = time.time() - start_time
        while total_time < self.max_time:

            for ti in range(len(self.time_delimiter)-1, -1,-1):
                if total_time >= self.time_delimiter[ti]:
                    best_vehicles_list.append(self.best_vehicles)
                    best_distance_list.append(self.best_distance)

            if num_vehicles == self.VN_LB and stage_number == 1:
                stage_number = 2
                self.sweep_angle=self.original_sweep_angle

            if time.time() - start_time > 600:
                print("=" * 10 + "10 min!" + "=" * 10)

            if time.time() - start_time >= self.max_time:
                break

            self.iterations += 1
            iter_start = time.time()

            try:
                sub_nodes, original_vehicle_number, partial_routes = self.select_subproblem_by_rotating_sweep(self.current_solution, better_flag, stage_number,num_vehicles)
                self.subproblem_sizes.append(len(sub_nodes) - 1)

                if stage_number == 1:
                    sub_time = 3 * len(sub_nodes) ** 2 / (10 * self.num_customers)

                    remaining = self.max_time - (time.time() - start_time)
                    sub_time = max(0.0, min(sub_time, remaining))
                    if len(partial_routes) == 1:
                        sub_file, orig_to_sub, sub_to_orig = self.generate_subproblem(sub_nodes, original_vehicle_number, 1)
                    else:
                        sub_file, orig_to_sub, sub_to_orig = self.generate_subproblem(sub_nodes, original_vehicle_number - 1, 1)


                    sub_instance = read(sub_file, round_func=self.precision_str)
                    ori_partial_routes=partial_routes

                    sub_initial_solution = []
                    if len(partial_routes) > 1:
                        min_pop_size = 40

                        for _ in range(min_pop_size):
                            partial_routes = ori_partial_routes


                            if partial_routes:
                                route_to_remove_idx = random.randint(0, len(partial_routes) - 1)
                                removed_customers = partial_routes.pop(route_to_remove_idx)


                                for customer in removed_customers:
                                    if not partial_routes:
                                        partial_routes.append([customer])
                                    else:
                                        target_route_idx = random.randint(0, len(partial_routes) - 1)
                                        insert_pos = random.randint(0, len(partial_routes[target_route_idx]))
                                        partial_routes[target_route_idx].insert(insert_pos, customer)
                            mapped_routes = self.map_routes(partial_routes, orig_to_sub)
                            sub_initial_solution.append(Solution(sub_instance, mapped_routes))


                    sub_solution = self.solve_subproblem(sub_file, sub_time, sub_initial_solution)
                    print(f"Successfully reducing vehicle number? {sub_solution.is_feasible()}")
                    if not sub_solution.is_feasible():
                        better_flag = 0

                        if len(sub_nodes) >= self.num_locations - 1:
                            print("Enter into stage 2")
                            stage_number = 2
                            self.sweep_angle = self.original_sweep_angle
                    else:
                        new_solution = self.merge_solutions(
                            self.current_solution, sub_nodes, sub_solution
                        )

                        cost, num_vehicles, distance = self.evaluate_solution(new_solution)

                        self.current_solution = new_solution

                        better_flag = 1

                        self.best_solution = new_solution
                        self.best_cost = cost
                        self.best_vehicles = num_vehicles
                        self.best_distance = distance
                        best_iteration = self.iterations
                        print(f"Iteration {self.iterations}: New best! Vehicles: {num_vehicles}, Distance: {distance}")

                if stage_number == 2:
                    sub_time = 3 * len(sub_nodes) ** 2 / (10 * self.num_customers)

                    remaining = self.max_time - (time.time() - start_time)
                    sub_time = max(0.0, min(sub_time, remaining))
                    sub_file, orig_to_sub, sub_to_orig = self.generate_subproblem(sub_nodes, original_vehicle_number, 1)


                    sub_instance = read(sub_file, round_func=self.precision_str)
                    ori_partial_routes = partial_routes
                    sub_initial_solution = []

                    if len(partial_routes) > 1:
                        min_pop_size = 40


                        mapped_routes = self.map_routes(partial_routes, orig_to_sub)
                        if mapped_routes:
                            sub_initial_solution.append(Solution(sub_instance, mapped_routes))


                        for _ in range(min_pop_size):

                            perturbed_routes = self.perturb_solution_simple(ori_partial_routes,
                                                                            num_customers_to_remove=int(
                                                                                np.sqrt(self.num_customers)))


                            mapped_routes = self.map_routes(perturbed_routes, orig_to_sub)
                            if mapped_routes:
                                sub_initial_solution.append(Solution(sub_instance, mapped_routes))

                    sub_solution = self.solve_subproblem(sub_file, sub_time, sub_initial_solution)

                    if sub_solution.is_feasible():
                        new_solution = self.merge_solutions(
                            self.current_solution, sub_nodes, sub_solution
                        )

                        cost, num_vehicles, distance = self.evaluate_solution(new_solution)

                        if cost < self.best_cost:
                            better_flag = 1
                            self.current_solution = new_solution
                            self.best_solution = new_solution
                            self.best_cost = cost
                            self.best_vehicles = num_vehicles
                            self.best_distance = distance
                            best_iteration = self.iterations
                            print(
                                f"Iteration {self.iterations}: New best! Vehicles: {num_vehicles}, Distance: {distance}")
                        else:
                            better_flag = 0
                        print(f"Stagnation counter: {self.stagnation}")
                    else:
                        better_flag = 0

                self.runtimes.append(time.time() - iter_start)
                self.costs.append(self.best_cost)
                self.vehicle_counts.append(self.best_vehicles)


                try:
                    if os.path.exists(sub_file):
                        os.unlink(sub_file)
                except Exception as e:
                    print(f"Error deleting subproblem file: {str(e)}")

            except Exception as e:
                print(f"Iteration {self.iterations} failed: {str(e)}")
                import traceback
                traceback.print_exc()
            total_time = time.time() - start_time
            if total_time > self.time_delimiter[td] and J:
                process_data_VN.append(self.best_vehicles)
                process_data_TD.append(self.best_distance)
                print("\n" + "=" * 60)
                print(f"Optimization Report in {total_time:.2f} s")
                print("=" * 60)
                print(f"Total iterations: {self.iterations}")
                print(f"Best solution found at iteration: {best_iteration}")
                print(f"Best solution vehicles: {self.best_vehicles}")
                print(f"Best solution distance: {self.best_distance}")
                print("=" * 60)
                td += 1
                if td > len(self.time_delimiter):
                    J = False
            best_vehicles_list.append(self.best_vehicles)
            best_distance_list.append(self.best_distance)
            total_time = time.time() - start_time

        return best_vehicles_list, best_distance_list, self.best_solution, process_data_VN, process_data_TD

    def save_solution(self, filename: str):
        if self.best_solution:
            with open(filename, 'w') as f:
                f.write(f"Best solution cost: {self.best_cost}\n")
                f.write(f"Vehicles: {self.best_vehicles}, Distance: {self.best_distance}\n\n")
                for idx, route in enumerate(self.best_solution.routes()):
                    f.write(f"Route #{idx + 1}: {route.visits()}\n")
            print(f"Solution saved to {filename}")
        else:
            print("No solution to save")

    def get_statistics(self) -> Dict[str, List]:
        return {
            'iterations': self.iterations,
            'runtimes': self.runtimes,
            'costs': self.costs,
            'vehicle_counts': self.vehicle_counts,
            'best_vehicles': self.best_vehicles,
            'best_distance': self.best_distance,
            'best_cost': self.best_cost,
            'subproblem_sizes': self.subproblem_sizes,
            'sweep_angles': self.sweep_angles
        }

    def plot_optimization_progress(self):
        stats = self.get_statistics()

        plt.figure(figsize=(16, 12))

        plt.subplot(4, 1, 1)
        plt.plot(stats['costs'], 'b-', alpha=0.6, label='Cost per Iteration')
        plt.axhline(y=stats['best_cost'], color='r', linestyle='--', label='Best Cost')
        plt.ylabel('Weighted Cost')
        plt.title('VRPTW Iterative Optimization with Rotating Sweep')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(stats['vehicle_counts'], 'g-', alpha=0.6, label='Vehicles per Iteration')
        plt.axhline(y=stats['best_vehicles'], color='r', linestyle='--', label='Best Vehicles')
        plt.ylabel('Number of Vehicles')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(stats['subproblem_sizes'], 'm-', alpha=0.7, label='Subproblem Size')
        plt.xlabel('Iteration')
        plt.ylabel('Customers in Subproblem')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(stats['sweep_angles'], 'c-', marker='o', alpha=0.7, label='Sweep Angle')
        plt.xlabel('Iteration')
        plt.ylabel('Sweep Angle (degrees)')
        plt.ylim(0, 360)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('optimization_progress_rotating_sweep.png', dpi=300)
        plt.show()

    def perturb_solution_simple(self, partial_routes, num_customers_to_remove=5):

        perturbed_routes = [route[:] for route in partial_routes]


        all_customers = []
        for route in perturbed_routes:
            all_customers.extend(route)


        if not all_customers:
            return perturbed_routes


        num_to_remove = min(num_customers_to_remove, len(all_customers))
        customers_to_remove = random.sample(all_customers, num_to_remove)


        for customer in customers_to_remove:
            for route in perturbed_routes:
                if customer in route:
                    route.remove(customer)
                    break


        random.shuffle(customers_to_remove)

        for customer in customers_to_remove:

            target_route = random.choice(perturbed_routes)

            insert_pos = random.randint(0, len(target_route))
            target_route.insert(insert_pos, customer)


        perturbed_routes = [route for route in perturbed_routes if route]

        return perturbed_routes

    def map_routes(self, routes, mapping_dict):
        mapped_routes = []
        for route in routes:
            mapped_route = []
            for node in route:
                if node in mapping_dict:
                    mapped_node = mapping_dict[node]
                    if mapped_node != mapping_dict[0]:
                        mapped_route.append(mapped_node)
            if mapped_route:
                mapped_routes.append(mapped_route)
        return mapped_routes


