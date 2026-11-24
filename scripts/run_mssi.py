
import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from mssi import IterativeVRPTWSolver
from odt import parse_vrp_instance, calculate_route_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', required=True,
                    help='path to a VRPPDTW .vrp instance')
    ap.add_argument('--minutes', type=float, default=30,
                    help='wall-clock budget in minutes')
    ap.add_argument('--seed', type=int, default=1,
                    help='random seed for Python and NumPy')
    ap.add_argument('--output', default=None,
                    help='optional path of the JSON result file')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    limit = int(round(args.minutes * 60))
    third = max(1, limit // 3)

    t0 = time.time()
    solver = IterativeVRPTWSolver(
        instance_path=args.instance,
        max_time=limit,
        num_each_problem=100,
        decomposition='C3D',
        algorithm='MSSI',
        time_delimiter=[third, 2 * third, limit],
    )
    solver.iterative_optimization()
    wall = time.time() - t0

    vn = int(solver.best_vehicles)
    td = float(solver.best_distance)

    node_coords, service_time, time_windows = parse_vrp_instance(args.instance)
    total_orig_wait = 0.0
    total_opt_wait = 0.0
    total_distance = 0.0
    routes = solver.best_solution.routes() if solver.best_solution else []
    for route in routes:
        nodes = [int(v) + 1 for v in route.visits()]
        if not nodes:
            continue
        (_, orig_wait, _, opt_wait, _, _, distance,
         _) = calculate_route_time(nodes, node_coords, time_windows,
                                   service_time)
        total_orig_wait += orig_wait
        total_opt_wait += opt_wait
        total_distance += distance

    result = {
        'instance': os.path.basename(args.instance),
        'seed': args.seed,
        'minutes': args.minutes,
        'runtime_seconds': round(wall, 1),
        'VN': vn,
        'TD': round(td, 3),
        'WT_default_departure': round(total_orig_wait, 3),
        'WT_optimized': round(total_opt_wait, 3),
        'routes': len(routes),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
