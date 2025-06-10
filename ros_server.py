import json
import os
import numpy as np
import roslibpy

from coverage_planner import CoveragePlanner, HeuristicType
from api_server import fetch_cell_values, HEURISTIC_MAP


def build_grid(rows, cols, start, obstacles):
    grid = np.zeros((rows, cols), dtype=int)
    for obs in obstacles:
        if isinstance(obs, (list, tuple)) and len(obs) == 2:
            r, c = obs
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = 1
    sr, sc = start
    if 0 <= sr < rows and 0 <= sc < cols:
        grid[sr][sc] = 2
    return grid


def plan_path(req):
    try:
        rows = int(req['rows'])
        cols = int(req['cols'])
        start = req['start']
        orientation = int(req.get('orientation', 0))
        heuristic_str = str(req.get('heuristic', 'vertical')).lower()
        obstacles = req.get('obstacles', [])
    except Exception as exc:
        print(f'Invalid request: {exc}')
        return None

    grid = build_grid(rows, cols, start, obstacles)
    cell_values = fetch_cell_values(rows, cols)

    cp = CoveragePlanner(grid, cell_values=cell_values)
    cp.start(initial_orientation=orientation,
             cp_heuristic=HEURISTIC_MAP.get(heuristic_str, HeuristicType.VERTICAL))
    cp.compute(return_home=True)
    res = cp.result()
    path = res[4]
    return rows, cols, grid, path


def create_occupancy_grid(rows, cols, grid, path):
    data = np.zeros((rows, cols), dtype=np.int8)
    data[grid == 1] = 100
    for r, c in path:
        if data[r, c] == 0:
            data[r, c] = 50
    msg = {
        'info': {
            'width': cols,
            'height': rows,
            'resolution': 1.0
        },
        'data': data.flatten().tolist()
    }
    return msg


def main():
    ros = roslibpy.Ros(host=os.getenv('ROSBRIDGE_HOST', 'localhost'),
                       port=int(os.getenv('ROSBRIDGE_PORT', '9090')))
    ros.run()

    pub = roslibpy.Topic(ros, '/planned_grid', 'nav_msgs/OccupancyGrid')

    def handle_message(message):
        if isinstance(message, roslibpy.Message):
            payload = message.msg
        else:
            payload = message
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                print('Received invalid JSON request')
                return
        result = plan_path(payload)
        if result is None:
            return
        rows, cols, grid, path = result
        og_msg = create_occupancy_grid(rows, cols, grid, path)
        pub.publish(roslibpy.Message(og_msg))

    sub = roslibpy.Topic(ros, '/getPlan', 'nav_msgs/GetPlan')
    sub.subscribe(handle_message)

    print('ROS planner node started')
    try:
        while ros.is_connected:
            ros.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sub.unsubscribe()
        pub.unadvertise()
        ros.terminate()


if __name__ == '__main__':
    main()
