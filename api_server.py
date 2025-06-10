import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import numpy as np
from coverage_planner import CoveragePlanner, HeuristicType

HEURISTIC_MAP = {
    'manhattan': HeuristicType.MANHATTAN,
    'chebyshev': HeuristicType.CHEBYSHEV,
    'vertical': HeuristicType.VERTICAL,
    'horizontal': HeuristicType.HORIZONTAL,
}


def fetch_cell_values(rows, cols):
    """Retrieve a matrix of cell values from an external service.

    The service URL can be configured using the ``CELL_VALUE_URL``
    environment variable. The request payload contains ``rows`` and
    ``cols`` and the service is expected to return a JSON object with a
    ``values`` field holding a 2-D list.
    """
    url = os.getenv('CELL_VALUE_URL', 'http://localhost:8001/values')
    payload = json.dumps({'rows': rows, 'cols': cols}).encode('utf-8')
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            values = data.get('values')
            if values and len(values) == rows and len(values[0]) == cols:
                return np.array(values, dtype=float)
    except Exception as exc:
        print(f'Error fetching cell values: {exc}')
    return np.zeros((rows, cols))

class PlannerHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, data):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/plan':
            self._send_json(404, {'error': 'Not found'})
            return
        length = int(self.headers.get('Content-Length', 0))
        try:
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
        except Exception:
            self._send_json(400, {'error': 'Invalid JSON'})
            return
        try:
            rows = int(payload.get('rows'))
            cols = int(payload.get('cols'))
            start = payload.get('start')
            obstacles = payload.get('obstacles', [])
            orientation = int(payload.get('orientation', 0))
            heuristic_str = str(payload.get('heuristic', 'vertical')).lower()
        except Exception:
            self._send_json(400, {'error': 'Invalid parameters'})
            return
        if start is None or len(start) != 2:
            self._send_json(400, {'error': 'start coordinate required'})
            return
        grid = np.zeros((rows, cols), dtype=int)
        for obs in obstacles:
            if isinstance(obs, (list, tuple)) and len(obs) == 2:
                r, c = obs
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r][c] = 1

        # Retrieve the value matrix from the external service
        cell_values = fetch_cell_values(rows, cols)
        sr, sc = start
        if not (0 <= sr < rows and 0 <= sc < cols):
            self._send_json(400, {'error': 'start out of bounds'})
            return
        grid[sr][sc] = 2

        # Prefer paths through cells with value >= 0.1 by initially treating
        # low-value cells as obstacles. If no valid path is found, fall back to
        # allow traversal with penalties.
        preferred_grid = np.copy(grid)
        for r in range(rows):
            for c in range(cols):
                if cell_values[r][c] < 0.1 and preferred_grid[r][c] == 0 and not (r == sr and c == sc):
                    preferred_grid[r][c] = 1

        cp = CoveragePlanner(preferred_grid, cell_values=cell_values)
        cp.start(initial_orientation=orientation,
                 cp_heuristic=HEURISTIC_MAP.get(heuristic_str, HeuristicType.VERTICAL))
        cp.compute(return_home=True)
        res = cp.result()

        if not res[0]:
            cp = CoveragePlanner(grid, cell_values=cell_values)
            cp.start(initial_orientation=orientation,
                     cp_heuristic=HEURISTIC_MAP.get(heuristic_str, HeuristicType.VERTICAL))
            cp.compute(return_home=True)
            res = cp.result()
        self._send_json(200, {
            'found': res[0],
            'steps': res[1],
            'cost': res[2],
            'path': res[4]
        })

def run(port=8000):
    server = HTTPServer(('', port), PlannerHandler)
    print(f'Starting server on port {port}')
    server.serve_forever()

if __name__ == '__main__':
    run()