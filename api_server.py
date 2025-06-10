import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from coverage_planner import CoveragePlanner, HeuristicType

HEURISTIC_MAP = {
    'manhattan': HeuristicType.MANHATTAN,
    'chebyshev': HeuristicType.CHEBYSHEV,
    'vertical': HeuristicType.VERTICAL,
    'horizontal': HeuristicType.HORIZONTAL,
}

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
        sr, sc = start
        if not (0 <= sr < rows and 0 <= sc < cols):
            self._send_json(400, {'error': 'start out of bounds'})
            return
        grid[sr][sc] = 2
        cp = CoveragePlanner(grid)
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