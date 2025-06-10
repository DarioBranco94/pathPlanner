import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import numpy as np
from coverage_planner import CoveragePlanner, HeuristicType
import matplotlib.pyplot as plt
import time

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



def save_debug_image(cell_values, grid, path, start, filename=None):
    """Save a PNG image of the grid with numeric values and the planned path."""
    rows, cols = cell_values.shape
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.imshow(cell_values, cmap="Blues", interpolation="none")

    # grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # show numeric values
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, f"{cell_values[r][c]:.1f}", ha="center", va="center", fontsize=8)

    # obstacles
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color="black"))

    # path arrows
    for i in range(len(path)-1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        ax.arrow(c1, r1, c2-c1, r2-r1, color="red", head_width=0.2, length_includes_head=True)

    if path:
        lr, lc = path[-1]
        ax.scatter([lc], [lr], c="orange", s=40, zorder=3)

    sr, sc = start
    ax.scatter([sc], [sr], c="green", s=40, zorder=3)

    if not filename:
        os.makedirs("output_images", exist_ok=True)
        filename = os.path.join("output_images", f"debug_{int(time.time())}.png")

    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Debug image saved to {filename}")


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

        # Run the planner with a penalty on low-value cells so the search will
        # try to avoid them but still traverse a minimal amount if necessary.
        cp = CoveragePlanner(grid, cell_values=cell_values)

        cp.start(initial_orientation=orientation,
                 cp_heuristic=HEURISTIC_MAP.get(heuristic_str, HeuristicType.VERTICAL))
        cp.compute(return_home=True)
        res = cp.result()

        # Save debug image with numeric values and planned path
        save_debug_image(cell_values, grid, res[4], start)

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
