import numpy as np
import cv2
# <<< CHANGE #1: ADD send_from_directory
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# --- Core PnP Calculation Logic (This part is correct and needs no changes) ---
def calculate_pnp_pose(world_points, image_points, camera_matrix, use_estimate, est_pos, est_rot):
    if len(world_points) < 4 or len(image_points) < 4:
        return {"error": "Need at least 4 points for solvePnPRansac."}
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    world_points_np = np.array(world_points, dtype=np.float64).reshape(-1, 1, 3)
    image_points_np = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    camera_matrix_np = np.array(camera_matrix, dtype=np.float64)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        world_points_np, image_points_np, camera_matrix_np, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return {"error": "solvePnPRansac failed to find a solution."}
    R, _ = cv2.Rodrigues(rvec)
    camera_position = (-R.T @ tvec).flatten().tolist()
    projected_image_points, _ = cv2.projectPoints(world_points_np, rvec, tvec, camera_matrix_np, dist_coeffs)
    error = cv2.norm(image_points_np, projected_image_points.reshape(-1, 1, 2), cv2.NORM_L2)
    mean_error = error / len(world_points)
    results = {
        "success": True,
        "camera_pose": {"position": camera_position, "rotation_matrix": R.T.tolist()},
        "mean_reprojection_error": mean_error,
        "inliers": inliers.flatten().tolist() if inliers is not None else list(range(len(world_points)))
    }
    return results

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# <<< CHANGE #2: ADD THIS NEW ROUTE FOR STATIC FILES
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename, as_attachment=False, mimetype='application/javascript')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute-pnp', methods=['POST'])
def compute_pnp_endpoint():
    data = request.get_json()
    world_points = data.get('world_points')
    image_points = data.get('image_points')
    intrinsics = data.get('intrinsics')
    if not all([world_points, image_points, intrinsics]):
        return jsonify({"error": "Missing required data: world_points, image_points, or intrinsics."}), 400
    camera_matrix = [[intrinsics['fx'], 0, intrinsics['cx']], [0, intrinsics['fy'], intrinsics['cy']], [0, 0, 1]]
    pnp_results = calculate_pnp_pose(world_points, image_points, camera_matrix, False, None, None)
    return jsonify(pnp_results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)