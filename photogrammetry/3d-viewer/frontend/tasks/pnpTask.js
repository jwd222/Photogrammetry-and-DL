class PnPTask extends TaskController {
    constructor(viewer, controlsContainer, infoContainer) {
        super(viewer, controlsContainer, infoContainer);
        // We bind 'this' so the method can be used as an event listener correctly.
        this.solvePnPHandler = this.solvePnP.bind(this); 
    }

    // 1. Provides its name for the dropdown menu in the UI.
    getName() {
        return "Perspective-n-Point (PnP)";
    }

    // 2. Provides the specific HTML controls needed for this task.
    getControlsHTML() {
        return `
            <div class="status-container"><div class="status success">Ready.</div></div>
            <div class="input-group">
                <h4>World Points (3D)</h4>
                <div id="pnp-world-points"></div>
            </div>
            <div class="input-group">
                <h4>Image Points (2D Pixels)</h4>
                <div id="pnp-image-points"></div>
            </div>
            <div class="input-group">
                <h4>Camera Intrinsics</h4>
                <div class="point-row">fx: <input type="number" class="param-input" id="pnp-fx" value="53025.0" step="10"></div>
                <div class="point-row">fy: <input type="number" class="param-input" id="pnp-fy" value="53025.0" step="10"></div>
                <div class="point-row">cx: <input type="number" class="param-input" id="pnp-cx" value="13200.0" step="1"></div>
                <div class="point-row">cy: <input type="number" class="param-input" id="pnp-cy" value="8470.0" step="1"></div>
            </div>
            <div class="input-group">
                <h4>Solver Options (RANSAC)</h4>
                <div class="point-row">Reproj. Err: <input type="number" class="param-input" id="pnp-reprojErr" value="5.0" step="0.1"></div>
                <div class="point-row">Iterations: <input type="number" class="param-input" id="pnp-iterations" value="100" step="10"></div>
            </div>
            <button id="pnp-solve-btn">Solve PnP</button>
            <button id="pnp-reset-view-btn">Reset View</button>
        `;
    }

    // 3. Called by AppManager when this task becomes active.
    // It sets up the UI and attaches its event listeners.
    activate() {
        super.activate(); // Injects the HTML from getControlsHTML()
        this.createPointInputs(); // Populates the point containers with default values
        document.getElementById('pnp-solve-btn').addEventListener('click', this.solvePnPHandler);
        document.getElementById('pnp-reset-view-btn').addEventListener('click', () => this.viewer.resetView());
    }

    // Called by AppManager when the user switches to another task.
    deactivate() {
        super.deactivate(); // Clears the HTML, which also removes the event listeners.
    }

    // This is the core logic method for the task.
    async solvePnP() {
        // 4. Gathers data from its own UI controls.
        const payload = this.readInputs();
        this.updateStatus('Solving...', 'loading');
        
        try {
            // 5. Communicates with the Python backend.
            const response = await fetch('http://localhost:5000/solve_pnp', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `Server error ${response.status}`);

            // 6. Uses the generic viewer to visualize the results.
            this.visualizeResults(payload.worldPoints, data);
            
            // 7. Updates the info panels with text results.
            this.updateInfoPanels(data);
            this.updateStatus('Solved successfully!', 'success');

        } catch (error) {
            this.updateStatus(`Error: ${error.message}`, 'error');
            this.viewer.clearAll();
        }
    }
    updateInfoPanels(data) {
        const pos = data.camera_position; const rot = data.rotation_euler_degrees;
        this.infoContainer.querySelector('#results-info').textContent = `--- Camera Pose (World Coords) ---\nPosition:\n  X: ${pos[0].toFixed(3)}\n  Y: ${pos[1].toFixed(3)}\n  Z: ${pos[2].toFixed(3)}\n\nRotation (Euler ZYX extrinsic, deg):\n  Yaw(Z):   ${rot.yaw_z.toFixed(4)}°\n  Pitch(Y): ${rot.pitch_y.toFixed(4)}°\n  Roll(X):  ${rot.roll_x.toFixed(4)}°`;

        let reprojectionInfo = "--- Reprojection Errors ---\n";
        data.reprojection_errors.forEach((error, i) => {
            const inlierStatus = data.inliers.includes(i) ? "(inlier)" : "(outlier)";
            reprojectionInfo += `P${i + 1}: ${error.toFixed(2)}px ${inlierStatus}\n`;
        });
        reprojectionInfo += `\nMean Error: ${data.mean_reprojection_error.toFixed(2)}px`;
        this.infoContainer.querySelector('#reprojection-info').textContent = reprojectionInfo;
    }

    _scalePointsForDisplay(points) {
        if (points.length === 0) return { scaledPoints: [], origin: new THREE.Vector3(), scale: 1 };
        const vectors = points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const origin = vectors[0].clone();
        const centeredPoints = vectors.map(p => p.clone().sub(origin));
        let maxRange = 0;
        centeredPoints.forEach(p => { maxRange = Math.max(maxRange, p.length()); });
        const targetDisplayRange = 10.0;
        const scale = (maxRange < 1e-6) ? 1.0 : targetDisplayRange / maxRange;
        const scaledPoints = centeredPoints.map(p => p.multiplyScalar(scale));
        return { scaledPoints: scaledPoints.map(v => new THREE.Vector3(v.x, v.y, v.z)), origin, scale };
    }
}