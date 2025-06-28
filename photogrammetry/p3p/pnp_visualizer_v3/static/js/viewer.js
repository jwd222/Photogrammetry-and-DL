import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * A modular 3D viewer for visualizing PnP results.
 * It handles scene setup, rendering, and dynamic updates.
 */
class PnPViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with ID "${containerId}" not found.`);
            return;
        }

        // Scene scaling parameters (will be calculated based on data)
        this.sceneCenter = new THREE.Vector3(0, 0, 0);
        this.scaleFactor = 1.0;

        // Core Three.js components
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, this.container.clientWidth / this.container.clientHeight, 0.1, 5000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);

        // Groups to hold dynamic objects for easy clearing
        this.worldPointsGroup = new THREE.Group();
        this.cameraModelGroup = new THREE.Group();
        this.raysGroup = new THREE.Group();

        this.init();
    }

    init() {
        // --- Renderer Setup ---
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x1a1a1a);
        this.container.appendChild(this.renderer.domElement);

        // --- Scene Setup ---
        this.scene.add(new THREE.AmbientLight(0x808080));
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(1, 1, 1);
        this.scene.add(light);
        
        this.scene.add(this.worldPointsGroup);
        this.scene.add(this.cameraModelGroup);
        this.scene.add(this.raysGroup);
        
        // --- Axes Helper ---
        const axesHelper = new THREE.AxesHelper(5); // X=red, Y=green, Z=blue
        this.scene.add(axesHelper);

        // --- Camera and Controls ---
        this.camera.position.set(15, 15, 15);
        this.controls.update();

        // --- Event Listeners ---
        window.addEventListener('resize', this.onWindowResize.bind(this), false);
        
        // --- Start Animation Loop ---
        this.animate();
    }
    
    animate() {
        requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    /**
     * Clears all dynamically added objects (points, camera models, rays) from the scene.
     */
    clearDynamicObjects() {
        const clearGroup = (group) => {
            while (group.children.length > 0) {
                group.remove(group.children[0]);
            }
        };
        clearGroup(this.worldPointsGroup);
        clearGroup(this.cameraModelGroup);
        clearGroup(this.raysGroup);
    }

    /**
     * Calculates the necessary scaling and offset to fit large-coordinate
     * world points into a manageable Three.js scene.
     * @param {Array<Array<number>>} worldPoints - Array of [x, y, z] points.
     */
    _calculateSceneScaling(worldPoints) {
        if (worldPoints.length === 0) {
            this.sceneCenter = new THREE.Vector3(0, 0, 0);
            this.scaleFactor = 1.0;
            return;
        }

        const vectors = worldPoints.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const box = new THREE.Box3().setFromPoints(vectors);
        box.getCenter(this.sceneCenter);

        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        
        // Scale so the largest dimension of the point cloud is ~20 units in the scene
        this.scaleFactor = maxDim > 0 ? 20.0 / maxDim : 1.0;
    }

    /**
     * Transforms a point from world coordinates to scaled scene coordinates.
     * @param {THREE.Vector3} worldVec - The vector in original world coordinates.
     * @returns {THREE.Vector3} The vector in scene coordinates.
     */
    _worldToScene(worldVec) {
        return worldVec.clone().sub(this.sceneCenter).multiplyScalar(this.scaleFactor);
    }

    /**
     * Main function to update the 3D scene with new data.
     * @param {object} data - The data object from the backend API.
     * @param {Array<Array<number>>} originalWorldPoints - The original untransformed world points.
     */
    updateScene(data, originalWorldPoints) {
        this.clearDynamicObjects();
        this._calculateSceneScaling(originalWorldPoints);

        // 1. Draw the world points
        originalWorldPoints.forEach(p => {
            const pos = this._worldToScene(new THREE.Vector3(...p));
            const geometry = new THREE.SphereGeometry(0.2, 16, 16);
            const material = new THREE.MeshStandardMaterial({ color: 0xffa500 }); // Orange
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.copy(pos);
            this.worldPointsGroup.add(sphere);
        });

        if (data.success) {
            const camPos = new THREE.Vector3(...data.camera_position);
            const camQuat = new THREE.Quaternion(...data.camera_quaternion);

            // 2. Draw the camera model
            this._drawCameraModel(camPos, camQuat);

            // 3. Draw rays from camera to points
            originalWorldPoints.forEach(p => {
                const start = this._worldToScene(camPos);
                const end = this._worldToScene(new THREE.Vector3(...p));
                const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
                const material = new THREE.LineBasicMaterial({ color: 0xcccccc });
                const line = new THREE.Line(geometry, material);
                this.raysGroup.add(line);
            });
        }
        
        // Reset camera controls to focus on the new scene center
        this.controls.target.copy(new THREE.Vector3(0,0,0));
        this.controls.update();
    }
    
    /**
     * Creates and adds a visual representation of the camera to the scene.
     * @param {THREE.Vector3} position - The camera's position in world coordinates.
     * @param {THREE.Quaternion} quaternion - The camera's orientation in world coordinates.
     */
    _drawCameraModel(position, quaternion) {
        const cameraGroup = new THREE.Group();

        // The OpenCV camera looks down its +Z axis. Three.js cameras look down -Z.
        // We add a 180-degree rotation around Y to the helper's parent group
        // to align the visualization with the OpenCV convention.
        const alignment = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);

        // Create a dummy camera to generate the frustum shape
        const helperCam = new THREE.PerspectiveCamera(45, 1.33, 0.5, 4);
        const helper = new THREE.CameraHelper(helperCam);
        
        const cameraBody = new THREE.Mesh(
            new THREE.BoxGeometry(0.5, 0.5, 0.8),
            new THREE.MeshStandardMaterial({color: 0x007bff})
        );
        
        // Apply alignment to the visual elements
        const visualGroup = new THREE.Group();
        visualGroup.add(helper);
        visualGroup.add(cameraBody);
        visualGroup.quaternion.copy(alignment);

        cameraGroup.add(visualGroup);

        // Set the final pose of the entire model
        cameraGroup.position.copy(this._worldToScene(position));
        cameraGroup.quaternion.copy(quaternion);

        this.cameraModelGroup.add(cameraGroup);
    }
}


// --- Main Application Logic ---

document.addEventListener('DOMContentLoaded', () => {
    const viewer = new PnPViewer('viewer-container');

    const computeBtn = document.getElementById('compute-btn');
    const resultsDiv = document.getElementById('results');

    // Helper function to parse text area points
    const parsePoints = (text, dimensions) => {
        return text.trim().split('\n')
            .map(line => line.trim().split(/[\s,]+/).map(Number))
            .filter(p => p.length === dimensions && !p.some(isNaN));
    };

    computeBtn.addEventListener('click', async () => {
        resultsDiv.textContent = "Computing...";
        
        // Collect data from the form
        const payload = {
            worldPoints: parsePoints(document.getElementById('world-points').value, 3),
            imagePoints: parsePoints(document.getElementById('image-points').value, 2),
            cameraIntrinsics: {
                fx: parseFloat(document.getElementById('fx').value),
                fy: parseFloat(document.getElementById('fy').value),
                cx: parseFloat(document.getElementById('cx').value),
                cy: parseFloat(document.getElementById('cy').value),
                width: parseInt(document.getElementById('img_w').value),
                height: parseInt(document.getElementById('img_h').value),
            }
        };

        // --- This is the modular API call ---
        try {
            const response = await fetch('http://127.0.0.1:5000/api/solve-pnp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            // Update the 3D scene
            viewer.updateScene(data, payload.worldPoints);

            // Update the results text display
            resultsDiv.textContent = `Pose Found!\n\n` +
                `Camera Position: \n  ${data.camera_position.map(v => v.toFixed(2)).join(', ')}\n` +
                `Camera Quaternion (x,y,z,w): \n  ${data.camera_quaternion.map(v => v.toFixed(4)).join(', ')}\n\n` +
                `Mean Reprojection Error: \n  ${data.reprojection.mean_error_pixels.toFixed(2)} pixels`;

        } catch (error) {
            console.error('API call failed:', error);
            resultsDiv.textContent = `Error: ${error.message}`;
            viewer.clearDynamicObjects(); // Clear scene on failure
        }
    });
    
    // Trigger initial computation on load
    computeBtn.click();
});