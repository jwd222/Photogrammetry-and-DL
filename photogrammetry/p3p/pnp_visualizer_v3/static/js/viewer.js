import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * A reusable, "headless" 3D viewer for visualizing camera poses and points.
 * It knows nothing about the application's UI or data sources.
 */
export class PnPViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) throw new Error(`Container with ID "${containerId}" not found.`);

        this.sceneCenter = new THREE.Vector3(0, 0, 0);
        this.scaleFactor = 1.0;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, this.container.clientWidth / this.container.clientHeight, 0.1, 5000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);

        this.worldPointsGroup = new THREE.Group();
        this.cameraModelGroup = new THREE.Group();
        this.raysGroup = new THREE.Group();

        this._init();
    }

    _init() {
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x1a1a1a);
        this.container.appendChild(this.renderer.domElement);

        this.scene.add(new THREE.AmbientLight(0x808080));
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(1, 1, 1);
        this.scene.add(light);
        
        this.scene.add(this.worldPointsGroup, this.cameraModelGroup, this.raysGroup);
        this.scene.add(new THREE.AxesHelper(5));

        this.camera.position.set(15, 15, 15);
        this.controls.update();

        window.addEventListener('resize', () => this._onWindowResize(), false);
        this._animate();
    }
    
    _animate() {
        requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    _onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    /**
     * Clears all dynamically added objects from the scene.
     */
    clear() {
        const clearGroup = (group) => {
            while (group.children.length > 0) group.remove(group.children[0]);
        };
        clearGroup(this.worldPointsGroup);
        clearGroup(this.cameraModelGroup);
        clearGroup(this.raysGroup);
    }

    /**
     * Main function to update the 3D scene with new data.
     * @param {Array<object>} poses - An array of pose objects from the backend.
     * @param {Array<Array<number>>} originalWorldPoints - The original untransformed world points.
     */
    updateScene(poses, originalWorldPoints) {
        this._calculateSceneScaling(originalWorldPoints);

        originalWorldPoints.forEach(p => {
            const pos = this._worldToScene(new THREE.Vector3(...p));
            const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.2, 16, 16), new THREE.MeshStandardMaterial({ color: 0xffa500 }));
            sphere.position.copy(pos);
            this.worldPointsGroup.add(sphere);
        });

        const colors = [0x007bff, 0xff4136, 0x2ecc40, 0xffdc00, 0xb10dc9, 0xf012be];
        poses.forEach((poseData, index) => {
            if (poseData.success) {
                const camPos = new THREE.Vector3(...poseData.camera_position);
                const color = colors[index % colors.length];
                this._drawCameraModel(camPos, new THREE.Quaternion(...poseData.camera_quaternion), color);
                this._drawRays(camPos, originalWorldPoints);
            }
        });
        
        this.controls.target.copy(new THREE.Vector3(0,0,0));
        this.controls.update();
    }

    _drawRays(cameraPosition, worldPoints) {
        worldPoints.forEach(p => {
            const start = this._worldToScene(cameraPosition);
            const end = this._worldToScene(new THREE.Vector3(...p));
            const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([start, end]), new THREE.LineBasicMaterial({ color: 0xcccccc, transparent: true, opacity: 0.5 }));
            this.raysGroup.add(line);
        });
    }

    _drawCameraModel(position, quaternion, color) {
        const cameraGroup = new THREE.Group();
        const alignment = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
        const helper = new THREE.CameraHelper(new THREE.PerspectiveCamera(45, 1.33, 0.5, 4));
        const cameraBody = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.8), new THREE.MeshStandardMaterial({ color: color }));
        const visualGroup = new THREE.Group();
        visualGroup.add(helper, cameraBody);
        visualGroup.quaternion.copy(alignment);
        cameraGroup.add(visualGroup);
        cameraGroup.position.copy(this._worldToScene(position));
        cameraGroup.quaternion.copy(quaternion);
        this.cameraModelGroup.add(cameraGroup);
    }
    
    // Internal helper methods for scaling
    _calculateSceneScaling(worldPoints) {
        if (worldPoints.length === 0) { this.sceneCenter.set(0, 0, 0); this.scaleFactor = 1.0; return; }
        const box = new THREE.Box3().setFromPoints(worldPoints.map(p => new THREE.Vector3(...p)));
        box.getCenter(this.sceneCenter);
        const maxDim = Math.max(...box.getSize(new THREE.Vector3()).toArray());
        this.scaleFactor = maxDim > 0 ? 20.0 / maxDim : 1.0;
    }
    _worldToScene(worldVec) { return worldVec.clone().sub(this.sceneCenter).multiplyScalar(this.scaleFactor); }
}