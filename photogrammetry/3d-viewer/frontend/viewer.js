class ThreeDViewer {
    constructor(containerSelector) {
        this.container = document.querySelector(containerSelector);
        if (!this.container) throw new Error("Viewer container not found!");

        this.scene = new THREE.Scene();
        this.camera = null;
        this.renderer = null;
        this.labelRenderer = null;
        this.controls = null;
        this.gridHelper = null;
        this.orbitCenter = new THREE.Vector3(0, 0, 0);
        this.managedObjects = new Map(); // To track objects by ID for easy removal

        this._initCore();
        this._initSceneElements();
        this._initControls();
        this._initEventListeners();
        this.animate();
    }

    // --- INITIALIZATION ---

    _initCore() {
        THREE.Object3D.DefaultUp.set(0, 0, 1);
        this.scene.background = new THREE.Color(0x1a1a1a);
        this.camera = new THREE.PerspectiveCamera(75, this.container.clientWidth / this.container.clientHeight, 0.1, 20000);
        this.camera.up.set(0, 0, -1);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);

        this.labelRenderer = new THREE.CSS2DRenderer();
        this.labelRenderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.labelRenderer.domElement.style.position = 'absolute';
        this.labelRenderer.domElement.style.top = '0px';
        this.container.appendChild(this.labelRenderer.domElement);
    }

    _initSceneElements() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(20, 30, 50);
        this.scene.add(directionalLight);

        const axesHelper = new THREE.AxesHelper(5);
        this.scene.add(axesHelper);
        this._createAxesLabels();

        this.gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        this.gridHelper.rotation.x = Math.PI / 2;
        this.scene.add(this.gridHelper);
    }

    _initControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.labelRenderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.screenSpacePanning = false;
        this.controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
        this.resetView();
    }

    _initEventListeners() {
        window.addEventListener('resize', this._onWindowResize.bind(this));
        // Click-to-center logic can remain here as it's a general viewer feature
    }

    // --- PUBLIC API ---

    /**
     * Adds a group of points to the scene.
     * @param {string} id - A unique identifier for this group of points.
     * @param {Array<THREE.Vector3>} points - An array of THREE.Vector3 objects.
     * @param {object} options - { color, size }.
     */
    addPoints(id, points, options = {}) {
        this.remove(id);
        const { color = 0xff0000, size = 0.1 } = options;
        const material = new THREE.MeshLambertMaterial({ color });
        const geometry = new THREE.SphereGeometry(size, 16, 16);
        
        const group = new THREE.Group();
        points.forEach(p => {
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.copy(p);
            group.add(sphere);
        });
        
        this.scene.add(group);
        this.managedObjects.set(id, group);
    }

    /**
     * Adds a camera frustum to the scene.
     * @param {string} id - A unique identifier for this camera.
     * @param {object} pose - { position: THREE.Vector3, rotation: THREE.Euler }.
     * @param {object} options - { color, scale }.
     */
    addCamera(id, pose, options = {}) {
        this.remove(id);
        const { color = 0xffaa00, scale = 1.0 } = options;
        
        const frustumDepth = 1.5 * scale;
        const aspect = 4 / 3;
        const h = frustumDepth;
        const w = h * aspect;

        const vertices = [
            new THREE.Vector3(0, 0, 0), new THREE.Vector3(-w / 2, -h / 2, -h),
            new THREE.Vector3(w / 2, -h / 2, -h), new THREE.Vector3(w / 2, h / 2, -h),
            new THREE.Vector3(-w / 2, h / 2, -h)
        ];
        const indices = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 3, 2, 1, 4, 3];

        const geometry = new THREE.BufferGeometry().setFromPoints(vertices);
        geometry.setIndex(indices);
        const material = new THREE.MeshBasicMaterial({ color, wireframe: true });
        const frustumMesh = new THREE.Mesh(geometry, material);

        const camGroup = new THREE.Group();
        camGroup.add(frustumMesh);
        camGroup.position.copy(pose.position);
        camGroup.setRotationFromEuler(pose.rotation);
        camGroup.rotateX(Math.PI); // Adjust for camera coordinate system

        this.scene.add(camGroup);
        this.managedObjects.set(id, camGroup);
    }

    /**
     * Adds a set of lines to the scene.
     * @param {string} id - A unique identifier for this set of lines.
     * @param {Array<[THREE.Vector3, THREE.Vector3]>} lineSegments - Array of [start, end] vectors.
     * @param {object} options - { color, opacity }.
     */
    addLines(id, lineSegments, options = {}) {
        this.remove(id);
        const { color = 0xffffff, opacity = 0.5 } = options;
        const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity });

        const group = new THREE.Group();
        lineSegments.forEach(segment => {
            const geometry = new THREE.BufferGeometry().setFromPoints(segment);
            const line = new THREE.Line(geometry, material);
            group.add(line);
        });

        this.scene.add(group);
        this.managedObjects.set(id, group);
    }

    /**
     * Removes an object or group of objects from the scene by its ID.
     * @param {string} id - The ID of the object group to remove.
     */
    remove(id) {
        if (this.managedObjects.has(id)) {
            this.scene.remove(this.managedObjects.get(id));
            this.managedObjects.delete(id);
        }
    }

    /** Clears all managed objects from the scene. */
    clearAll() {
        for (const id of this.managedObjects.keys()) {
            this.remove(id);
        }
    }

    resetView() {
        this.orbitCenter.set(0, 0, 0);
        this.camera.position.set(10, 5, -10);
        this.controls.target.copy(this.orbitCenter);
        this.camera.lookAt(this.orbitCenter);
    }

    setGridLevel(z) {
        this.gridHelper.position.z = z;
    }

    // --- PRIVATE HELPERS & LOOP ---

    _createAxesLabels() {
        // This is a static part of the scene, so it doesn't need to be managed
        const axisLength = 5.2;
        const labels = { 'X': [axisLength, 0, 0], 'Y': [0, axisLength, 0], 'Z': [0, 0, axisLength] };
        for (const text in labels) {
            const div = document.createElement('div');
            div.className = 'label';
            div.textContent = text;
            const label = new THREE.CSS2DObject(div);
            label.position.fromArray(labels[text]);
            this.scene.add(label);
        }
    }

    _onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.labelRenderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate = () => {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        this.labelRenderer.render(this.scene, this.camera);
    }
}