class TaskController {
    /**
     * @param {ThreeDViewer} viewer - The main 3D viewer instance.
     * @param {HTMLElement} controlsContainer - The DOM element to inject controls into.
     * @param {HTMLElement} infoContainer - The DOM element for results.
     */
    constructor(viewer, controlsContainer, infoContainer) {
        if (this.constructor === TaskController) {
            throw new Error("Abstract classes can't be instantiated.");
        }
        this.viewer = viewer;
        this.controlsContainer = controlsContainer;
        this.infoContainer = infoContainer;
    }

    /** @returns {string} The display name of the task. */
    getName() {
        throw new Error("Method 'getName()' must be implemented.");
    }

    /** @returns {string} The HTML content for the task's controls. */
    getControlsHTML() {
        throw new Error("Method 'getControlsHTML()' must be implemented.");
    }

    /** Called when the task becomes active. Use to add event listeners. */
    activate() {
        this.controlsContainer.innerHTML = this.getControlsHTML();
    }

    /** Called when the task becomes inactive. Use to clean up. */
    deactivate() {
        this.controlsContainer.innerHTML = '';
    }
    
    /** Utility to update a status message in a standard way. */
    updateStatus(message, type) {
        const statusEl = this.controlsContainer.querySelector('.status-container');
        if (statusEl) {
            statusEl.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
    }
}