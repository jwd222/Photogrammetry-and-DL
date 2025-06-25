class AppManager {
    constructor() {
        this.viewer = new ThreeDViewer('#viewer-container');
        
        this.controlsContainer = document.getElementById('task-controls-container');
        this.infoContainer = document.getElementById('info-panel');
        this.taskSelector = document.getElementById('task-selector');

        this.tasks = {
            'pnp': new PnPTask(this.viewer, this.controlsContainer, this.infoContainer),
            // 'triangulation': new TriangulationTask(...) // Future tasks go here
        };
        
        this.activeTaskKey = null;

        this.init();
    }

    init() {
        // Check if there are any tasks to begin with
        const taskKeys = Object.keys(this.tasks);
        if (taskKeys.length === 0) {
            this.controlsContainer.innerHTML = '<div class="status error">No tasks have been defined.</div>';
            return;
        }

        // Populate the dropdown
        taskKeys.forEach(key => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = this.tasks[key].getName();
            this.taskSelector.appendChild(option);
        });

        // Add event listener for task switching
        this.taskSelector.addEventListener('change', (e) => this.switchTask(e.target.value));

        // *** THE FIX IS HERE ***
        // Activate the first task by default using its key directly from the object.
        const firstTaskKey = taskKeys[0];
        this.switchTask(firstTaskKey);
    }

    switchTask(key) {
        // Check if the key is valid
        if (!key || !this.tasks[key]) {
            console.error(`Attempted to switch to an invalid task key: "${key}"`);
            return;
        }

        if (this.activeTaskKey === key) return;

        // Deactivate the current task if there is one
        if (this.activeTaskKey && this.tasks[this.activeTaskKey]) {
            this.tasks[this.activeTaskKey].deactivate();
        }
        
        // Clear the viewer and info panels for a clean slate
        this.viewer.clearAll();
        this.infoContainer.querySelector('#results-info').textContent = 'Select a task and run it.';
        this.infoContainer.querySelector('#reprojection-info').textContent = '';

        // Activate the new task
        this.activeTaskKey = key;
        this.tasks[key].activate();
    }
}

// Initialize the application once the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new AppManager();
});s