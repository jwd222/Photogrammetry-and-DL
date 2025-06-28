import { PnPViewer } from './viewer.js';

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize the "Engine"
    const viewer = new PnPViewer('viewer-container');

    // 2. Get DOM elements for the application logic
    const computeBtn = document.getElementById('compute-btn');
    const addTaskBtn = document.getElementById('add-task-btn');
    const tasksContainer = document.getElementById('tasks-container');
    const taskTemplate = document.getElementById('camera-task-template');
    const resultsDiv = document.getElementById('results');

    // --- Main Function to Orchestrate Everything ---
    const runComputation = async () => {
        resultsDiv.textContent = "Computing...";
        
        // A. Dynamically collect all "tasks" from the UI
        const taskElements = tasksContainer.querySelectorAll('.camera-task');
        const tasksPayload = [];
        
        const sharedIntrinsics = {
            fx: parseFloat(document.getElementById('fx').value),
            fy: parseFloat(document.getElementById('fy').value),
            cx: parseFloat(document.getElementById('cx').value),
            cy: parseFloat(document.getElementById('cy').value)
        };

        taskElements.forEach(taskEl => {
            const imagePointsText = taskEl.querySelector('textarea').value;
            tasksPayload.push({
                imagePoints: parsePoints(imagePointsText, 2),
                cameraIntrinsics: sharedIntrinsics // Can be made dynamic later if needed
            });
        });

        // B. Collect the shared world points
        const worldPoints = parsePoints(document.getElementById('world-points').value, 3);

        // C. Build the final payload for the API
        const apiPayload = {
            worldPoints: worldPoints,
            tasks: tasksPayload
        };

        // D. Call the API and update the viewer
        try {
            const response = await fetch('http://127.0.0.1:5000/api/solve-multiple-pnp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(apiPayload)
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            // E. Tell the "Engine" to render the results
            viewer.clear();
            viewer.updateScene(data.results, worldPoints);
            displayResults(data.results);

        } catch (error) {
            console.error('API call failed:', error);
            resultsDiv.textContent = `Error: ${error.message}`;
            viewer.clear();
        }
    };

    // --- Helper and UI Event Listener Functions ---
    
    const parsePoints = (text, dimensions) => {
        return text.trim().split('\n')
            .map(line => line.trim().split(/[\s,]+/).map(Number))
            .filter(p => p.length === dimensions && !p.some(isNaN));
    };

    const displayResults = (results) => {
        let resultsText = `Found ${results.filter(r => r.success).length} Poses:\n\n`;
        results.forEach((res, i) => {
            resultsText += `--- Camera ${i + 1} ---\n`;
            if (res.success) {
                resultsText += `Position: ${res.camera_position.map(v => v.toFixed(2)).join(', ')}\n`;
                resultsText += `Mean Reprojection Error: ${res.reprojection.mean_error_pixels.toFixed(2)} pixels\n\n`;
            } else {
                resultsText += `Failed: ${res.error}\n\n`;
            }
        });
        resultsDiv.textContent = resultsText;
    };

    const addCameraTask = () => {
        const clone = taskTemplate.content.cloneNode(true);
        const taskDiv = clone.querySelector('.camera-task');
        const taskNumber = tasksContainer.children.length + 1;
        taskDiv.querySelector('h4').textContent = `2D Image Points (Camera ${taskNumber})`;
        tasksContainer.appendChild(clone);
    };

    // --- Attach Event Listeners ---
    computeBtn.addEventListener('click', runComputation);
    addTaskBtn.addEventListener('click', addCameraTask);
    
    // Use event delegation for the "Remove" buttons on dynamically added tasks
    tasksContainer.addEventListener('click', (e) => {
        if (e.target && e.target.classList.contains('remove-task-btn')) {
            // Find the parent .camera-task and remove it
            e.target.closest('.camera-task').remove();
            // Re-number the remaining tasks
            tasksContainer.querySelectorAll('.camera-task h4').forEach((h4, index) => {
                h4.textContent = `2D Image Points (Camera ${index + 1})`;
            });
        }
    });

    // --- Initial State ---
    addCameraTask(); // Add the first camera task on load
    computeBtn.click(); // Run initial computation
});