# 📷 PnP Camera Pose Viewer

This project is a modular and extensible 3D visualization tool using **Three.js** on the frontend and **Flask** on the backend. It is designed to display camera poses and 3D world points based on user input and PnP computations.

---

## 🧠 Project Architecture

The codebase is split into three key layers to ensure **reusability**, **maintainability**, and **clarity**:

### 🔧 `viewer.js` — The Engine

- Contains the `PnPViewer` class.
- **Purpose:** Pure rendering logic using Three.js.
- **Does not depend** on HTML structure, API calls, or user interaction.
- Acts as a **reusable library** for visualizing:
  - Camera frustums
  - Poses
  - World points

---

### 🧪 `app.js` — The Glue (Application Logic)

- Handles user interactions and data flow.
- **Responsibilities:**
  - Collect data from the HTML form
  - Format payload for API requests
  - Call the Flask backend
  - Pass the result to `PnPViewer`
  - Handle button events (`Add Camera`, `Compute`, `Remove`)
- Designed to be **clean, readable, and easily debuggable**.

---

### 🧱 `index.html` — The Template

- Includes a reusable `<template>` block for dynamic camera input fields.
- **Features:**
  - Add new cameras dynamically by cloning the template
  - Minimal static code — dynamic behavior handled via `app.js`
  - Buttons to add/remove camera input groups and trigger computation

---

## ✅ Summary of Benefits

### 🔄 Flexibility
- Add unlimited cameras without changing code.
- Application logic dynamically gathers camera inputs.

### 🧼 Maintainability
- Rendering logic (`PnPViewer`) is fully decoupled from business logic.
- Improvements to visualization won’t affect how data is collected or processed.

### 👁️ Clarity
- Clear data flow:
  1. **Collect** UI input
  2. **Send** API request
  3. **Render** API response
- Easier to read, debug, and test.

### 🚀 Extensibility
Want to add a new feature like **bundle adjustment**? Simply:
- Add a new endpoint in `app.py`
- Add corresponding form/button in `index.html`
- Handle logic in `app.js`

**No changes to `viewer.js` are required** as long as data format stays consistent.

---

## 🧩 File Structure

```

project/
├── static/
│   ├── viewer.js       # The reusable 3D rendering engine
│   └── app.js          # The application controller / logic
├── templates/
│   └── index.html      # The dynamic user interface
└── app.py              # Flask backend for processing API calls

````

---

## 🔜 Planned Features

- [ ] Bundle Adjustment support
- [ ] Load/save camera configurations
- [ ] Additional 3D overlays (e.g., coordinate axes, camera field of view)
- [ ] Download screenshot / export pose data

---

## 📎 Getting Started

### Frontend
1. Open `index.html` in a browser (served via Flask).
2. Use the UI to add camera inputs.
3. Click **Compute** to send data to backend and view 3D pose.

### Backend
Start the Flask server:
```bash
python app.py
````

Ensure your API returns results in the following format:

```json
{
  "poses": [...],
  "worldPoints": [...]
}
```

---

## 🧠 Notes

* Make sure Three.js is included in your project (via CDN or bundler).
* This architecture follows modern best practices for building **modular, data-driven** web applications.

---

## 📃 License

MIT License. Use freely with attribution.
