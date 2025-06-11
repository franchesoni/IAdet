This document outlines the structure and functionality of the IAdet Annotation Tool.

### Components:

1.  **`anns.db` (SQLite Database)**
    *   Stores image annotation data.
    *   Table: `image_annotations`
        *   `id INTEGER PRIMARY KEY AUTOINCREMENT`
        *   `filename TEXT UNIQUE NOT NULL` (full path to the image file)
        *   `ann_bboxes TEXT` (JSON string of list of lists for annotated bounding boxes, or NULL)
        *   `pred_bboxes TEXT` (JSON string of list of lists for predicted bounding boxes, or NULL)
    *   An index `idx_filename` is created on the `filename` column for faster lookups.
    *   Uses `PRAGMA journal_mode=WAL;` for improved concurrency and performance.

2.  **`io_data.py`**
    *   `get_data()`: A mock function to provide initial sample data if no database is specified. This data includes filenames and optional initial bounding boxes.
    *   `export_data(data)`: Exports the provided data (typically all data from the database) to a JSON file named `exported_data.json`.

3.  **`db_api.py` (Database API)**
    *   Provides a class `DatabaseAPI` to interact with the `anns.db` SQLite database.
    *   **Key Methods:**
        *   `__init__(db_path, timeout)`: Connects to the database and creates the `image_annotations` table and `idx_filename` index if they don't exist. Sets WAL mode.
        *   `_validate_entry(entry)`, `_validate_bboxes(bboxes, bbox_type)`: Internal methods to validate data format before writing to the database. Checks for file existence, correct types, and valid bbox coordinates (x1 < x2, y1 < y2, non-negative integers).
        *   `write_data(data)`: Clears the existing `image_annotations` table and writes a list of new entries.
        *   `write_ann_bboxes(filename, bboxes)`: Updates or inserts annotation bounding boxes for a given image file.
        *   `write_pred_bboxes(filename, bboxes)`: Updates or inserts prediction bounding boxes for a given image file.
        *   `read_all()`: Retrieves all entries from the database.
        *   `read_filenames()`: Retrieves a sorted list of all unique filenames from the database.
        *   `read_entry(filename)`: Retrieves a single entry (filename, ann_bboxes, pred_bboxes) for a given filename.
        *   `read_next_unlabeled_entry()`: Retrieves the first entry (ordered by filename) that does not yet have `ann_bboxes`.
        *   `close()`: Closes the database connection.

4.  **`annotation_app.py` (Backend - Starlette Application)**
    *   Serves the frontend and handles API requests.
    *   Initializes `DatabaseAPI`. If no DB path is provided, it uses `io_data.get_data()` to populate a new in-memory or default `anns.db`.
    *   **API Endpoints:**
        *   `GET /`: Serves the `annotation_interface.html` frontend.
        *   `GET /api/read_filenames`: Returns a JSON list of all image filenames stored in the database.
        *   `POST /api/read_entry`:
            *   **Request Body (JSON):** `{ "filename": "current_image_path.jpg", "bboxes": [[x1,y1,x2,y2], ...], "next_filename": "next_image_path.jpg" }`
                *   `filename`: The filename of the image whose annotations are being saved. Can be "FIRST" for the initial load.
                *   `bboxes`: A list of bounding boxes (annotations) for the `filename`.
                *   `next_filename`: The filename of the image to be loaded and displayed next.
            *   **Functionality:**
                1.  If `filename` is not "FIRST", it saves the provided `bboxes` as `ann_bboxes` for `filename` using `db.write_ann_bboxes()`.
                2.  Retrieves the entry for `next_filename` from the database using `db.read_entry()`.
                3.  Determines which bboxes to send to the frontend for `next_filename`: `ann_bboxes` if available, otherwise `pred_bboxes`, otherwise an empty list.
            *   **Response:** `FileResponse` serving the image file specified by `next_filename`. The bounding boxes (ann or pred) are sent in a custom HTTP header `bboxes` as a JSON string.
        *   `GET /api/export`: Triggers the export of all data from the database to `exported_data.json` via `io_data.export_data()`. Returns a JSON response with the `export_path`.

5.  **`annotation_interface.html` (Frontend - HTML, CSS, JavaScript)**
    *   Provides the user interface for viewing images and annotating bounding boxes.
    *   **Initialization:**
        *   Fetches all filenames via `GET /api/read_filenames`.
        *   Loads the first image by making a `POST /api/read_entry` request with `filename: "FIRST"` and `next_filename` set to the first filename from the fetched list.
    *   **Navigation:**
        *   "Previous" (`◀`) / "Next" (`▶`) buttons and Left/Right arrow keys.
        *   When navigating, it sends a `POST /api/read_entry` request. The request body includes:
            *   `filename`: The path of the currently displayed image.
            *   `bboxes`: The current list of user-drawn bounding boxes for that image.
            *   `next_filename`: The path of the image to be loaded (previous or next in the list).
        *   The UI is blocked (buttons disabled, cursor changes to 'wait') during image loading and saving operations (`isLoading` flag).
    *   **Annotation Features:**
        *   **Add Bounding Box:** Two left-clicks on the image define the opposite corners of a new box.
        *   **Remove Bounding Box:** Right-click near a bounding box side. The closest side of the closest box (within a threshold) will be highlighted on hover, and a right-click removes that box.
        *   **Modify Bounding Box:**
            *   Hovering near a box side highlights that side.
            *   Left-click on a highlighted side to select it for moving.
            *   Move the mouse to drag the selected side.
            *   Left-click again to confirm the new position.
        *   **Pan:** Middle-mouse button drag, or Ctrl+Left-click drag, or Shift+Left-click drag.
        *   **Zoom:** Mouse wheel (zooms in/out centered on the mouse cursor).
        *   **Undo (`Undo` button / Ctrl+Z):** Reverts the last annotation change (add, remove, modify, clear all). Maintains a history of bbox states.
        *   **Clear All (`Clear All` button / Delete or Backspace key):** Removes all bounding boxes from the current image. This action can be undone.
        *   **Reset View (`Reset View` button):** Resets pan and zoom to fit the image within the canvas.
        *   **Export (`Export` button):** Sends a `GET /api/export` request to the backend to save all annotations to a JSON file.
    *   **Display Information:**
        *   Current image: Index, total count, and filename.
        *   Number of bounding boxes on the current image.
        *   Cursor coordinates in image space.
        *   Original image dimensions.
        *   Crosshair cursor.
    *   **Visual Feedback:**
        *   Bounding boxes are drawn with a default style.
        *   Hovered box sides are highlighted (e.g., different color or thicker line).
        *   The side being actively moved/resized is distinctly styled.
        *   A temporary box is shown while drawing a new one.

### Data Flow Diagram:

```
+-----------------------------------+
|    annotation_interface.html      |   <-- Frontend (Browser)
|-----------------------------------|
| - Image Display & BBox Canvas     |
| - Navigation (Prev/Next buttons,  |
|   Arrow Keys)                     |
| - BBox Tools (Add, Remove, Modify)|
| - Pan/Zoom/Undo/Clear/Reset/Export|
| - Displays: Img Info, BBox Count, |
|   Cursor Coords, Img Size         |
|                                   |
| On Startup:                       |
|   GET /api/read_filenames         |
|   -> Receives: { filenames: [...] } |
|                                   |
| On Nav / Initial Load:            |
|   POST /api/read_entry            |
|   { filename: "current_or_FIRST", |
|     bboxes: [[...], ...],         |  (Annotations for 'filename')
|     next_filename: "requested" }  |
|                                   |
|   <- Receives: Image File (body)  |
|      Header: "bboxes": "[JSON_str]"|  (ann_bboxes or pred_bboxes for 'next_filename')
|                                   |
| On Export Click:                  |
|   GET /api/export                 |
|   -> Receives: { export_path: ...}|
+-----------------------------------+
        |         ^
        | HTTP    | (JSON / File Data)
        v         |
+-----------------------------------+
|        annotation_app.py          |   <-- Backend (Starlette)
|-----------------------------------|
| - Serves HTML (GET /)             |
|                                   |
| - GET /api/read_filenames:        |
|   1. db.read_filenames()          |
|   2. Returns { filenames }        |
|                                   |
| - POST /api/read_entry:           |
|   (Requires: filename, bboxes,    |
|             next_filename)        |
|   1. If filename != "FIRST":      |
|      db.write_ann_bboxes(filename,|
|                          bboxes)  |
|   2. entry = db.read_entry(       |
|                  next_filename)   |
|   3. Determine bboxes_to_send     |
|      (ann or pred or [])          |
|   4. Return FileResponse(img_path)|
|      with bboxes_to_send in header|
|                                   |
| - GET /api/export:                |
|   1. data = db.read_all()         |
|   2. path = io.export_data(data)  |
|   3. Returns { export_path }      |
+-----------------------------------+
        |         ^
        | Python  | (Data Objects / Lists)
        v         |
+-----------------------------------+
|            db_api.py              |   <-- DB API / Validation
|-----------------------------------|
| class DatabaseAPI:                |
|  - __init__                       |
|  - _validate_entry, _validate_bboxes|
|  - write_data(data)               |
|  - write_ann_bboxes(fname, bboxes)|
|  - write_pred_bboxes(fname, bboxes)|
|  - read_all()                     |
|  - read_filenames()               |
|  - read_entry(fname)              |
|  - read_next_unlabeled_entry()    |
|  - close()                        |
+-----------------------------------+
        |
        v
+-----------------------------------+
|             anns.db               |   <-- SQLite Database
|-----------------------------------|
| Table: image_annotations          |
|  - id INTEGER PRIMARY KEY AUTOINC |
|  - filename TEXT UNIQUE NOT NULL  |
|  - ann_bboxes TEXT (JSON or NULL) |
|  - pred_bboxes TEXT (JSON or NULL)|
| Index: idx_filename on filename   |
| PRAGMA journal_mode=WAL;          |
+-----------------------------------+
        ^
        | (Initial data if no DB specified)
        |
+-----------------------------------+
|           io_data.py              |
|-----------------------------------|
| - get_data() (mock initial data)  |
| - export_data(data) (to JSON)   |
+-----------------------------------+
```
