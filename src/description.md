- anns.db is a sqlite database that has entries {'filename': str, 'ann_bboxes': list of lists or None, 'pred_bboxes': same than prev}
- database_utils.py exposes a class that can be used to 1. create the database, 2. write ann bboxes or pred bboxes for an image in the database, 3. get the next unlabeled image in the database, 4. export database to json, 5. import database from json, 6. validate the input format
- annotation_app.py is the backend of the webapp, which 1. receives the annotations and writes them to the database, 2. loads and provides the images and bounding boxes for the annotation app at specific images or the next unlabeled image endpoints. the saving of the annotations is done in the same request
- annotation_interface.html should 1. ask for previous (from its history) or next image when the user navigates, 2. keep track of visited image history to allow navigation to previously seen images, 3. ensure requests for next or a specific previous image include the annotations to be saved (of the image being left), 4. block navigation (next/previous buttons disabled) until the save operation for the current image's annotations is confirmed successful and the new image data is loaded and displayed.

+-----------------------------+
|  annotation_interface.html  |   <-- Frontend
|-----------------------------|
| - next / prev nav           |
| - zoom / pan / undo / clear |
| - has image list            |
| - on nav:                   |
|   POST /api/read_entry      |
|   { filename, bboxes,       |
|     [requested_filename] }  |
|                             |
| <- Receives:                |
|   { image, bboxes }         |
+-----------------------------+
        |         ^
        | HTTP    | (JSON)
        v         |
+-----------------------------+
|     annotation_app.py       |   <-- Backend
|-----------------------------|
| POST /api/read_filenames    |
| POST /api/read_entry        |
|  requires (fname,           |
             bboxes,          |
             requested_fname) |
|                             |
| On read_entry:              |
| 1. write_ann_bboxes         |
| 2. read_entry               |
| 3. return { image,          |
|            ann_bboxes or    |
|            pred_bboxes }    |
+-----------------------------+
        |         ^
        | Python  |
        v         |
+-----------------------------+
|    db_api.py                | <-- DB API 
|-----------------------------|
| - write_data                |
| - write_pred_bboxes         |
| - write_ann_bboxes          |
| - read_all                  |
| - read_filenames            |
| - read_entry                |
| - read_next_unlabeled_entry |
+-----------------------------+
        |
        v
+--------------------------------+
|          anns.db               | <-- SQLite
|--------------------------------|
| id INTEGER PRIMARY KEY AUTOINCREMENT 
| filename TEXT UNIQUE NOT NULL  |
| ann_bboxes TEXT -- or NULL     |
| pred_bboxes TEXT -- or NULL    |
+--------------------------------+
