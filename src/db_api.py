import sqlite3
import json
import os


class DatabaseAPI:
    def __init__(self, db_path="anns.db", timeout=1.0):
        # connect to db
        self.conn = sqlite3.connect(db_path, timeout=timeout)
        self.cursor = self.conn.cursor()  # shortcut for cursor
        self.cursor.execute("PRAGMA journal_mode=WAL;")

        # create table if not exists
        with self.conn:  # when this block exits, it commits changes
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    ann_bboxes TEXT, -- Stores JSON string of list of lists, or NULL
                    pred_bboxes TEXT  -- Stores JSON string of list of lists, or NULL
                )
            """
            )
            self.cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_filename ON image_annotations (filename);
            """
            )

    def _validate_data(self, data):
        for entry in data:
            self._validate_entry(entry)

    def _validate_entry(self, entry):
        assert "filename" in entry, "Each entry must have a 'filename' key."
        assert isinstance(entry["filename"], str), "'filename' must be a string."
        assert os.path.isfile(
            entry["filename"]
        ), f"File {entry['filename']} does not exist."
        for bbox_type in ["pred", "ann"]:
            assert (
                f"{bbox_type}_bboxes" in entry
            ), f"Each entry must have a '{bbox_type}_bboxes' key."
            assert isinstance(
                entry[f"{bbox_type}_bboxes"], (list, type(None))
            ), f"'{bbox_type}_bboxes' must be a list or None."
            self._validate_bboxes(entry[f"{bbox_type}_bboxes"], bbox_type)

    def _validate_bboxes(self, bboxes, bbox_type):
        assert bboxes is None or (
            all(
                all(isinstance(bbox[i], int) and bbox[i] >= 0 for i in range(4))
                and (bbox[0] < bbox[2])  # x1 < x2  (left < right)
                and (bbox[1] < bbox[3])  # y1 < y2  (top < bottom)
                and isinstance(bbox, list)
                and len(bbox) == 4
                for bbox in bboxes
            )
        ), f"'{bbox_type}_bboxes' must be None or a list of lists with 4 non-negative integers each, but is {bboxes}."

    # --- Write methods --- #
    def write_data(self, data):
        self._validate_data(data)
        assert self.conn and self.cursor
        with self.conn:
            self.cursor.execute("DELETE FROM image_annotations")  # Clear existing data
            self.cursor.executemany(
                """
                INSERT INTO image_annotations (filename, ann_bboxes, pred_bboxes) VALUES (?, ?, ?)
                """,
                [
                    (
                        entry["filename"],
                        (
                            json.dumps(entry.get("ann_bboxes"))
                            if entry.get("ann_bboxes")
                            else None
                        ),
                        (
                            json.dumps(entry.get("pred_bboxes"))
                            if entry.get("pred_bboxes")
                            else None
                        ),
                    )
                    for entry in data
                ],
            )

    def _write_bboxes(self, filename, bboxes, bbox_type):
        assert self.conn and self.cursor
        assert bbox_type in ["ann", "pred"], "bbox_type must be 'ann' or 'pred'."
        self._validate_bboxes(bboxes, bbox_type)
        with self.conn:
            if bboxes is not None:
                bboxes = json.dumps(bboxes)
            self.cursor.execute(
                f"""
                UPDATE image_annotations SET {bbox_type}_bboxes = ? WHERE filename = ?
                """,
                (bboxes, filename),
            )

    def write_pred_bboxes(self, filename, bboxes):
        """Writes prediction bounding boxes (as JSON) for a given filename."""
        self._write_bboxes(filename, bboxes, "pred")

    def write_ann_bboxes(self, filename, bboxes):
        """Writes annotation bounding boxes (as JSON) for a given filename."""
        self._write_bboxes(filename, bboxes, "ann")

    # --- Read methods --- #
    def read_all(self):
        assert self.conn and self.cursor
        with self.conn:  # is not strictly necessary, but good practice
            self.cursor.execute("SELECT * FROM image_annotations")
            rows = self.cursor.fetchall()
        return [
            {
                "filename": row[1],
                "ann_bboxes": json.loads(row[2]) if row[2] else None,
                "pred_bboxes": json.loads(row[3]) if row[3] else None,
            }
            for row in rows
        ]

    def read_filenames(self):
        assert self.conn and self.cursor
        with self.conn:
            self.cursor.execute(
                "SELECT filename FROM image_annotations ORDER BY filename"
            )
            rows = self.cursor.fetchall()
        return [row[0] for row in rows]

    def read_entry(self, filename):
        assert self.conn and self.cursor
        with self.conn:
            self.cursor.execute(
                "SELECT filename, ann_bboxes, pred_bboxes FROM image_annotations WHERE filename = ?",
                (filename,),
            )
            row = self.cursor.fetchone()
        if row:
            return {
                "filename": row[0],
                "ann_bboxes": json.loads(row[1]) if row[1] else None,
                "pred_bboxes": json.loads(row[2]) if row[2] else None,
            }
        return None

    def read_next_unlabeled_entry(self):
        assert self.conn and self.cursor
        with self.conn:
            self.cursor.execute(
                "SELECT filename, ann_bboxes, pred_bboxes FROM image_annotations WHERE ann_bboxes IS NULL ORDER BY filename LIMIT 1"
            )
            row = self.cursor.fetchone()
        if row:
            return {
                "filename": row[0],
                "ann_bboxes": json.loads(row[1]) if row[1] else None,
                "pred_bboxes": json.loads(row[2]) if row[2] else None,
            }
        return None

    def close(self):
        assert self.conn and self.cursor
        self.conn.close()
        self.conn = None
        self.cursor = None
        print("Database connection closed.")


def demo():
    db = DatabaseAPI()
    sample_data = [
        {
            "filename": "/home/franchesoni/mine/repos/IAdet/figs/000001.jpg",
            "ann_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
            "pred_bboxes": [[15, 25, 35, 45]],
        },
        {
            "filename": "/home/franchesoni/mine/repos/IAdet/figs/000002.jpg",
            "ann_bboxes": None,
            "pred_bboxes": [[5, 10, 15, 20]],
        },
        {
            "filename": "/home/franchesoni/mine/repos/IAdet/figs/000003.jpg",
            "ann_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
            "pred_bboxes": [[5, 10, 15, 20]],
        },
    ]
    db.write_data(sample_data)
    print("all data:")
    print(db.read_all())
    print("filenames:")
    print(db.read_filenames())
    print("second entry:")
    print(db.read_entry(db.read_filenames()[1]))
    print("next unlabeled entry:")
    print(db.read_next_unlabeled_entry())
    print("writing new pred_bboxes for first entry:")
    db.write_pred_bboxes(
        sample_data[0]["filename"], [[12, 22, 32, 42], [52, 62, 72, 82]]
    )
    print(db.read_entry(sample_data[0]["filename"]))
    db.close()


if __name__ == "__main__":
    demo()
