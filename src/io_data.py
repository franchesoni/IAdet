def get_data():
    """Mock function to simulate reading your data.
    You need to implement here what's necessary to get a list of entries following the schema below.
    We require full paths for filenames and None or list of lists of ints for bboxes.
    """
    return [
        {
            "filename": "/home/franchesoni/mine/repos/IAdet/figs/000001.jpg",
            "ann_bboxes": None,
            "pred_bboxes": [[10, 20, 30, 40]],
        },
        {
            "filename": "/home/franchesoni/mine/repos/IAdet/figs/000002.jpg",
            "ann_bboxes": None,
            "pred_bboxes": [[5, 10, 25, 45], [20, 20, 30, 40]],
        },
        {
            "filename": "/home/franchesoni/mine/repos/IAdet/figs/000003.jpg",
            "ann_bboxes": [[50, 60, 70, 80]],
            "pred_bboxes": [[15, 25, 35, 45]],
        },
        {
            "filename": "/home/franchesoni/mine/repos/IAdet/figs/000004.jpg",
            "ann_bboxes": None,
            "pred_bboxes": None,
        },
    ]


def export_data(data):
    """
    This function will export the data to a JSON file. This will be called by the main app.
    You can modify the export format as needed.
    """
    import json

    export_path = "exported_data.json"
    with open(export_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data exported to {export_path}")
    return export_path
