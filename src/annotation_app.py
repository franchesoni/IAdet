import json
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.routing import Route

from db_api import DatabaseAPI
from io_data import get_data, export_data


async def homepage(request):
    """Serve the main HTML page"""
    with open("annotation_interface.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


async def read_filenames(request):
    """Read all image filenames from the database"""
    filenames = db.read_filenames()
    return JSONResponse({"filenames": filenames})


async def export(request):
    """Export the current data to a JSON file"""
    data = db.read_all()
    export_path = export_data(data)
    return JSONResponse({"export_path": export_path})


async def read_entry(request):
    """Read a specific image entry from the database and save the annotations of the previous image.
    Requires a POST containing (filename: str, bboxes: List[List[int]], next_filename: str)
    """
    body = await request.json()
    filename = body.get("filename")
    ann_bboxes = body.get("bboxes")
    next_filename = body.get("next_filename")

    assert filename is not None, "filename must be provided"
    assert next_filename is not None, "next_filename must be provided"
    assert isinstance(ann_bboxes, list), "bboxes must be a list of lists"

    if filename != "FIRST":
        db.write_ann_bboxes(filename, ann_bboxes)
    next_entry = db.read_entry(next_filename)
    filename = next_entry["filename"]
    ann_bboxes, pred_bboxes = next_entry["ann_bboxes"], next_entry["pred_bboxes"]
    if ann_bboxes is not None:
        bboxes = ann_bboxes
    elif pred_bboxes is not None:
        bboxes = pred_bboxes
    else:
        bboxes = []
    return FileResponse(path=filename, headers={"bboxes": json.dumps(bboxes)})


def create_app():
    """Create the Starlette application"""
    data = get_data()
    global db
    db = DatabaseAPI()
    db.write_data(data)

    app = Starlette(
        debug=True,
        routes=[
            Route("/", homepage),
            Route("/api/read_filenames", read_filenames, methods=["GET"]),
            Route("/api/read_entry", read_entry, methods=["POST"]),
            Route("/api/export", export, methods=["GET"]),
        ],
    )

    return app


def main():
    """Main function to run the app"""
    import argparse

    parser = argparse.ArgumentParser(description="IAdet Annotation App")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
