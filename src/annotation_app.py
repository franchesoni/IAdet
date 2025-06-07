import json
import os
from typing import List, Dict, Any
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.routing import Route
from data_validator import validate_data


class AnnotationApp:
    def __init__(self, ann_file: str = "ann.json", dont_backup: bool = False):
        self.ann_file = ann_file
        self.dont_backup = dont_backup
        self.data = []
        self.current_index = 0
        
        # Validate and load data
        self._load_data()

    def _load_data(self):
        """Load and validate the annotation data"""
        try:
            validate_data(self.ann_file, self.dont_backup)
            with open(self.ann_file, 'r') as f:
                self.data = json.load(f)
            print(f"✓ Loaded {len(self.data)} images from {self.ann_file}")
        except Exception as e:
            print(f"✗ Failed to load data: {e}")
            raise

    def _save_data(self):
        """Save the annotation data back to file"""
        try:
            with open(self.ann_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"✓ Saved data to {self.ann_file}")
        except Exception as e:
            print(f"✗ Failed to save data: {e}")
            raise

    def get_current_image_data(self) -> Dict[str, Any]:
        """Get data for the current image"""
        if not self.data or self.current_index >= len(self.data):
            return {}
        
        item = self.data[self.current_index].copy()
        item['index'] = self.current_index
        item['total'] = len(self.data)
        return item

    def update_current_image(self, ann_bboxes: List[List[int]]) -> Dict[str, Any]:
        """Update annotations for current image and mark as annotated"""
        if not self.data or self.current_index >= len(self.data):
            return {"error": "Invalid image index"}
        
        self.data[self.current_index]['ann_bboxes'] = ann_bboxes
        self.data[self.current_index]['state'] = 'annotated'
        self._save_data()
        
        return {"success": True, "message": "Annotations saved"}

    def navigate_to(self, index: int) -> Dict[str, Any]:
        """Navigate to a specific image index"""
        if 0 <= index < len(self.data):
            self.current_index = index
            return self.get_current_image_data()
        return {"error": "Invalid index"}

    def next_image(self) -> Dict[str, Any]:
        """Navigate to next image"""
        return self.navigate_to(self.current_index + 1)

    def prev_image(self) -> Dict[str, Any]:
        """Navigate to previous image"""
        return self.navigate_to(self.current_index - 1)


# Global app instance
app_instance = None


async def homepage(request):
    """Serve the main HTML page"""
    with open('annotation_interface.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(html_content)


async def get_current_image(request):
    """Get current image data"""
    data = app_instance.get_current_image_data()
    return JSONResponse(data)


async def save_annotations(request):
    """Save annotations for current image"""
    body = await request.json()
    ann_bboxes = body.get('ann_bboxes', [])
    result = app_instance.update_current_image(ann_bboxes)
    return JSONResponse(result)


async def navigate(request):
    """Navigate between images"""
    body = await request.json()
    direction = body.get('direction', 'next')
    
    if direction == 'next':
        result = app_instance.next_image()
    elif direction == 'prev':
        result = app_instance.prev_image()
    else:
        index = body.get('index', 0)
        result = app_instance.navigate_to(index)
    
    return JSONResponse(result)


async def serve_image(request):
    """Serve image files"""
    # Get the image path from current data
    current_data = app_instance.get_current_image_data()
    if not current_data or 'filepath' not in current_data:
        return JSONResponse({"error": "No image found"}, status_code=404)
    
    image_path = current_data['filepath']
    if not os.path.exists(image_path):
        return JSONResponse({"error": "Image file not found"}, status_code=404)
    
    return FileResponse(image_path)


def create_app(ann_file: str = "ann.json", dont_backup: bool = False):
    """Create the Starlette application"""
    global app_instance
    app_instance = AnnotationApp(ann_file, dont_backup)
    
    app = Starlette(
        debug=True,
        routes=[
            Route("/", homepage),
            Route("/api/current", get_current_image),
            Route("/api/save", save_annotations, methods=["POST"]),
            Route("/api/navigate", navigate, methods=["POST"]),
            Route("/api/image", serve_image),
        ],
    )
    
    return app


def main():
    """Main function to run the app"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IAdet Annotation App")
    parser.add_argument("--ann_file", default="ann.json", help="Path to ann.json file")
    parser.add_argument("--dont_backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    app = create_app(args.ann_file, args.dont_backup)
    
    print(f"🚀 Starting IAdet Annotation App on http://{args.host}:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
