import glob
import os.path
import PIL.Image
from PIL import ImageTk
from tkinter import (
    Tk,
    Canvas,
    Frame,
    Label,
    Button,
    messagebox,
    N,
    S,
    E,
    W,
)
from mmengine import load, dump
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules



def show_help(*args):
    messagebox.showinfo(
        title="How to use",
        message="\
    \nMouse controls: \
    \nCreate bounding box: Left click (x2) \
    \nRemove closest bounding box: Right click \
    \n\nKey controls: \
    \nPrevious Image: <p>, <a>, Left or BackSpace \
    \nNext Image: <n>, <d>, Right or Space \
    \nClear all boxes in the image: Delete \
    \n\nGood to know: \
    \nCurrent bounding boxes are saved as annotations when changing the image. \
    \nClose the interface to finish. \
    \nColor code: \
    \n  Green: model prediction. \
    \n  Red: new bounding box. \
    \n  Black: previous annotation. \
    \n\nFeedback at: marchesoniacland@gmail.com \
    ",
    )


class IAdetApp(Frame):
    def __init__(
        self, filenames, root=None, canvas_width=500, file_list_lim=20
    ):
        super().__init__(master=root)
        # Help message
        self.help_btn = Button(self, background="white", text="Help")
        self.help_btn.grid(row=1, column=1)
        self.help_btn.bind("<Button>", show_help)

        # File management
        self.filenames = [str(fname) for fname in filenames]
        self.im_ind = 0
        self.file_list_lim = file_list_lim

        self.file_list = Label(self)
        self.file_list.grid(row=0, column=1)

        self.bind_all("<Left>", self.prev)
        self.bind_all("<a>", self.prev)
        self.bind_all("<BackSpace>", self.prev)
        self.bind_all("<p>", self.prev)
        self.bind_all("<Right>", self.next)
        self.bind_all("<d>", self.next)
        self.bind_all("<space>", self.next)
        self.bind_all("<n>", self.next)

        # Annotation files
        annotated_filename, to_annotate_filename = (
            "annotated_iadet.json",
            "to_annotate_iadet.json",
        )
        self.annotated, self.to_annotate = self.load_initial_annotations_iadet(
            self.filenames, annotated_filename, to_annotate_filename
        )
        annotated_files = list(self.annotated.keys())
        to_annotate_files = list(self.to_annotate.keys())
        assert sorted(self.filenames) == sorted(
            annotated_files + to_annotate_files
        ), "The existing annotations don't correspond to the dataset"

        self.bind_all("<Delete>", self.clear_bboxes)

        # The image visualizer
        self.canvas_width = canvas_width
        self.clear_button = Button(self, text="Clear")
        self.canvas = Canvas(self, cursor="cross", width=canvas_width)
        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)
        self.label = Label(self, text="Mouse position")
        self.label.grid(row=1, column=0)

        self.canvas.bind("<ButtonPress-3>", self.on_button_press_right)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press_left)
        self.canvas.bind("<B1-Motion>", self.on_move_press_left)
        self.canvas.bind("<Motion>", self.on_any_move)

        self.drawing = False
        self.rect = None
        self.start_x = None
        self.start_y = None

        self.canvas.create_image(0, 0, anchor="nw", tags="image")
        # The bounding boxes' indices
        self.bboxes_ids = []
        # initialize
        filename = self.load_new_image()
        self.ckpt_loadtime = 0
        self.load_bboxes(filename, self.im_ind)
        register_all_modules()
        self.check_for_and_load_model()

    def check_for_and_load_model(self):
        config_name = "faster_iadet"
        ckpt_path = sorted(glob.glob(f"work_dirs/{config_name}/epoch_*.pth"))[-1]
        if not os.path.isfile(ckpt_path):
            return
        mtime = os.path.getmtime(ckpt_path)
        if self.ckpt_loadtime < mtime:
            # load model as in https://mmdetection.readthedocs.io/en/3.x/user_guides/inference.html
            config_file = f"work_dirs/{config_name}/{config_name}.py"
            print("Loading new model")
            self.model = init_detector(config_file, ckpt_path)
            self.ckpt_loadtime = mtime





    @staticmethod
    def load_initial_annotations_iadet(
        filenames, annotated_filename, to_annotate_filename
    ):
        if os.path.isfile(annotated_filename):
            annotated_iadet = load(annotated_filename)  # mmdet format
        else:
            annotated_iadet = {}
        if os.path.isfile(to_annotate_filename):
            to_annotate_iadet = load(to_annotate_filename)
        else:
            to_annotate_iadet = {}
            for filename in filenames:
                pilimg = PIL.Image.open(filename)
                to_annotate_iadet[filename] = dict(
                    img_path=filename,
                    height=pilimg.height,
                    width=pilimg.width,
                    instances=[
                        # "bbox": [0, 0, 10, 20],
                        # "bbox_label": 1,
                        # "ignore_flag": 0
                    ]
                )
        return annotated_iadet, to_annotate_iadet

    @staticmethod
    def iadet2mmdet(annotations_iadet: dict) -> list[dict]:
        """Converts from iadet format (dict of dicts) to iadet format (list of dicts).
        Allows for more operation with mmdetection."""
        return {"metainfo":{}, "data_list":list(annotations_iadet.values())}

    # file management
    def load_new_image(self):
        self.update_text()
        filename = self.filenames[self.im_ind]
        self.set_image(filename)
        return filename  # we need this to load bounding boxes afterwards

    def prev(self, event):
        self.update_annotations(self.filenames[self.im_ind])  # save current
        self.clear_bboxes()
        # get prev
        self.im_ind = max(0, self.im_ind - 1)
        filename = self.load_new_image()
        self.dump_annotations()
        self.load_bboxes(filename, self.im_ind)
        self.check_for_and_load_model()

    def next(self, event):
        self.update_annotations(
            self.filenames[self.im_ind]
        )  # save current, slowest
        self.clear_bboxes()
        # get next
        self.im_ind = min(len(self.filenames) - 1, self.im_ind + 1)
        filename = self.load_new_image()
        self.dump_annotations()
        self.load_bboxes(filename, self.im_ind)
        self.check_for_and_load_model()

    def update_text(self):
        text = "\n".join(
            self.filenames[
                max(0, self.im_ind - self.file_list_lim // 2) : self.im_ind
            ]
            + [">>>" + self.filenames[self.im_ind]]
            + self.filenames[
                self.im_ind
                + 1 : min(
                    len(self.filenames), self.im_ind + self.file_list_lim // 2
                )
            ]
        )
        self.file_list.configure(text=text)

    def set_image(self, image_path):
        self.im = PIL.Image.open(image_path)
        # resize im to max width
        self.wpercent = self.canvas_width / float(self.im.size[0])
        self.hsize = int((float(self.im.size[1]) * float(self.wpercent)))
        self.im = self.im.resize(
            (self.canvas_width, self.hsize), PIL.Image.NEAREST
        )
        # resize canvas to image
        self.canvas.configure(height=self.hsize)
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.itemconfigure("image", image=self.tk_im)

    # bounding box handling
    def load_bboxes(self, filename, ind):
        if filename in self.annotated:
            bboxes = [instance["bbox"] for instance in self.annotated[filename]["instances"]]
            bboxes = self.scale_bboxes(bboxes, mode="load")
            color = "black"
        else:
            if 0 < self.ckpt_loadtime:  # there is a loaded model
                bboxes = [list(t) for t in inference_detector(self.model, filename).pred_instances.bboxes]
            else:
                bboxes = []
            color = "green"
        self.draw_bboxes(bboxes, color=color)

    def update_annotations(self, filename):
        bboxes = self.get_bboxes()
        bboxes = self.scale_bboxes(bboxes, mode="save")
        if filename not in self.annotated:
            self.annotated[filename] = self.to_annotate.pop(filename)
        self.annotated[filename]["instances"] = [dict(bbox=bbox, bbox_label=0, ignore_flag=0) for bbox in bboxes]

    def dump_annotations(self):
        dump(self.annotated, "annotated_iadet.json")
        dump(self.to_annotate, "to_annotate_iadet.json")

    def scale_bboxes(self, bboxes, mode):
        if mode == "load":
            bboxes = [
                [float(bbox[j]) * self.wpercent for j in range(4)]
                for bbox in bboxes
            ]
        elif mode == "save":
            bboxes = [
                [bbox[j] / self.wpercent for j in range(4)] for bbox in bboxes
            ]
        else:
            raise ValueError('mode should be in ["load", "save"]')
        return bboxes

    def get_bboxes(self):
        bboxes = [self.canvas.coords(bbox_id) for bbox_id in self.bboxes_ids]
        return [
            bbox for bbox in bboxes if bbox != []
        ]  # empty bboxes cause problems

    def draw_bboxes(self, bboxes, reset_ids=True, color="black"):
        if reset_ids:
            self.bboxes_ids = []
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=color, tags="rect"
            )
            self.bboxes_ids.append(id)

    def clear_bboxes(self, *args):
        self.canvas.delete("rect")

    # image viewer
    def on_any_move(self, event):
        if self.drawing:
            curX = self.canvas.canvasx(event.x)
            curY = self.canvas.canvasy(event.y)

            # expand rectangle as you drag the mouse
            self.canvas.coords(
                self.rect, self.start_x, self.start_y, curX, curY
            )
        # save mouse drag start position
        self.label.config(text=f"{event.x}, {event.y}")

    def on_button_press_right(self, event):
        self.canvas, self.bboxes_ids = self.delete_bbox(
            self.canvas, self.bboxes_ids, event
        )

    def distance_to_bbox(self, bbox, x, y):
        """Compute L1 distance to bounding box borders"""
        lx, uy = bbox[0], bbox[1]
        rx, by = bbox[2], bbox[3]
        # L1 distance if exterior to bounding box
        dist = (lx - x) * (x < lx) + (x - rx) * (rx < x) + (uy - y) * (y < uy) + (y - by) * (by < y)
        if dist == 0:  # distance if interior to bounding box
            dist = min(x - lx, rx - x, y - uy, by - y)
        return dist



    def delete_bbox(self, canvas, bboxes_ids, event):
        """
        Deletes the bounding box whose border is closest to click.
        In case of a tie takes the last bounding box drawn.
        """
        if 0 < len(bboxes_ids):
            x, y = event.x, event.y
            id_to_remove = None
            min_dist = 99999
            for ind, bbox_id in enumerate(bboxes_ids):
                bbox = canvas.coords(bbox_id)
                dist = self.distance_to_bbox(bbox, x, y)
                if dist <= min_dist:
                    min_dist = dist
                    id_to_remove = bbox_id
                    ind_to_remove = ind
            canvas.delete(id_to_remove)
            del bboxes_ids[ind_to_remove]
        return canvas, bboxes_ids

    def on_button_press_left(self, event):
        # save mouse start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if not self.drawing:
            self.rect = self.canvas.create_rectangle(
                0, 0, 0, 0, outline="red", tags="rect"
            )
            self.bboxes_ids.append(self.rect)
        self.drawing = not self.drawing

    def on_move_press_left(self, event):
        self.on_any_move(event)


def run_app(filenames, canvas_width):
    root_win = Tk()
    root_win.resizable(False, False)
    app = IAdetApp(
        filenames=filenames, root=root_win, canvas_width=canvas_width
    )
    app.pack()
    root_win.mainloop()

