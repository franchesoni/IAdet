
# Data Validator
The data we use is always the `ann.json` file. It is validated when the app starts. The validation includes a backup to `ann_backup.json` just in case (it can be removed with `--dont_backup`). 

The `ann.json` file should contain items of the form:
```
{"filepath":"/path/to/img_000123.jpg",
 "pred_bboxes":[],
 "ann_bboxes":[[left, top, right, bottom]],
 "state":"predicted"          // one of ["unseen", "predicted", "annotated"] 
}
```

The data validator will:
- ensure `ann.json` exists and it's writable
- check for the existence of all image files
- check that all the fields are present in all entries
- check that all the values are allowed: each bbox should be a 4-tuple of non-negative integers and the "state" should be one of ["unseen", "predicted", "annotated"]
- backup the file to `ann_backup.json` (unless otherwise specified)

# Annotation Web App
The annotation webapp allows the user to annotate bounding boxes on the images. The allowed actions are:
- previous / next image: when *leaving* an image, the user is *confirming* that the annotation of an image is correct. **We assume all images seen by the user have correct annotations after the user leaves them.**
- clear all: remove all bounding boxes (both annotated and unannotated)
- remove bounding box: right click on a bounding box
- add bounding box: two left clicks

The annotation webapp shows:
- the image
- the annotated bounding boxes (if any), otherwise the predicted bounding boxes (if any), otherwise nothing else

