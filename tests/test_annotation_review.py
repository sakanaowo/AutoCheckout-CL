"""Review crops must show both disagreeing geometries, including image edges."""

from copy import deepcopy

import pytest
from PIL import Image

from tools import annotation_review


@pytest.mark.parametrize("bbox,polygons,padding,expected", [
    ([20, 10, 20, 20], [[55, 5, 60, 5, 60, 10, 55, 10]], 4, (16, 1, 65, 34)),
    ([0, 0, 20, 20], [[5, 5, 10, 5, 10, 10, 5, 10]], 8, (0, 0, 28, 28)),
    ([85.5, 66.25, 14.5, 13.75], [[90, 70, 98, 70, 98, 78, 90, 78]], 8, (77, 58, 100, 80)),
])
def test_crop_keeps_bbox_and_all_polygon_fragments_inside_image(bbox, polygons, padding, expected):
    assert annotation_review.review_crop_box(bbox, polygons, (100, 80), padding=padding) == expected


def test_review_card_shows_both_geometries_without_changing_source_or_labels():
    image = Image.new("RGB", (120, 100), (90, 90, 90))
    annotation = {"id": 9, "image_id": 7, "category_id": 196, "bbox": [10, 10, 95, 80],
                  "segmentation": [[40, 30, 85, 30, 85, 75, 40, 75]]}
    case = {"source_task": 4, "annotation_id": 9, "image_id": 7,
            "rpc_category_id": 196, "iou": 0.727}
    original_bytes = image.tobytes()
    original_ann = deepcopy(annotation)

    card = annotation_review.render_review_case(image, annotation, case)

    colors = {color for _, color in card.crop((0, 180, card.width, card.height)).getcolors(card.width * card.height)}
    assert (235, 65, 65) in colors, "Stored bbox must be visible in red"
    assert (0, 180, 225) in colors, "All polygon boundaries must be visible in cyan"
    assert card.width > image.width and card.height > image.height
    assert image.tobytes() == original_bytes and annotation == original_ann
