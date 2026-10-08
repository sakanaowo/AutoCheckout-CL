"""Split preparation must preserve source pixels and boxes without copying/resizing images."""

from copy import deepcopy

from tools.resize import merge_annotations


def test_native_merge_preserves_geometry_and_resolves_source_id_and_filename_collisions():
    categories = [{"id": 1, "name": "sku", "supercategory": "drink"}]
    name = "20180827-13-42-20-204.jpg"
    cocos = {}
    for source, width, height, box in (
        ("val2019", 1860, 1859, [123.123456, 234.0, 321.5, 456.25]),
        ("test2019", 1840, 1840, [12.0, 34.123456, 56.0, 78.0]),
    ):
        cocos[source] = {
            "categories": categories,
            "images": [{"id": 7, "file_name": name, "width": width, "height": height, "level": "easy"}],
            "annotations": [{"id": 9, "image_id": 7, "category_id": 1, "bbox": box,
                             "area": 12345.123456, "iscrowd": 0}],
        }
    original = deepcopy(cocos)

    merged = merge_annotations(cocos, size=None)

    assert cocos == original
    assert {i["id"] for i in merged["images"]} == {1, 2}
    assert {a["id"] for a in merged["annotations"]} == {1, 2}
    for image, annotation in zip(merged["images"], merged["annotations"], strict=True):
        source = image["source"]
        assert image["file_name"] == f"{source}/{name}"
        assert (image["width"], image["height"]) == (
            original[source]["images"][0]["width"], original[source]["images"][0]["height"]
        )
        assert annotation["image_id"] == image["id"]
        assert annotation["bbox"] == original[source]["annotations"][0]["bbox"]
        assert annotation["area"] == original[source]["annotations"][0]["area"]
        assert annotation["orig_id"] == 9
        assert annotation["source"] == source
