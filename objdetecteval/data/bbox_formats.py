import math


__all__ = [
    "denormalize_bbox_values",
    "convert_pascal_voc_bbox_to_cxcywh",
    "convert_cxcywh_bbox_to_corner_values",
    "convert_cxcywh_bbox_to_coco_format",
    "convert_cxcywh_bbox_to_pascal_voc_format",
    "convert_corner_bbox_to_pascal_voc",
    "get_rectangle_edges_from_corners_format_bbox",
    "get_rectangle_edges_from_coco_bbox",
    "get_rectangle_edges_from_pascal_bbox",
]


def denormalize_bbox_values(
    normalised_x_centre,
    normalised_y_centre,
    normalised_width,
    normalised_height,
    im_width=2456,
    im_height=2052,
    bbox_format_conversion_fn=None,
):
    x_centre = normalised_x_centre * im_width
    y_centre = normalised_y_centre * im_height
    width = normalised_width * im_width
    height = normalised_height * im_height

    if bbox_format_conversion_fn is None:
        return [
            math.floor(x_centre),
            math.floor(y_centre),
            math.floor(width),
            math.floor(height),
        ]
    else:
        return bbox_format_conversion_fn(x_centre, y_centre, width, height)


def convert_pascal_voc_bbox_to_cxcywh(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    x_centre = xmin + width / 2.0
    y_centre = ymin + height / 2.0

    return [x_centre, y_centre, width, height]


def convert_cxcywh_bbox_to_corner_values(x_centre, y_centre, width, height):
    top = math.floor(y_centre + height / 2)
    left = math.floor(x_centre - width / 2)
    bottom = math.floor(y_centre - height / 2)
    right = math.floor(x_centre + width / 2)
    return [top, left, bottom, right]


def convert_cxcywh_bbox_to_coco_format(x_centre, y_centre, width, height):
    x_min = math.floor(x_centre - width / 2)
    y_min = math.floor(y_centre - height / 2)

    return [x_min, y_min, width, height]


def convert_cxcywh_bbox_to_pascal_voc_format(x_centre, y_centre, width, height):
    xmin = math.floor(x_centre - width / 2)
    ymin = math.floor(y_centre - height / 2)
    xmax = math.floor(x_centre + width / 2)
    ymax = math.floor(y_centre + height / 2)

    return [xmin, ymin, xmax, ymax]


def convert_corner_bbox_to_pascal_voc(top, left, bottom, right):
    xmin = left
    ymin = bottom
    xmax = right
    ymax = top

    return [xmin, ymin, xmax, ymax]


def get_rectangle_edges_from_corners_format_bbox(bbox):
    top, left, bottom, right = bbox

    bottom_left = (left, bottom)
    width = right - left
    height = top - bottom

    return bottom_left, width, height


def get_rectangle_edges_from_coco_bbox(bbox):
    x_min, y_min, width, height = bbox

    bottom_left = (x_min, y_min)

    return bottom_left, width, height


def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def convert_pascal_bbox_to_coco(xmin, ymin, xmax, ymax):
    """
    pascal: top-left-x, top-left-y, x-bottom-right, y-bottom-right
    coco:   top-left-x, top-left-y, width and height
    """
    return [xmin, ymin, xmax - xmin, ymax - ymin]
