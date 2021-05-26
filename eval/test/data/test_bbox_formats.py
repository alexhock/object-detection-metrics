from unittest.mock import MagicMock, Mock

from pytest import fixture, mark

from eval.data.bbox_formats import (
    convert_cxcywh_bbox_to_corner_values,
    convert_cxcywh_bbox_to_coco_format,
    convert_cxcywh_bbox_to_pascal_voc_format,
    denormalize_bbox_values,
    convert_pascal_voc_bbox_to_cxcywh,
)

X_CENTRE = 1
Y_CENTRE = 1
WIDTH = 2
HEIGHT = 2

IM_WIDTH = 5
IM_HEIGHT = 6


@fixture
def square_bbox():
    return [X_CENTRE, Y_CENTRE, WIDTH, HEIGHT]


def test_can_convert_bbox_to_corner_format(square_bbox):
    expected_bbox = [2, 0, 0, 2]

    converted_bbox = convert_cxcywh_bbox_to_corner_values(*square_bbox)

    assert expected_bbox == converted_bbox


def test_can_convert_bbox_to_coco_format(square_bbox):
    expected_bbox = [0, 0, WIDTH, HEIGHT]

    converted_bbox = convert_cxcywh_bbox_to_coco_format(*square_bbox)

    assert expected_bbox == converted_bbox


def test_convert_bbox_to_pascal_voc_format(square_bbox):
    expected_bbox = [0, 0, 2, 2]

    converted_bbox = convert_cxcywh_bbox_to_pascal_voc_format(*square_bbox)

    assert expected_bbox == converted_bbox


def test_can_denormalize_bbox(square_bbox):
    expected_bbox = [
        X_CENTRE * IM_WIDTH,
        Y_CENTRE * IM_HEIGHT,
        WIDTH * IM_WIDTH,
        HEIGHT * IM_HEIGHT,
    ]

    denormalized_bbox = denormalize_bbox_values(
        *square_bbox, im_height=IM_HEIGHT, im_width=IM_WIDTH
    )

    assert expected_bbox == denormalized_bbox


def test_can_denormalize_and_convert_bbox(square_bbox):
    expected_denormalized_bbox = [
        X_CENTRE * IM_WIDTH,
        Y_CENTRE * IM_HEIGHT,
        WIDTH * IM_WIDTH,
        HEIGHT * IM_HEIGHT,
    ]
    expected_bbox = Mock()
    conversion_fn = MagicMock(return_value=expected_bbox)

    denormalized_bbox = denormalize_bbox_values(
        *square_bbox,
        im_height=IM_HEIGHT,
        im_width=IM_WIDTH,
        bbox_format_conversion_fn=conversion_fn
    )

    conversion_fn.assert_called_once_with(*expected_denormalized_bbox)
    assert expected_bbox == denormalized_bbox


def test_can_convert_pascal_voc_to_cxcywh():
    expected_bbox = [
        X_CENTRE * IM_WIDTH,
        Y_CENTRE * IM_HEIGHT,
        WIDTH * IM_WIDTH,
        HEIGHT * IM_HEIGHT,
    ]
    pascal_voc_box = convert_cxcywh_bbox_to_pascal_voc_format(*expected_bbox)

    converted_box = convert_pascal_voc_bbox_to_cxcywh(*pascal_voc_box)

    assert expected_bbox == converted_box
