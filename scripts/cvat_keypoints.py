"""Build and read CVAT skeleton shapes for the 42-point face field.

FiftyOne's CVAT uploader sends keypoints as one unlabeled points shape, so a
point can be added or deleted and the index changes. These helpers use a CVAT
skeleton instead: one sublabel per point, in the dataset skeleton order.
A point that is missing in FiftyOne is uploaded as outside and comes back as NaN.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

POINT_COUNT = 42
FACE_LABEL = "kpt_face"
KEYPOINTS_FIELD = "keypoints"
RUNS_KEY = "cvat_keypoint_runs"
COORD_DECIMALS = 6


def skeleton_svg(labels: list[str], edges: list[list[int]]) -> str:
    """Return the inner markup of a CVAT skeleton. Node ids follow ``labels`` order.

    CVAT wraps this string in ``<svg>`` itself. An extra wrapper hides the
    circles from the canvas, and opening the task then fails on ``attr``.

    Circles are element nodes and lines are edges. CVAT's canvas looks up both
    by ``data-type``; without those attributes it cannot draw or activate the
    skeleton. ``data-label-name`` is replaced by the server with ``data-label-id``.
    """
    circles = []
    for index, name in enumerate(labels):
        x = 8 + (index % 7) * 12
        y = 8 + (index // 7) * 14
        circles.append(
            f'<circle r="0.75" cx="{x}" cy="{y}" '
            f'data-type="element node" data-element-id="{index}" data-node-id="{index}" '
            f'data-label-name="{escape(name)}"/>'
        )
    lines = []
    for edge in edges:
        if len(edge) < 2:
            continue
        lines.append(
            f'<line x1="0" y1="0" x2="1" y2="1" data-type="edge" '
            f'data-node-from="{edge[0]}" data-node-to="{edge[1]}"/>'
        )
    return "".join(circles) + "".join(lines)


def box_label_name(label: str) -> str:
    """Return the CVAT rectangle name for a FiftyOne keypoint label."""
    return f"{label}_box"


def require_kpt_label(label: str) -> None:
    """Require a FiftyOne label reserved for the keypoint export."""
    if not label.startswith("kpt"):
        raise ValueError(f"Label must start with 'kpt', found {label}")


def skeleton_label(labels: list[str], edges: list[list[int]], name: str = FACE_LABEL) -> dict:
    """Return one CVAT project label for a 42-point face skeleton."""
    if len(labels) != POINT_COUNT:
        raise ValueError(f"Expected {POINT_COUNT} keypoint names, found {len(labels)}")
    require_kpt_label(name)
    return {
        "name": name,
        "type": "skeleton",
        "svg": skeleton_svg(labels, edges),
        "sublabels": [{"name": point_name, "type": "points", "attributes": []} for point_name in labels],
        "attributes": [],
    }


def rectangle_label(name: str) -> dict:
    """Return the CVAT rectangle label that travels with a face skeleton."""
    return {"name": name, "type": "rectangle", "attributes": []}


def face_project_labels(
    labels: list[dict], skeleton_name: str = FACE_LABEL, box_name: str | None = None
) -> tuple[dict, dict] | None:
    """Return the skeleton and rectangle when the project contains exactly that pair."""
    box_name = box_label_name(skeleton_name) if box_name is None else box_name
    if len(labels) != 2:
        return None
    skeleton = None
    rectangle = None
    for label in labels:
        kind = label.get("type")
        if label.get("name") == skeleton_name and kind in (None, "skeleton") and label.get("sublabels"):
            skeleton = label
        elif label.get("name") == box_name and kind in (None, "rectangle"):
            rectangle = label
    if skeleton is None or rectangle is None:
        return None
    return skeleton, rectangle


def project_sublabel_names(labels: list[dict], skeleton_name: str = FACE_LABEL) -> list[str] | None:
    """Return sublabel names in CVAT order, or None when the face project is absent."""
    found = face_project_labels(labels, skeleton_name)
    if found is None:
        return None
    skeleton, _rectangle = found
    return [str(item.get("name", "")) for item in skeleton.get("sublabels") or []]


def sublabel_ids(label: dict) -> list[int]:
    """Return sublabel ids in the order CVAT stored them."""
    return [int(item["id"]) for item in label.get("sublabels") or []]


def _outside(point: object) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return True
    x, y = float(point[0]), float(point[1])
    return math.isnan(x) or math.isnan(y)


def absolute_point(point: object, width: int, height: int) -> tuple[list[int], bool]:
    """Return CVAT pixel coordinates and whether the point is outside."""
    if _outside(point):
        return [0, 0], True
    x, y = float(point[0]), float(point[1])
    return [int(round(x * width)), int(round(y * height))], False


def skeleton_shape(
    label_id: int,
    sublabel_ids_in_order: list[int],
    points: list,
    width: int,
    height: int,
    frame: int,
    group: int = 0,
) -> dict:
    """Return one CVAT skeleton shape. Missing points are marked outside."""
    if len(points) != len(sublabel_ids_in_order):
        raise ValueError(
            f"Expected {len(sublabel_ids_in_order)} points, found {len(points)}"
        )
    elements = []
    for sublabel_id, point in zip(sublabel_ids_in_order, points, strict=True):
        xy, outside = absolute_point(point, width, height)
        elements.append({
            "type": "points",
            "occluded": False,
            "outside": outside,
            "z_order": 0,
            "rotation": 0,
            "points": xy,
            "frame": frame,
            "label_id": sublabel_id,
            "group": group,
            "source": "manual",
            "attributes": [],
        })
    return {
        "type": "skeleton",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0,
        "points": [],
        "frame": frame,
        "label_id": label_id,
        "group": group,
        "source": "manual",
        "attributes": [],
        "elements": elements,
    }


def rectangle_shape(
    label_id: int,
    bbox: object,
    width: int,
    height: int,
    frame: int,
    group: int,
) -> dict:
    """Return one CVAT rectangle. ``bbox`` is FiftyOne relative top-left xywh."""
    x, y, box_width, box_height = [float(value) for value in list(bbox)]
    return {
        "type": "rectangle",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0,
        "points": [
            int(round(x * width)),
            int(round(y * height)),
            int(round((x + box_width) * width)),
            int(round((y + box_height) * height)),
        ],
        "frame": frame,
        "label_id": label_id,
        "group": group,
        "source": "manual",
        "attributes": [],
    }


def relative_box(xy: object, width: int, height: int) -> list[float]:
    """Return a FiftyOne box from CVAT rectangle points ``[xtl, ytl, xbr, ybr]``."""
    xtl, ytl, xbr, ybr = [float(value) for value in list(xy)[:4]]
    return [
        round(xtl / width, COORD_DECIMALS),
        round(ytl / height, COORD_DECIMALS),
        round((xbr - xtl) / width, COORD_DECIMALS),
        round((ybr - ytl) / height, COORD_DECIMALS),
    ]


def relative_point(xy: object, width: int, height: int, outside: bool) -> tuple[tuple[float, float], int]:
    """Return a FiftyOne point and visibility. Outside points stay NaN at their index."""
    if outside or not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return (float("nan"), float("nan")), 0
    return (
        round(float(xy[0]) / width, COORD_DECIMALS),
        round(float(xy[1]) / height, COORD_DECIMALS),
    ), 1


def label_items(value: object, child_name: str) -> list:
    """Return detections or keypoints from a FiftyOne label container."""
    if value is None:
        return []
    children = getattr(value, child_name, None)
    if children is None:
        return list(value) if isinstance(value, (list, tuple)) else []
    return list(children or [])


def instance_id(label: object) -> str | None:
    """Return the FiftyOne instance id shared by a box and its keypoints."""
    instance = getattr(label, "instance", None)
    if instance is None or getattr(instance, "id", None) is None:
        return None
    return str(instance.id)


def face_pairs(detections: object, keypoints: object, label: str) -> list[tuple]:
    """Pair boxes with keypoint sets of ``label``.

    One box and one keypoint set pair even when their instance ids differ.
    Several faces pair only when each box and keypoint set share an instance id.
    """
    boxes = [item for item in label_items(detections, "detections") if getattr(item, "label", None) == label]
    faces = [item for item in label_items(keypoints, "keypoints") if getattr(item, "label", None) == label]
    if not boxes and not faces:
        return []
    if len(boxes) == 1 and len(faces) == 1:
        return [(boxes[0], faces[0])]
    boxes_by_id: dict[str, object] = {}
    paired: list[tuple] = []
    duplicate = False
    for box in boxes:
        key = instance_id(box)
        if key is None or key in boxes_by_id:
            duplicate = True
            break
        boxes_by_id[key] = box
    if not duplicate:
        for face in faces:
            key = instance_id(face)
            box = boxes_by_id.get(key) if key is not None else None
            if box is None:
                paired = []
                break
            paired.append((box, face))
    if not duplicate and len(paired) == len(faces) == len(boxes):
        return paired
    raise ValueError(
        f"Label '{label}' has {len(boxes)} boxes and {len(faces)} keypoint sets that do not pair"
    )


def skeletons_from_shapes(
    shapes: list[dict],
    frame: int,
    skeleton_id: int,
    node_order: list[int],
    width: int,
    height: int,
) -> list[tuple[list[tuple[float, float]], list[int], int]]:
    """Read one frame's skeletons as ``(points, visibility, group)``."""
    faces = []
    for shape in shapes:
        if int(shape.get("frame", -1)) != frame:
            continue
        if shape.get("type") != "skeleton" or int(shape.get("label_id", -1)) != skeleton_id:
            continue
        by_id = {}
        for element in shape.get("elements") or []:
            by_id[int(element["label_id"])] = element
        points = []
        visibility = []
        for sublabel_id in node_order:
            element = by_id.get(sublabel_id)
            outside = element is None or bool(element.get("outside"))
            xy = None if element is None else element.get("points")
            point, flag = relative_point(xy, width, height, outside)
            points.append(point)
            visibility.append(flag)
        faces.append((points, visibility, int(shape.get("group") or 0)))
    return faces


def faces_from_shapes(
    shapes: list[dict],
    frame: int,
    skeleton_id: int,
    node_order: list[int],
    width: int,
    height: int,
) -> list[tuple[list[tuple[float, float]], list[int]]]:
    """Read one frame's skeleton faces in sublabel order."""
    return [
        (points, visibility)
        for points, visibility, _group in skeletons_from_shapes(
            shapes, frame, skeleton_id, node_order, width, height
        )
    ]


def boxes_from_shapes(
    shapes: list[dict],
    frame: int,
    box_label_id: int,
    width: int,
    height: int,
) -> list[tuple[list[float], int]]:
    """Read one frame's rectangles as ``(relative xywh, group)``."""
    boxes = []
    for shape in shapes:
        if int(shape.get("frame", -1)) != frame:
            continue
        if shape.get("type") != "rectangle" or int(shape.get("label_id", -1)) != box_label_id:
            continue
        points = list(shape.get("points") or [])
        if len(points) < 4:
            continue
        boxes.append((relative_box(points, width, height), int(shape.get("group") or 0)))
    return boxes


def pair_downloaded(
    faces: list[tuple[list, list, int]],
    boxes: list[tuple[list[float], int]],
) -> list[tuple[list[float], list, list]]:
    """Pair downloaded boxes and skeletons by CVAT group.

    A group with only the box or only the skeleton is omitted. Ungrouped shapes
    pair only when the frame has exactly one of each. A larger ungrouped mix raises.
    """
    paired: list[tuple[list[float], list, list]] = []
    used_boxes: set[int] = set()
    ungrouped_faces: list[tuple[list, list]] = []
    for points, visibility, group in faces:
        if not int(group or 0):
            ungrouped_faces.append((points, visibility))
            continue
        match = next(
            (
                index for index, (_box, box_group) in enumerate(boxes)
                if index not in used_boxes and int(box_group or 0) == int(group)
            ),
            None,
        )
        if match is None:
            continue
        used_boxes.add(match)
        paired.append((boxes[match][0], points, visibility))
    ungrouped_boxes = [
        box for index, (box, group) in enumerate(boxes)
        if index not in used_boxes and not int(group or 0)
    ]
    if len(ungrouped_faces) == 1 and len(ungrouped_boxes) == 1:
        paired.append((ungrouped_boxes[0], ungrouped_faces[0][0], ungrouped_faces[0][1]))
    elif ungrouped_faces and ungrouped_boxes:
        raise ValueError(
            f"CVAT frame has {len(ungrouped_boxes)} ungrouped boxes and "
            f"{len(ungrouped_faces)} ungrouped skeletons that do not pair"
        )
    return paired


def box_moved(
    before: object,
    after: object,
    width: int,
    height: int,
    tolerance: float = 5.0,
) -> bool:
    """Return whether any edge of a relative xywh box moved by more than ``tolerance`` pixels."""
    before_box = [float(value) for value in list(before)]
    after_box = [float(value) for value in list(after)]
    if len(before_box) != 4 or len(after_box) != 4:
        return True
    before_edges = [
        before_box[0] * width,
        before_box[1] * height,
        (before_box[0] + before_box[2]) * width,
        (before_box[1] + before_box[3]) * height,
    ]
    after_edges = [
        after_box[0] * width,
        after_box[1] * height,
        (after_box[0] + after_box[2]) * width,
        (after_box[1] + after_box[3]) * height,
    ]
    return any(abs(left - right) > tolerance for left, right in zip(before_edges, after_edges, strict=True))


def points_moved(
    before: list,
    after: list,
    width: int,
    height: int,
    tolerance: float = 5.0,
) -> bool:
    """Return whether any indexed point moved by more than ``tolerance`` pixels."""
    if len(before) != len(after):
        return True
    for left, right in zip(before, after, strict=True):
        left_outside = _outside(left)
        right_outside = _outside(right)
        if left_outside or right_outside:
            if left_outside != right_outside:
                return True
            continue
        dx = abs(float(left[0]) - float(right[0])) * width
        dy = abs(float(left[1]) - float(right[1])) * height
        if dx > tolerance or dy > tolerance:
            return True
    return False
