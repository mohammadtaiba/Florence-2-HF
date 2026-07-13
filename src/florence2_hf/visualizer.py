from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from florence2_hf.errors import ImageLoadError, ResultWriteError


class NoRegionsFoundError(Exception):
    """Die Modellantwort enthält keine markierbaren Regionen."""


def save_annotated_image(
    image_path: Path,
    result: dict[str, Any],
    output_path: Path,
) -> bool:
    image = _load_image(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    regions = _extract_regions(result)

    if not regions:
        return False

    for region in regions:
        if region["type"] == "bbox":
            _draw_bbox(
                draw=draw,
                bbox=region["coords"],
                label=region["label"],
                font=font,
            )
        elif region["type"] == "polygon":
            _draw_polygon(
                draw=draw,
                polygon=region["coords"],
                label=region["label"],
                font=font,
            )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    except OSError as error:
        raise ResultWriteError(
            f"Das markierte Bild konnte nicht gespeichert werden: {output_path}"
        ) from error

    return True


def _load_image(image_path: Path) -> Image.Image:
    if not image_path.is_file():
        raise ImageLoadError(f"Das Bild wurde nicht gefunden: {image_path}")

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except OSError as error:
        raise ImageLoadError(
            f"Das Bild konnte nicht geladen werden: {image_path}"
        ) from error


def _extract_regions(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not result:
        return []

    _, payload = next(iter(result.items()))

    if not isinstance(payload, dict):
        return []

    regions = []

    regions.extend(_extract_bboxes(payload))
    regions.extend(_extract_quad_boxes(payload))
    regions.extend(_extract_polygons(payload))

    return regions


def _extract_bboxes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    bboxes = payload.get("bboxes", [])
    labels = payload.get("bboxes_labels") or payload.get("labels") or []

    regions = []

    for index, bbox in enumerate(bboxes):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        label = _get_label(labels, index)
        regions.append(
            {
                "type": "bbox",
                "coords": [float(value) for value in bbox],
                "label": label,
            }
        )

    return regions


def _extract_quad_boxes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    quad_boxes = payload.get("quad_boxes", [])
    labels = payload.get("labels", [])

    regions = []

    for index, quad_box in enumerate(quad_boxes):
        points = _normalize_polygon_points(quad_box)

        if not points:
            continue

        label = _get_label(labels, index)
        regions.append(
            {
                "type": "polygon",
                "coords": points,
                "label": label,
            }
        )

    return regions


def _extract_polygons(payload: dict[str, Any]) -> list[dict[str, Any]]:
    polygons = payload.get("polygons", [])
    labels = payload.get("polygons_labels") or payload.get("labels") or []

    regions = []

    for index, polygon_entry in enumerate(polygons):
        polygons_to_draw = _normalize_polygon_groups(polygon_entry)
        label = _get_label(labels, index)

        for polygon in polygons_to_draw:
            regions.append(
                {
                    "type": "polygon",
                    "coords": polygon,
                    "label": label,
                }
            )

    return regions


def _normalize_polygon_groups(polygon_entry: Any) -> list[list[tuple[float, float]]]:
    if not isinstance(polygon_entry, list):
        return []

    if polygon_entry and isinstance(polygon_entry[0], (int, float)):
        points = _normalize_polygon_points(polygon_entry)
        return [points] if points else []

    polygons = []

    for polygon in polygon_entry:
        points = _normalize_polygon_points(polygon)

        if points:
            polygons.append(points)

    return polygons


def _normalize_polygon_points(values: Any) -> list[tuple[float, float]]:
    if not isinstance(values, (list, tuple)):
        return []

    if len(values) < 6:
        return []

    if len(values) % 2 != 0:
        return []

    points = []

    for index in range(0, len(values), 2):
        x_value = float(values[index])
        y_value = float(values[index + 1])
        points.append((x_value, y_value))

    return points


def _get_label(labels: list[Any], index: int) -> str:
    if index < len(labels):
        return str(labels[index])

    return f"object_{index + 1}"


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: list[float],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = bbox

    draw.rectangle(
        [(x1, y1), (x2, y2)],
        outline="red",
        width=3,
    )

    _draw_label(
        draw=draw,
        x=x1,
        y=y1,
        label=label,
        font=font,
    )


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: list[tuple[float, float]],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.polygon(
        polygon,
        outline="blue",
        width=3,
    )

    first_x, first_y = polygon[0]

    _draw_label(
        draw=draw,
        x=first_x,
        y=first_y,
        label=label,
        font=font,
    )


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = draw.textbbox((x, y), label, font=font)

    padding = 4
    background_box = (
        left - padding,
        top - padding,
        right + padding,
        bottom + padding,
    )

    draw.rectangle(
        background_box,
        fill="yellow",
        outline="black",
    )

    draw.text(
        (x, y),
        label,
        fill="black",
        font=font,
    )