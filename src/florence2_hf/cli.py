import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from florence2_hf.constants import (
    DEFAULT_MODEL_ID,
    TASK_PROMPTS,
    TASKS_REQUIRING_TEXT_INPUT,
)
from florence2_hf.errors import (
    FlorenceError,
    InvalidTaskInputError,
    ResultWriteError,
)
from florence2_hf.runner import FlorenceRunner
from florence2_hf.visualizer import save_annotated_image


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_DIRECTORY = Path("images")
DEFAULT_OUTPUT_DIRECTORY = Path("outputs")

SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verarbeitet einzelne Bilder oder alle Bilder eines Ordners mit Florence-2."
    )

    input_group = parser.add_mutually_exclusive_group()

    input_group.add_argument(
        "--image",
        type=Path,
        help="Pfad zu einem einzelnen Eingabebild.",
    )

    input_group.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIRECTORY,
        help="Ordner mit den Eingabebildern. Standard: images",
    )

    parser.add_argument(
        "--task",
        choices=sorted(TASK_PROMPTS),
        default="caption",
        help="Auszuführende Florence-2-Aufgabe.",
    )

    parser.add_argument(
        "--text-input",
        type=str,
        default=None,
        help="Texteingabe für textabhängige Aufgaben.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging-Face-Modell-ID oder lokaler Modellpfad.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Rechengerät. Standard: automatische Auswahl.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Ordner für JSON-Dateien und markierte Bilder.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Durchsucht auch Unterordner des Eingabeordners.",
    )

    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Zeigt alle unterstützten Aufgaben an.",
    )

    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.list_tasks:
        return

    if arguments.task in TASKS_REQUIRING_TEXT_INPUT and not arguments.text_input:
        raise InvalidTaskInputError(
            f"Die Aufgabe '{arguments.task}' benötigt --text-input."
        )

    if arguments.image is not None and not arguments.image.is_file():
        raise InvalidTaskInputError(
            f"Das Bild wurde nicht gefunden: {arguments.image}"
        )

    if arguments.image is None and not arguments.input_dir.is_dir():
        raise InvalidTaskInputError(
            f"Der Eingabeordner wurde nicht gefunden: {arguments.input_dir}"
        )


def _collect_image_paths(arguments: argparse.Namespace) -> list[Path]:
    if arguments.image is not None:
        return [arguments.image]

    if arguments.recursive:
        candidates = arguments.input_dir.rglob("*")
    else:
        candidates = arguments.input_dir.glob("*")

    image_paths = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]

    return sorted(image_paths)


def _write_result(result: dict[str, Any], output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(result, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as error:
        raise ResultWriteError(
            f"Die JSON-Datei konnte nicht gespeichert werden: {output_path}"
        ) from error


def _build_output_paths(
    image_path: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    json_output_path = output_directory / f"{image_path.stem}.json"
    visual_output_path = output_directory / f"{image_path.stem}_marked.jpg"

    return json_output_path, visual_output_path


def _format_duration(duration_seconds: float) -> str:
    total_milliseconds = round(duration_seconds * 1000)

    hours, remaining_milliseconds = divmod(
        total_milliseconds,
        3_600_000,
    )

    minutes, remaining_milliseconds = divmod(
        remaining_milliseconds,
        60_000,
    )

    seconds, milliseconds = divmod(
        remaining_milliseconds,
        1_000,
    )

    if hours > 0:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}."
            f"{milliseconds:03d}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}."
        f"{milliseconds:03d}"
    )


def _print_tasks() -> None:
    for task_name, task_prompt in sorted(TASK_PROMPTS.items()):
        if task_name in TASKS_REQUIRING_TEXT_INPUT:
            text_requirement = "mit --text-input"
        else:
            text_requirement = "ohne Text"

        print(
            f"{task_name:35} "
            f"{task_prompt:45} "
            f"{text_requirement}"
        )


def _process_image(
    runner: FlorenceRunner,
    image_path: Path,
    arguments: argparse.Namespace,
) -> bool:
    json_output_path, visual_output_path = _build_output_paths(
        image_path=image_path,
        output_directory=arguments.output_dir,
    )

    LOGGER.info("Verarbeite Bild: %s", image_path)

    result = runner.run(
        image_path=image_path,
        task_prompt=TASK_PROMPTS[arguments.task],
        text_input=arguments.text_input,
    )

    _write_result(
        result=result,
        output_path=json_output_path,
    )

    image_created = save_annotated_image(
        image_path=image_path,
        result=result,
        output_path=visual_output_path,
    )

    LOGGER.info("JSON gespeichert: %s", json_output_path)

    if image_created:
        LOGGER.info(
            "Markiertes Bild gespeichert: %s",
            visual_output_path,
        )
    else:
        LOGGER.warning(
            "Keine markierbaren Objekte für '%s' gefunden.",
            image_path.name,
        )

    return image_created


def main() -> None:
    _configure_logging()
    load_dotenv()

    total_start_time = time.perf_counter()

    try:
        arguments = _parse_arguments()
        _validate_arguments(arguments)

        if arguments.list_tasks:
            _print_tasks()
            return

        image_paths = _collect_image_paths(arguments)

        if not image_paths:
            raise InvalidTaskInputError(
                "Im Eingabeordner wurden keine unterstützten Bilder gefunden."
            )

        arguments.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_id = arguments.model_id or os.getenv(
            "FLORENCE_MODEL_ID",
            DEFAULT_MODEL_ID,
        )

        runner = FlorenceRunner(
            model_id=model_id,
            device_name=arguments.device,
        )

        successful_images = 0
        images_with_regions = 0
        failed_images = 0

        LOGGER.info(
            "%d Bilder wurden gefunden.",
            len(image_paths),
        )

        for image_index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            image_start_time = time.perf_counter()

            LOGGER.info(
                "Bild %d von %d: %s",
                image_index,
                len(image_paths),
                image_path.name,
            )

            try:
                image_created = _process_image(
                    runner=runner,
                    image_path=image_path,
                    arguments=arguments,
                )

                successful_images += 1

                if image_created:
                    images_with_regions += 1
            except FlorenceError:
                failed_images += 1

                LOGGER.exception(
                    "Das Bild '%s' konnte nicht verarbeitet werden.",
                    image_path,
                )
            finally:
                image_duration = time.perf_counter() - image_start_time

                LOGGER.info(
                    "Dauer für '%s': %s",
                    image_path.name,
                    _format_duration(image_duration),
                )

        total_duration = time.perf_counter() - total_start_time

        LOGGER.info(
            "Verarbeitung abgeschlossen: %d erfolgreich, "
            "%d mit Markierungen, %d fehlgeschlagen.",
            successful_images,
            images_with_regions,
            failed_images,
        )

        LOGGER.info(
            "Gesamtdauer: %s",
            _format_duration(total_duration),
        )

        if failed_images > 0:
            raise SystemExit(1)
    except FlorenceError:
        total_duration = time.perf_counter() - total_start_time

        LOGGER.exception(
            "Die Florence-2-Verarbeitung wurde abgebrochen."
        )

        LOGGER.info(
            "Gesamtdauer bis zum Abbruch: %s",
            _format_duration(total_duration),
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()