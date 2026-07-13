"""Projektspezifische Fehlerklassen."""


class FlorenceError(Exception):
    """Basisklasse für erwartete Florence-2-Fehler."""


class ImageLoadError(FlorenceError):
    """Das Eingabebild konnte nicht geladen werden."""


class ModelLoadError(FlorenceError):
    """Das Florence-2-Modell konnte nicht geladen werden."""


class InferenceError(FlorenceError):
    """Die Modellinferenz ist fehlgeschlagen."""


class InvalidTaskInputError(FlorenceError):
    """Die gewählte Aufgabe hat ungültige oder fehlende Eingaben."""


class ResultWriteError(FlorenceError):
    """Das Ergebnis konnte nicht gespeichert werden."""
