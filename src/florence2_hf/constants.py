"""Konstanten und unterstützte Florence-2-Aufgaben."""

DEFAULT_MODEL_ID = "microsoft/Florence-2-large-ft"
DEFAULT_OUTPUT_PATH = "outputs/result.json"

TASK_PROMPTS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "od": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
}

TASKS_REQUIRING_TEXT_INPUT = {
    "phrase_grounding",
    "open_vocabulary_detection",
    "referring_expression_segmentation",
}
