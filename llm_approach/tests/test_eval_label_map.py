"""label_map: every POC label maps, to a real intent/operation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_approach.eval import datasets, label_map
from llm_approach.schema import Intent, Operation

LABELS_FILE = Path(__file__).resolve().parents[2] / "poc_classifier" / "labels.py"


def poc_labels() -> list[str]:
    spec = importlib.util.spec_from_file_location("_poc_labels", LABELS_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.LABELS)


def test_every_poc_label_maps():
    labels = poc_labels()
    assert len(labels) == 23
    assert set(labels) == set(label_map.POC_TO_NEW)
    for label in labels:
        intent, op = label_map.map_label(label)
        assert intent in Intent.__members__
        assert op is None or op in Operation.__members__


def test_room_followup_always_has_an_operation_and_others_none():
    for label, (intent, op) in label_map.POC_TO_NEW.items():
        assert (intent == "ROOM_FOLLOWUP") == (op is not None), label


def test_known_mappings():
    assert label_map.map_label("SHOW_MORE_OPTIONS") == ("SHOW_MORE", None)
    assert label_map.map_label("COMPARE_ROOMS") == ("ROOM_FOLLOWUP", "COMPARE")
    assert label_map.map_label("ROOM_TABLE") == ("ROOM_FOLLOWUP", "ROOM_DETAILS")  # addition to PLAN 6.4
    assert label_map.intent_of("GREETING") == label_map.intent_of("CLOSING") == "GENERAL_CHAT"
    assert label_map.map_label("NOT_A_LABEL") is None and label_map.map_label(None) is None
    assert "ROOM_TABLE" in label_map.ADDED_TO_PLAN


def test_acceptable_intents():
    assert label_map.acceptable_intents(["UNCLEAR", "OUT_OF_SCOPE", "bogus"]) == {"UNCLEAR", "OUT_OF_SCOPE"}
    assert label_map.acceptable_intents(None) == set()


def test_every_dataset_label_is_mapped():
    for name in datasets.INTENT_DATASETS:
        for item in datasets.load_dataset(name):
            assert item["gold_intent"] is not None, item["item"]


def test_new_only_intents_are_not_poc_targets():
    assert label_map.NEW_ONLY_INTENTS == {"CONVERSATION_RECALL", "CONVERSATION_QUESTION"}
    assert all(intent not in label_map.NEW_ONLY_INTENTS for intent, _ in label_map.POC_TO_NEW.values())
    assert label_map.intent_of("OUT_OF_SCOPE") == "OUT_OF_SCOPE"  # personal chat questions are not remapped
