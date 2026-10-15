"""Script projection must not rewrite native snapshots or weaken bounds."""
from types import SimpleNamespace
import unittest
from core.exceptions import ScriptValidationError
from infrastructure.audio.metadata_engine import AudioMetadataEngine
from models.audio_metadata import AudioWritePolicy, RawTagDescriptor
from services.native.tagging_scripts import TaggingScriptEngine


def document(storage, raw):
    return SimpleNamespace(
        probe=SimpleNamespace(detected_format="mp3" if storage == "id3" else "flac"),
        native_tags=SimpleNamespace(storage_kind=storage), raw_tags=tuple(raw))


def validate(custom):
    TaggingScriptEngine._validate_custom_output(
        {tag.name: tag.values for tag in custom}, "projection regression", 1, 1)


def test_opaque_id3_projection_preserves_original_document():
    raw = [RawTagDescriptor(key="GEOB:Serato Overview", values=("x" * 9000,), value_kind="text"),
           RawTagDescriptor(key="COMM::\x00\x00\x00", values=("comment",), value_kind="text"),
           RawTagDescriptor(key="COMM:note:eng", values=("valid comment",), value_kind="text"),
           RawTagDescriptor(key="TXXX:NORMAL", values=("value",), value_kind="text")]
    current = document("id3", raw)
    custom = AudioMetadataEngine().custom_tags(current, AudioWritePolicy())
    assert {tag.name for tag in custom} == {"COMM:note:eng", "NORMAL"}
    validate(custom)
    assert current.raw_tags == tuple(raw)
    assert current.native_tags.storage_kind == "id3"


def ordinary_text_still_obeys_native_bounds(storage, key):
    current = document(storage, [RawTagDescriptor(key=key, values=("x" * 9000,), value_kind="text")])
    custom = AudioMetadataEngine().custom_tags(current, AudioWritePolicy())
    assert len(custom) == 1
    try:
        validate(custom)
    except ScriptValidationError:
        return
    raise AssertionError("Ordinary custom text escaped native bounds")


class ScriptViewTests(unittest.TestCase):
    def test_preservation(self):
        test_opaque_id3_projection_preserves_original_document()

    def test_ordinary_text_bounds(self):
        for storage, key in [
            ("id3", "TXXX:GEOB:ordinary custom text"),
            ("id3", "TXXX:COMM:ordinary\x00custom text"),
            ("id3", "TXXX:TOO_BIG"),
            ("vorbis_comments", "GEOB:ordinary Vorbis text"),
        ]:
            with self.subTest(storage=storage, key=key):
                ordinary_text_still_obeys_native_bounds(storage, key)
