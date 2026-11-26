from dataclasses import dataclass

from dataclasses_json import dataclass_json

from modules.Speech_Recognition.TranscribedData import TranscribedData


@dataclass_json
@dataclass
class TranscriptionResult:
    """Transcription result"""

    transcribed_data: list[TranscribedData]
    detected_language: str

    def rounded(self, decimals: int = 3) -> "TranscriptionResult":
        """Word timings rounded to 1 ms. A fresh transcription (float32) and the same one read back
        from the JSON cache (float64) differ by ~1e-7 s; downstream models (GAME chunk cuts) react
        to that, so fresh and cached runs gave different charts. Rounding makes them identical."""
        for t in self.transcribed_data:
            if t.start is not None and t.end is not None:   # untimed words stay None (words.place_untimed)
                t.start, t.end = round(float(t.start), decimals), round(float(t.end), decimals)
        return self
