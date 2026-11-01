"""
Audio windowing and batching (roadmap T2.4 and T2.5).

T2.4: clips shorter than one 4 s window get a real verdict, and the end of
every clip is analysed. T2.5: the model runs in batches of 32, and very long
files are capped at 300 evenly spaced windows.

The model, librosa and torch are replaced with small fakes.
"""

import contextlib
import io
import types

import numpy as np
import pytest

from conftest import AUDIO_SERVICE

SR = 16000
CHUNK = 4 * SR
STRIDE = 2 * SR


@pytest.fixture
def audio(load_service):
    return load_service(AUDIO_SERVICE)


# --- T2.4: which windows get analysed ---

@pytest.mark.parametrize("seconds, expected_starts", [
    (0, []),
    (2, [0]),               # short clip: one padded window (was: none, so automatic "real")
    (4, [0]),
    (5, [0, 1]),            # second window is aligned to the end
    (9, [0, 2, 4, 5]),
    (60, list(range(0, 57, 2))),  # already ends exactly on the last stride: 29 windows
])
def test_window_starts(audio, seconds, expected_starts):
    starts = audio.window_starts(seconds * SR, CHUNK, STRIDE)
    assert [s / SR for s in starts] == expected_starts


def test_every_sample_is_covered(audio):
    for n in (1, CHUNK - 1, CHUNK + 1, 7 * SR + 123, 61 * SR + 5):
        starts = audio.window_starts(n, CHUNK, STRIDE)
        covered = np.zeros(n, bool)
        for s in starts:
            covered[s:s + CHUNK] = True
        assert covered.all(), n
        assert max(starts) + CHUNK >= n


# --- T2.5: cap on very long files ---

def test_cap_keeps_first_and_last_window_and_spreads_evenly(audio):
    starts = audio.window_starts(60 * 60 * SR, CHUNK, STRIDE)  # one hour
    assert len(starts) == 1799
    capped = audio.cap_windows(starts)
    assert len(capped) == 300
    assert capped[0] == starts[0] and capped[-1] == starts[-1]
    gaps = np.diff(capped)
    assert gaps.max() - gaps.min() <= STRIDE  # evenly spread


def test_short_list_is_not_capped(audio):
    starts = list(range(10))
    assert audio.cap_windows(starts) == starts


# --- The endpoint, with fake model ---

class Arr(np.ndarray):
    """numpy array that also answers .cpu().numpy() like a torch tensor."""

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


@pytest.fixture
def endpoint(audio, monkeypatch):
    """Fake librosa, preprocessing, torch and model. fake_from_second marks where 'fake' audio begins."""
    state = types.SimpleNamespace(seconds=0, fake_from_second=None, batch_sizes=[])

    monkeypatch.setattr(audio.librosa, "load", lambda _p, sr: (np.full(int(state.seconds * SR), 0.1), SR),
                        raising=False)

    def preprocess(chunk, sr, idx, start_time):
        meta = {"index": idx, "start_time": start_time, "is_silence": False, "is_noisy": False}
        return np.array([[start_time + len(chunk) / sr]]), meta  # carries the window's end time

    monkeypatch.setattr(audio, "preprocess_chunk_with_metadata", preprocess)
    # scipy is stubbed in tests; "no spikes" keeps the decision on the statistics rules.
    monkeypatch.setattr(audio, "find_peaks", lambda probs, **_: (np.array([], int), {}))

    def model(batch):
        state.batch_sizes.append(len(batch))
        ends = batch[:, 0]
        fake = state.fake_from_second is not None
        is_fake = (ends > state.fake_from_second) if fake else np.zeros(len(ends), bool)
        return np.stack([np.where(is_fake, -5.0, 5.0), np.where(is_fake, 5.0, -5.0)], axis=1)

    monkeypatch.setattr(audio, "model", model)

    def softmax(x, dim):
        e = np.exp(x - x.max(axis=dim, keepdims=True))
        return (e / e.sum(axis=dim, keepdims=True)).view(Arr)

    fake_torch = types.SimpleNamespace(
        no_grad=contextlib.nullcontext,
        cat=lambda tensors, dim: np.concatenate(tensors, axis=dim),
        nn=types.SimpleNamespace(functional=types.SimpleNamespace(softmax=softmax)),
    )
    monkeypatch.setattr(audio, "torch", fake_torch)

    client = audio.app.test_client()

    def post(seconds, fake_from_second=None):
        state.seconds, state.fake_from_second = seconds, fake_from_second
        state.batch_sizes.clear()
        return client.post("/predict_audio", data={"file": (io.BytesIO(b"x"), "clip.wav")},
                           content_type="multipart/form-data")

    state.post = post
    return state


def test_short_fake_clip_gets_a_real_verdict(endpoint):
    r = endpoint.post(2, fake_from_second=0)
    body = r.get_json()
    assert r.status_code == 200
    assert body["label"] == "fake" and body["confidence"] > 90  # was: "real", 0%
    assert body["segments"][0]["end"] == 2.0  # clamped to the clip length


def test_fake_tail_is_detected(endpoint):
    # 9 s clip, fake after 7.5 s. Old windows (0-4, 2-6, 4-8) saw only one fake
    # window and said "real"; the extra end window (5-9) tips it to "fake".
    body = endpoint.post(9, fake_from_second=7.5).get_json()
    assert body["windows_total"] == 4
    assert body["label"] == "fake"
    assert any(seg["end"] == 9.0 for seg in body["segments"])


def test_empty_audio_returns_422_not_real(endpoint):
    r = endpoint.post(0)
    assert r.status_code == 422  # was: "real", 0%


def test_model_runs_in_batches_of_32(endpoint):
    body = endpoint.post(120).get_json()  # 2 minutes: 59 windows
    assert body["windows_analysed"] == 59 and not body["capped"]
    assert endpoint.batch_sizes == [32, 27]


def test_one_hour_file_is_capped(endpoint):
    body = endpoint.post(60 * 60).get_json()
    assert body["windows_total"] == 1799
    assert body["windows_analysed"] == 300 and body["capped"] is True
    assert max(endpoint.batch_sizes) == 32
    assert body["label"] == "real"
