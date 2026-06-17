# novelty.py — Embedding-based dedup gate before storing correction pairs.
#
# Implements D006: discard a sample when BOTH speaker similarity AND semantic
# similarity exceed their configured thresholds against the existing corpus.
# Uses pure numpy (no heavy deps) so this module has zero inference overhead.
#
# Public API:
#   is_novel(audio, text) -> bool   — True = store this pair; False = already known

import numpy as np

from voice_agent.config import settings
from voice_agent.learning import store

# Dimension constants — changing these invalidates any existing corpus.npz
_N_MELS     = 20   # mel filterbank bands
_SPEAKER_DIM = _N_MELS * 2   # mean + std across time = 40 dims
_SEMANTIC_DIM = 26            # character frequency histogram (a–z)
_EMBED_DIM   = _SPEAKER_DIM + _SEMANTIC_DIM  # 66 total


def is_novel(audio: np.ndarray, text: str) -> bool:
    """Return True if (audio, text) is novel enough to be worth storing.

    Computes a 66-dim combined embedding, checks cosine similarity against
    all previously stored embeddings. If both the speaker and semantic
    sub-dimensions exceed their thresholds, the pair is considered redundant.
    Per D006: we only update the corpus when the sample IS novel.
    """
    emb = _combined_embedding(audio, text)
    corpus = store.load_corpus_embeddings()

    if corpus is None or corpus.shape[0] == 0:
        store.save_corpus_embeddings(emb.reshape(1, -1))
        return True

    cfg = settings.load()
    spk_thresh  = cfg["rl"]["novelty_speaker_threshold"]
    sem_thresh  = cfg["rl"]["novelty_semantic_threshold"]

    corpus_spk = corpus[:, :_SPEAKER_DIM]
    corpus_sem = corpus[:, _SPEAKER_DIM:]

    emb_spk = emb[:_SPEAKER_DIM]
    emb_sem = emb[_SPEAKER_DIM:]

    max_spk_sim = _max_cosine_sim(emb_spk, corpus_spk)
    max_sem_sim = _max_cosine_sim(emb_sem, corpus_sem)

    if max_spk_sim > spk_thresh and max_sem_sim > sem_thresh:
        return False  # model already knows this pattern — discard

    # Novel: append embedding to corpus and persist
    updated = np.vstack([corpus, emb.reshape(1, -1)])
    store.save_corpus_embeddings(updated)
    return True


# --- embedding functions ---

def _combined_embedding(audio: np.ndarray, text: str) -> np.ndarray:
    """Return a normalised float32 vector of shape (_EMBED_DIM,)."""
    spk = _speaker_embedding(audio)
    sem = _semantic_embedding(text)
    return np.concatenate([spk, sem]).astype(np.float32)


def _speaker_embedding(audio: np.ndarray) -> np.ndarray:
    """40-dim log-mel filterbank statistics (mean + std across time)."""
    if len(audio) < 400:
        return np.zeros(_SPEAKER_DIM, dtype=np.float32)

    frame_len = 400   # 25 ms at 16 kHz
    hop_len   = 160   # 10 ms

    frames = _frame(audio, frame_len, hop_len)
    window = np.hanning(frame_len)
    windowed = frames * window

    fft_mag = np.abs(np.fft.rfft(windowed, n=512))  # (n_frames, 257)
    power   = fft_mag ** 2

    mel_fb  = _mel_filters(sr=16000, n_fft=512, n_mels=_N_MELS)  # (20, 257)
    mel_spec = mel_fb @ power.T                                    # (20, n_frames)
    log_mel  = np.log(mel_spec + 1e-8)

    mean = log_mel.mean(axis=1)
    std  = log_mel.std(axis=1)
    raw  = np.concatenate([mean, std])
    return _l2_normalize(raw)


def _semantic_embedding(text: str) -> np.ndarray:
    """26-dim normalised character-frequency histogram (a–z)."""
    text = text.lower()
    hist = np.zeros(26, dtype=np.float32)
    for c in text:
        if "a" <= c <= "z":
            hist[ord(c) - ord("a")] += 1.0
    return _l2_normalize(hist)


# --- numpy helpers ---

def _frame(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Split a 1-D array into overlapping frames, shape (n_frames, frame_len)."""
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    idx = np.clip(idx, 0, len(audio) - 1)
    return audio[idx]


def _mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Return a (n_mels, n_fft//2+1) mel filterbank matrix."""
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel  = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sr / 2.0)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = np.array([_mel_to_hz(m) for m in mel_pts])
    bins     = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            fb[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)

    return fb


def _max_cosine_sim(vec: np.ndarray, matrix: np.ndarray) -> float:
    """Return the maximum cosine similarity between vec and any row of matrix."""
    if matrix.shape[0] == 0:
        return 0.0
    vec_norm    = _l2_normalize(vec)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = matrix_norm @ vec_norm
    return float(sims.max())


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)
