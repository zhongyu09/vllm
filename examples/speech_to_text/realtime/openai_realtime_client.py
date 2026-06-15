# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script demonstrates how to use the vLLM Realtime WebSocket API to perform
audio transcription by uploading an audio file.

Before running this script, you must start the vLLM server with a realtime-capable
model, for example:

    vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --enforce-eager

Requirements:
- vllm with audio support
- websockets
- numpy

The script:
1. Connects to the Realtime WebSocket endpoint
2. Converts an audio file to PCM16 @ 16kHz
3. Sends audio chunks to the server
4. Receives and prints transcription as it streams
"""

import argparse
import asyncio
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
import pybase64 as base64
import websockets


def _load_audio(
    path: Union[BytesIO, Path, str],
    *,
    sr: float = 16000,
    mono: bool = True,
) -> tuple:
    """Load an audio file without depending on the full vLLM import chain.

    Tries soundfile first, falls back to PyAV (FFmpeg). Returns
    ``(waveform, sample_rate)`` where *waveform* is a 1-D float32 array.
    """
    try:
        import soundfile  # type: ignore[import]

        with soundfile.SoundFile(path) as f:
            native_sr = f.samplerate
            y = f.read(dtype="float32", always_2d=False).T

        if mono and y.ndim > 1:
            y = np.mean(y, axis=tuple(range(y.ndim - 1)))

        if sr is not None and int(sr) != native_sr:
            y = _resample_pyav(y, orig_sr=native_sr, target_sr=int(sr))
            return y, int(sr)
        return y, native_sr
    except Exception:
        pass

    import av  # type: ignore[import]

    with av.open(path) as container:
        stream = container.streams.audio[0]
        native_sr = stream.rate
        target_sr = int(sr) if sr is not None else native_sr
        needs_resample = target_sr != native_sr
        resampler = (
            av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            if needs_resample or mono
            else None
        )
        chunks = []
        for frame in container.decode(stream):
            if resampler is not None:
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
            else:
                chunks.append(frame.to_ndarray())

    if not chunks:
        raise ValueError("No audio found in the file.")

    audio = np.concatenate(chunks, axis=-1).astype(np.float32)
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    return audio, target_sr


def _resample_pyav(
    audio: np.ndarray,
    *,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample a 1-D float32 waveform using PyAV / libswresample."""
    if orig_sr == target_sr:
        return audio

    import av  # type: ignore[import]

    expected_len = int(math.ceil(len(audio) * target_sr / orig_sr))
    min_samples = 1024
    padded = audio
    if len(audio) < min_samples:
        padded = np.concatenate(
            [audio, np.zeros(min_samples - len(audio), dtype=np.float32)]
        )

    frame = av.AudioFrame.from_ndarray(
        padded[np.newaxis, :], format="fltp", layout="mono"
    )
    frame.sample_rate = orig_sr

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    out_chunks = []
    for out in resampler.resample(frame):
        out_chunks.append(out.to_ndarray())
    for out in resampler.resample(None):
        out_chunks.append(out.to_ndarray())

    if not out_chunks:
        return audio
    result = np.concatenate(out_chunks, axis=-1).flatten().astype(np.float32)
    return result[:expected_len]


def audio_to_pcm16_base64(
    audio_path: str,
    start_ms: int = 0,
    end_ms: int | None = None,
    pad_with_silence: bool = False,
) -> str:
    """
    Load an audio file and convert it to base64-encoded PCM16 @ 16kHz.

    Args:
        audio_path: Path to the audio file.
        start_ms: Start offset in milliseconds (inclusive). Defaults to 0.
        end_ms: End offset in milliseconds (exclusive). If None or beyond the
            audio length, the actual end of the audio is used.
        pad_with_silence: If True and ``end_ms`` extends past the actual audio
            length, pad the tail with silence so the segment spans the full
            ``[start_ms, end_ms)`` range. Has no effect when ``end_ms`` is None.

    Returns:
        Base64-encoded PCM16 @ 16kHz of the selected segment.
    """
    sr = 16000
    # Load audio and resample to 16kHz mono
    audio, _ = _load_audio(audio_path, sr=sr, mono=True)
    # Slice the requested [start_ms, end_ms) segment, clamping to audio bounds.
    start_sample = max(0, int(start_ms * sr / 1000))
    if end_ms is None:
        end_sample = len(audio)
    else:
        end_sample = int(end_ms * sr / 1000)
    audio = audio[start_sample : min(len(audio), end_sample)]
    # Pad the tail with silence if the requested end runs past the audio.
    if pad_with_silence and end_ms is not None and end_sample > start_sample:
        pad_samples = (end_sample - start_sample) - len(audio)
        if pad_samples > 0:
            audio = np.concatenate([audio, np.zeros(pad_samples, dtype=audio.dtype)])
    # Convert to PCM16
    pcm16 = (audio * 32767).astype(np.int16)
    # Encode as base64
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")


async def realtime_transcribe(
    audio_path: str,
    host: str,
    port: int,
    model: str,
    uri: str | None = None,
    start_ms: int = 0,
    end_ms: int | None = None,
):
    """
    Connect to the Realtime API and transcribe an audio file.
    """
    if uri is None:
        uri = f"ws://{host}:{port}/v1/realtime"

    async with websockets.connect(uri) as ws:
        # Wait for session.created
        response = json.loads(await ws.recv())
        if response["type"] == "session.created":
            print(f"Session created: {response['id']}")
        else:
            print(f"Unexpected response: {response}")
            return

        # Validate model
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # Signal ready to start
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Convert audio file to base64 PCM16
        print(f"Loading audio from: {audio_path}")
        audio_base64 = audio_to_pcm16_base64(
            audio_path, start_ms, end_ms, pad_with_silence=True
        )

        # Send audio in chunks (4KB of raw audio = ~8KB base64)
        chunk_size = 4096
        audio_bytes = base64.b64decode(audio_base64)
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size

        print(f"Sending {total_chunks} audio chunks...")
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )

        # Signal all audio is sent
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        print("Audio sent. Waiting for transcription...\n")

        # Receive transcription
        print("Transcription: ", end="", flush=True)
        while True:
            response = json.loads(await ws.recv())
            if response["type"] == "transcription.delta":
                print(response["delta"], end="", flush=True)
            elif response["type"] == "transcription.done":
                print(f"\n\nFinal transcription: {response['text']}")
                if response.get("usage"):
                    print(f"Usage: {response['usage']}")
                break
            elif response["type"] == "error":
                print(f"\nError: {response['error']}")
                break


def main(args):
    if args.audio_path:
        audio_path = args.audio_path
    else:
        # Use default audio asset — import vllm only when needed
        from vllm.assets.audio import AudioAsset  # noqa: PLC0415

        audio_path = str(AudioAsset("mary_had_lamb").get_local_path())
        print(f"No audio path provided, using default: {audio_path}")

    asyncio.run(
        realtime_transcribe(
            audio_path,
            args.host,
            args.port,
            args.model,
            args.uri,
            args.start_ms,
            args.end_ms,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime WebSocket Transcription Client"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
        help="Model that is served and should be pinged.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="vLLM server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="Full WebSocket URI (e.g. ws://host:port/v1/realtime). "
        "If provided, --host and --port are ignored.",
    )
    parser.add_argument(
        "--start_ms",
        type=int,
        default=0,
        help="Start offset of audio in milliseconds (default: 0).",
    )
    parser.add_argument(
        "--end_ms",
        type=int,
        default=None,
        help="End offset of audio in milliseconds. If not set, uses the full audio.",
    )
    args = parser.parse_args()
    main(args)
