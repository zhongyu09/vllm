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

import numpy as np
import pybase64 as base64
import websockets

from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio


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
    audio, _ = load_audio(audio_path, sr=sr, mono=True)
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
        # Use default audio asset
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
