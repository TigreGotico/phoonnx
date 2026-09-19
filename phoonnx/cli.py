import click
from phoonnx.model_manager import TTSModelManager, TTSModelInfo
import requests
from huggingface_hub.errors import HfHubHTTPError


# --- Helper Function for CLI Output ---
def print_voice_info(voice: TTSModelInfo):
    """Prints detailed info about a single voice."""
    click.echo(f"  ID:          {voice.voice_id}")
    if voice.display_name:
        click.echo(f"  Name:        {voice.display_name}")
    click.echo(f"  Language:    {voice.lang}")
    click.echo(f"  Engine:      {voice.engine.value}")
    click.echo(f"  Phoneme Type: {voice.phoneme_type}")
    click.echo(f"  Model URL:   {voice.model_url}")
    click.echo(f"  Config URL:  {voice.config_url}")
    if voice.requires_reference:
        click.echo(f"  Requires reference: yes (needs speaker_reference / a style path)")
    click.echo("-" * 40)


# --- CLI Group ---
@click.group()
def cli():
    """CLI to manage and download phoonnx TTS voice models."""
    pass


# --- CLI Commands ---

@cli.command(name="update-cache")
@click.option("--no-clear", is_flag=True, help="Do not clear the existing cache before updating. Only adds new voices.")
def update_cache(no_clear):
    """
    Clears the local voice cache, loads the bundled voice indexes (Piper,
    Mimic3, OpenVoiceOS, etc.), and saves them to the cache.
    """
    manager = TTSModelManager()

    if not no_clear:
        click.echo("Clearing local voice cache...")
        manager.clear()
    else:
        click.echo("Loading existing voice cache...")
        manager.load()

    click.echo("Loading bundled voice indexes...")

    try:
        manager.merge_default_voices(store=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred while loading voice lists: {e}", err=True)
        return

    click.echo(f"\nCache updated successfully!")
    click.echo(f"Total voices available: {len(manager.all_voices)}")
    click.echo(f"Total languages supported: {len(manager.supported_langs)}")


@cli.command(name="list-langs")
def list_langs():
    """Lists all supported language codes available in the local cache."""
    manager = TTSModelManager()
    manager.load()

    if not manager.supported_langs:
        click.echo("No languages found. Run 'update-cache' first.")
        return

    click.echo(f"\nSupported Languages ({len(manager.supported_langs)}):")
    for l in manager.supported_langs:
        click.echo(f" - {l}")

@cli.command(name="list-voices")
@click.option("--lang", default=None, help="Filter voices by language code (e.g., 'en-US' or 'pt-PT').")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information for each voice.")
def list_voices(lang, verbose):
    """Lists all available voice models."""
    manager = TTSModelManager()
    manager.load()

    if not manager.all_voices:
        click.echo("No voices found in cache. Run 'update-cache' first.")
        return

    if lang:
        voices = manager.get_lang_voices(lang)
        click.echo(f"\nFound {len(voices)} voices for language '{lang}':")
    else:
        voices = manager.all_voices
        click.echo(f"\nFound {len(voices)} total voices:")

    if not voices:
        click.echo(f"No voices found for '{lang}'.")
        return

    for voice in voices:
        if verbose:
            print_voice_info(voice)
        else:
            if voice.display_name:
                click.echo(f"* {voice.display_name} ({voice.lang})")
            else:
                click.echo(f"* {voice.voice_id} ({voice.lang})")

@cli.command(name="list-available")
def list_available():
    """
    Lists all voice IDs bundled with phoonnx, grouped by source (Piper,
    Mimic3, OVOS, etc.), without downloading any config or model files.
    Use 'download <VOICE_ID>' to fetch a specific voice on demand.
    """
    manager = TTSModelManager()

    available_voices = manager.get_available_voice_ids_by_source()

    total_voices = sum(len(ids) for ids in available_voices.values())
    click.echo(f"\nTotal available voices found (bundled indexes): {total_voices}\n")

    for source, voice_ids in available_voices.items():
        click.echo(f"--- {source.upper()} Voices ({len(voice_ids)}) ---")
        for voice_id in voice_ids:
            click.echo(f"  {voice_id}")
        click.echo("-" * 40)
    click.echo("Hint: Use 'update-cache' to populate the local cache, or use 'download <VOICE_ID>' to download a specific voice directly.")

@cli.command(name="download")
@click.argument("voice_id", type=str)
def download_voice(voice_id):
    """
    Downloads the model, config, and token files for a specific VOICE_ID.
    Looks the voice up in the local cache first (run update-cache to
    populate it); if it isn't cached yet, falls back to the bundled
    voice indexes so a single voice can be downloaded on demand, without
    having to load the whole catalog first.
    """
    manager = TTSModelManager()
    manager.load()

    voice_info = manager.get_voice(voice_id)

    try:
        if voice_info:
            click.echo(f"Attempting to download files for: {voice_id} ({voice_info.lang})")

            click.echo("-> Downloading ONNX model and side files (this may take a while)...")
            voice_info.download_all()

            click.echo(f"\nDownload complete. Files saved to: {voice_info.voice_path}")
        else:
            click.echo(f"Voice ID '{voice_id}' not found in local cache, looking it up in bundled indexes...")
            if not manager.download_voice_by_id(voice_id):
                click.echo(f"Error: Voice ID '{voice_id}' not found.", err=True)
                click.echo("Hint: Run 'phoonnx-voices list-available' to see available voice IDs.", err=True)
                return
            click.echo(f"\nDownload complete.")

    except (HfHubHTTPError, requests.exceptions.RequestException) as e:
        # Voice files come from the HuggingFace hub, so a network failure now
        # surfaces as a hub error; a self-hosted voice still raises the
        # requests error from the direct download path.
        click.echo(f"\nDownload failed due to network error: {e}", err=True)
    except Exception as e:
        click.echo(f"\nAn unexpected error occurred during download: {e}", err=True)


@cli.command(name="add-alignment")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False),
              help="Where to write the patched model. Defaults to "
                   "'<model>.alignment.onnx' next to the original, which is "
                   "where phoonnx looks for it at load time.")
def add_alignment(model_path, output):
    """
    Expose a model's per-phoneme duration tensor as a graph output, writing
    the patched copy alongside the original.

    Voices synthesize with alignments (`include_alignments=True`) only if
    their model exposes durations. Running this once, offline, means every
    later load finds the patched model instead of paying for the surgery on
    the first request.
    """
    from phoonnx.alignment import export_alignment_model

    try:
        dest = export_alignment_model(model_path, output)
    except ImportError:
        click.echo("Error: the 'onnx' package is required for graph surgery; "
                   "install it with `pip install phoonnx[streaming]`.", err=True)
        raise SystemExit(1)
    except (ValueError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    click.echo(f"Wrote alignment-capable model to: {dest}")


if __name__ == "__main__":
    cli()
