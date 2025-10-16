import click
from phoonnx.model_manager import TTSModelManager, TTSModelInfo
import requests


# --- Helper Function for CLI Output ---
def print_voice_info(voice: TTSModelInfo):
    """Prints detailed info about a single voice."""
    click.echo(f"  ID:          {voice.voice_id}")
    click.echo(f"  Language:    {voice.lang}")
    click.echo(f"  Engine:      {voice.engine.value}")
    click.echo(f"  Phoneme Type: {voice.phoneme_type}")
    click.echo(f"  Model URL:   {voice.model_url}")
    click.echo(f"  Config URL:  {voice.config_url}")
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
    Clears the local voice cache, fetches the latest voice lists from upstream
    sources (Piper, Mimic3, OpenVoiceOS), and saves them to the cache.
    """
    manager = TTSModelManager()

    if not no_clear:
        click.echo("Clearing local voice cache...")
        manager.clear()
    else:
        click.echo("Loading existing voice cache...")
        manager.load()

    click.echo("Fetching voice lists from upstream sources (this requires an internet connection)...")

    # Run the fetch methods to populate the cache
    try:
        manager.get_ovos_voice_list()
        click.echo("-> Fetched OpenVoiceOS voices.")
        manager.get_proxectonos_voice_list()
        click.echo("-> Fetched Proxectonos voices.")
        manager.get_phonikud_voice_list()
        click.echo("-> Fetched Phonikud voices.")
        manager.get_piper_voice_list()
        click.echo("-> Fetched Piper voices.")
        manager.get_mimic3_voice_list()
        click.echo("-> Fetched Mimic3 voices.")
    except requests.exceptions.RequestException as e:
        click.echo(f"\nError: Could not fetch voice lists due to a network or connection error.", err=True)
        click.echo(f"Details: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"An unexpected error occurred while fetching voice lists: {e}", err=True)
        return

    manager.save()
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
            click.echo(f"* {voice.voice_id} ({voice.lang})")


@cli.command(name="download")
@click.argument("voice_id", type=str)
def download_voice(voice_id):
    """
    Downloads the model, config, and token files for a specific VOICE_ID.
    The VOICE_ID must exist in the local cache (run update-cache first).
    """
    manager = TTSModelManager()
    manager.load()

    if voice_id not in manager.voices:
        click.echo(f"Error: Voice ID '{voice_id}' not found in cache.", err=True)
        click.echo("Hint: Run 'phoonnx_cli.py update-cache' first to fetch the list.", err=True)
        return

    voice_info = manager.voices[voice_id]

    try:
        click.echo(f"Attempting to download files for: {voice_id} ({voice_info.lang})")

        # Download config (model.json)
        click.echo("-> Downloading config...")
        voice_info.download_config()

        # Download tokens/phoneme map
        if voice_info.tokens_url:
            click.echo("-> Downloading tokens/phoneme map...")
            voice_info.download_phoneme_map()

        # Download model (model.onnx)
        click.echo("-> Downloading ONNX model (this may take a while)...")
        voice_info.download_model()

        click.echo(f"\nDownload complete. Files saved to: {voice_info.voice_path}")

    except requests.exceptions.RequestException as e:
        click.echo(f"\nDownload failed due to network error: {e}", err=True)
    except Exception as e:
        click.echo(f"\nAn unexpected error occurred during download: {e}", err=True)


if __name__ == "__main__":
    cli()