//! Example: synthesize TTS via FluidAudio Kokoro.
//! Usage: cargo run --example kokoro --features tts -- "Hello world" am_michael en-us > out.wav
//!        cargo run --example kokoro --features tts -- "你好" zm_yunjian zh > out.wav
//! `lang` selects the KokoroAne variant (`zh` → Mandarin, else English).
use fluidaudio_rs::FluidAudio;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let text = args.get(1).map(String::as_str).unwrap_or("Hello world");
    let voice = args.get(2).map(String::as_str).unwrap_or("am_michael");
    let lang = args.get(3).map(String::as_str).unwrap_or("en-us");

    let audio = FluidAudio::new()?;
    eprintln!("Initializing Kokoro (downloads model on first run)...");
    audio.init_kokoro(voice, lang)?;
    eprintln!("Kokoro available: {}", audio.is_kokoro_available());

    let wav = audio.synthesize_kokoro(text, voice, 1.0)?;
    eprintln!("WAV bytes: {}", wav.len());
    std::io::stdout().write_all(&wav)?;
    Ok(())
}
