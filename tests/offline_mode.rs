//! `set_offline_mode` behaviour, in its own test binary because the flag is
//! process-global: a test here must not leak into the download-dependent tests
//! in `ffi_bindings.rs`. Within this binary a mutex keeps the two sequential.
//!
//! Neither test downloads anything — that is the point of the second one.

use std::sync::Mutex;

use fluidaudio_rs::{
    model_registry_base_url, offline_mode, set_offline_mode, FluidAudio, FluidAudioError,
};

static SERIAL: Mutex<()> = Mutex::new(());

#[test]
fn the_flag_round_trips() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    assert!(!offline_mode(), "offline mode must default to off");
    set_offline_mode(true);
    assert!(offline_mode());
    set_offline_mode(false);
    assert!(!offline_mode());
}

/// The half of offline mode upstream's flag does not provide: `AssetDownloader`
/// consults no flag, so the only thing that stops `ensureVoicePack` and the
/// Mandarin aux probes is the registry they resolve against. Asserting the base
/// moves off the network — and comes back — is how that is observable at all.
#[test]
fn offline_mode_moves_the_registry_off_the_network_and_back() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let online = model_registry_base_url();
    assert!(
        online.starts_with("http"),
        "expected a fetchable registry, got {online:?}"
    );

    set_offline_mode(true);
    let offline = model_registry_base_url();
    set_offline_mode(false);

    assert!(
        !offline.starts_with("http"),
        "offline registry must not be fetchable, got {offline:?}"
    );
    assert_eq!(
        model_registry_base_url(),
        online,
        "turning offline mode off must restore the caller's registry"
    );
}

/// Repeat calls must not lose the original registry — the naive shape captures
/// the previous base on every enable, so a second `true` would capture the
/// offline base and restore it forever.
#[test]
fn enabling_twice_still_restores_the_original_registry() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let online = model_registry_base_url();
    set_offline_mode(true);
    set_offline_mode(true);
    set_offline_mode(false);
    assert_eq!(model_registry_base_url(), online);
}

/// The whole chain under one assertion: with the flag on and a models root that
/// holds nothing, `KokoroAneManager.initialize` cannot fetch the ANE bundle, so
/// the caller gets a variant it can act on instead of an opaque bridge string.
#[test]
fn initialize_without_staged_assets_reports_assets_unavailable() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let root = std::env::temp_dir().join("fluidaudio-rs-offline-empty-root");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create empty models root");

    set_offline_mode(true);
    let audio = FluidAudio::with_models_dir(&root).expect("bridge creation");
    let err = audio
        .init_kokoro("af_heart", "en-us")
        .expect_err("init must fail with nothing staged and downloads disabled");
    set_offline_mode(false);

    let _ = std::fs::remove_dir_all(&root);
    assert!(
        matches!(err, FluidAudioError::AssetsUnavailable(_)),
        "unexpected error variant: {err:?}"
    );
}
