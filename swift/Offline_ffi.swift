import FluidAudio
import Foundation

// MARK: - Offline enforcement C FFI
//
// `ModelHub.offlineMode` is a static on upstream's `HFClient`, so this is
// process-global by construction and takes no bridge pointer — unlike every
// other entry point here.

/// Turn FluidAudio's offline-only enforcement on (`enabled != 0`) or off.
///
/// With it on, `ModelHub.download`, `fetchFile`, `fetchWithAuth`, the
/// `HFTreeLister` walk and `loadModels`' retry-with-redownload fallback throw
/// `DownloadError.networkDisabled` / `.modelMissing` instead of reaching
/// HuggingFace. Set it before touching any loader — upstream reads the flag
/// per request, so flipping it mid-download only stops the *next* one.
///
/// **What it does not cover, at FluidAudio 0.15.5:** `AssetDownloader` talks to
/// `ModelHub.session` directly and consults no flag, so the KokoroAne helpers
/// built on it still reach the network with offline mode on —
/// `ensureVoicePack`, `ensureEnglishLexicon`, `ensureMandarinG2P`,
/// `ensureMandarinJiebaHmm` and `ensureMandarinG2pw`. Callers that need those
/// assets never fetched must pre-stage them; the flag alone is not a
/// network kill-switch for the Kokoro path.
@_cdecl("fluidaudio_set_offline_mode")
public func fluidaudio_set_offline_mode(_ enabled: Int32) {
    ModelHub.offlineMode = enabled != 0
}

/// Current state of `ModelHub.offlineMode`.
@_cdecl("fluidaudio_offline_mode")
public func fluidaudio_offline_mode() -> Int32 {
    ModelHub.offlineMode ? 1 : 0
}
