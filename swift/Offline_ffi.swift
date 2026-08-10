import FluidAudio
import Foundation

// MARK: - Offline enforcement C FFI
//
// `ModelHub.offlineMode` is a static on upstream's `HFClient`, so this is
// process-global by construction and takes no bridge pointer — unlike every
// other entry point here.

/// Registry base the offline switch installs. Any absolute URL string works;
/// this one is chosen so the scheme cannot resolve — `URLSession` rejects it
/// with `unsupportedURL` before name resolution, so nothing leaves the host and
/// nothing waits on a timeout. It also reads as an explanation in any log line
/// that echoes the URL.
private let offlineRegistryBase = "unavailable-offline://fluidaudio"

/// Registry base to put back when offline mode is turned off. Captured on the
/// way in rather than assumed, so a caller that configured its own mirror (or
/// set `REGISTRY_URL`) gets that back and not HuggingFace.
nonisolated(unsafe) private var registryBaseBeforeOffline: String?

/// Turn FluidAudio's offline-only enforcement on (`enabled != 0`) or off.
///
/// Two mechanisms, because upstream's flag alone does not cover the library:
///
/// 1. `ModelHub.offlineMode`, which makes `ModelHub.download`, `fetchFile`,
///    `fetchWithAuth`, the `HFTreeLister` walk and `loadModels`'
///    retry-with-redownload throw `DownloadError.networkDisabled` /
///    `.modelMissing` instead of reaching HuggingFace.
/// 2. `ModelRegistry.baseURL`, repointed at an unresolvable scheme. At
///    FluidAudio 0.15.5 `AssetDownloader` talks to the shared `URLSession` and
///    consults no flag, so `ensureVoicePack`, `ensureEnglishLexicon`,
///    `ensureMandarinG2P`, `ensureMandarinJiebaHmm` and `ensureMandarinG2pw`
///    would still hit the network with (1) alone. Every one of them builds its
///    URL through `ModelRegistry.resolveModel`, so moving the base is what
///    actually stops them. Each already treats a failed fetch as best-effort
///    (dict-only Mandarin, BART-only English), so they degrade exactly as
///    upstream intends instead of erroring.
///
/// Set it before touching any loader — upstream reads the flag per request, so
/// flipping it mid-download only stops the *next* one.
///
/// This is enforcement, not a substitute for having the assets: an application
/// that needs a model must still stage it, and (2) turns a would-be download
/// into an immediate local failure rather than into the file appearing.
@_cdecl("fluidaudio_set_offline_mode")
public func fluidaudio_set_offline_mode(_ enabled: Int32) {
    let wantOffline = enabled != 0
    guard wantOffline != ModelHub.offlineMode else { return }

    if wantOffline {
        registryBaseBeforeOffline = ModelRegistry.baseURL
        ModelRegistry.baseURL = offlineRegistryBase
    } else if let previous = registryBaseBeforeOffline {
        ModelRegistry.baseURL = previous
        registryBaseBeforeOffline = nil
    }
    ModelHub.offlineMode = wantOffline
}

/// Current state of `ModelHub.offlineMode`.
@_cdecl("fluidaudio_offline_mode")
public func fluidaudio_offline_mode() -> Int32 {
    ModelHub.offlineMode ? 1 : 0
}

/// The registry every download URL is built from. Ownership transfers to the
/// caller, which must free it with `fluidaudio_free_string`.
///
/// Exposed so a caller can verify which host it is pointed at — including that
/// offline mode moved it off the network, which is otherwise only observable by
/// watching for packets that never come.
@_cdecl("fluidaudio_model_registry_base_url")
public func fluidaudio_model_registry_base_url() -> UnsafeMutablePointer<CChar>? {
    strdup(ModelRegistry.baseURL)
}
