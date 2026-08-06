import CoreML
import FluidAudio
import Foundation

/// Emit a diagnostic to stderr. Diarization callers marshal their own results, but
/// some stream unrelated bytes on stdout, so diagnostics must not go there.
private func diarizeLog(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// MARK: - Cancellation

/// Cancellation handle shared between the thread blocked in a diarize call and
/// whichever thread decides to stop it.
///
/// Cancellation is *sticky*: a `cancel()` that lands before the diarize `Task` is
/// adopted still cancels it, so a caller racing a watchdog against a fast start
/// cannot lose the request.
final class DiarizeCancelToken {
    private let lock = NSLock()
    private var cancelled = false
    private var task: Task<Void, Never>?

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let running = task
        lock.unlock()
        running?.cancel()
    }

    func adopt(_ task: Task<Void, Never>) {
        lock.lock()
        self.task = task
        let alreadyCancelled = cancelled
        lock.unlock()
        if alreadyCancelled { task.cancel() }
    }

    /// Drop the finished task so a token reused for a later call cannot cancel it.
    func release() {
        lock.lock()
        task = nil
        lock.unlock()
    }
}

@_cdecl("fluidaudio_diarize_cancel_token_new")
public func fluidaudio_diarize_cancel_token_new() -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(DiarizeCancelToken()).toOpaque()
}

/// Request cancellation. Safe to call from any thread, more than once, and after the
/// diarize call it targets has already returned.
@_cdecl("fluidaudio_diarize_cancel")
public func fluidaudio_diarize_cancel(_ token: UnsafeMutableRawPointer?) {
    guard let token = token else { return }
    Unmanaged<DiarizeCancelToken>.fromOpaque(token).takeUnretainedValue().cancel()
}

@_cdecl("fluidaudio_diarize_cancel_token_free")
public func fluidaudio_diarize_cancel_token_free(_ token: UnsafeMutableRawPointer?) {
    guard let token = token else { return }
    Unmanaged<DiarizeCancelToken>.fromOpaque(token).release()
}

// MARK: - Compute units

/// Kebab-case spellings accepted for the diarization compute-unit selector.
/// `all` is the default and the only one that uses the Neural Engine for the whole
/// model; the rest exist for hosts with no usable ANE.
private let diarizeComputeUnitPresets: [(name: String, units: MLComputeUnits)] = [
    ("all", .all),
    ("cpu-and-ane", .cpuAndNeuralEngine),
    ("cpu-and-gpu", .cpuAndGPU),
    ("cpu-only", .cpuOnly),
]

/// Parse `raw` into `MLComputeUnits`. NULL/empty keeps `.all`. An unrecognised value
/// is a caller bug: fail rather than silently diarizing on units nobody asked for.
private func parseDiarizeComputeUnits(_ raw: UnsafePointer<CChar>?) -> MLComputeUnits? {
    let value = (raw.map { String(cString: $0) } ?? "").trimmingCharacters(in: .whitespaces)
    if value.isEmpty { return .all }
    if let match = diarizeComputeUnitPresets.first(where: { $0.name == value.lowercased() }) {
        return match.units
    }
    diarizeLog(
        "Diarize error: unknown compute-units preset '\(value)' "
            + "(expected one of: \(diarizeComputeUnitPresets.map(\.name).joined(separator: ", ")))")
    return nil
}

// MARK: - Model-path diarization C FFI
//
// Diarize from a pre-staged Sortformer .mlpackage (no HuggingFace download), for
// callers that pin/verify the model themselves. Output marshalling mirrors
// fluidaudio_diarize_file; the caller frees the arrays with
// fluidaudio_free_diarization_result.

@_cdecl("fluidaudio_diarize_file_with_models")
public func fluidaudio_diarize_file_with_models(
    _ ptr: UnsafeMutableRawPointer?,
    _ audioPath: UnsafePointer<CChar>?,
    _ modelPath: UnsafePointer<CChar>?,
    _ outSpeakerIds: UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?>?,
    _ outStartTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outEndTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outQualityScores: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let audioPath = audioPath, let modelPath = modelPath else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    let audioString = String(cString: audioPath)
    let modelString = String(cString: modelPath)

    do {
        let segments = try bridge.diarizeFileWithModels(audioPath: audioString, modelPath: modelString)
        emitDiarizationSegments(
            segments,
            outSpeakerIds: outSpeakerIds,
            outStartTimes: outStartTimes,
            outEndTimes: outEndTimes,
            outQualityScores: outQualityScores,
            outCount: outCount
        )
        return 0
    } catch {
        print("Diarize (model path) error: \(error)")
        return -1
    }
}

/// As above, but with progress reporting, cancellation and compute-unit selection.
///
/// A separate C symbol so the original's arity stays untouched. Both callbacks fire on
/// the diarizer's own thread while the caller's is parked in this synchronous call, so
/// they must be safe to run from another thread, and both receive `callbackContext`.
/// `modelReady` fires exactly once, when the `MLModel` is loaded and the diarizer
/// initialised but before a single audio sample has been read — everything after it is
/// audio work, which is what lets a caller bound the load and the processing separately.
/// `progress` fires once per processed chunk with
/// `(context, processedSamples, totalSamples, chunksProcessed)`.
/// Returns 0 on success, -2 if cancelled, -1 otherwise.
@_cdecl("fluidaudio_diarize_file_with_models_controlled")
public func fluidaudio_diarize_file_with_models_controlled(
    _ ptr: UnsafeMutableRawPointer?,
    _ audioPath: UnsafePointer<CChar>?,
    _ modelPath: UnsafePointer<CChar>?,
    _ computeUnits: UnsafePointer<CChar>?,
    _ cancelToken: UnsafeMutableRawPointer?,
    _ modelReady: (@convention(c) (UnsafeMutableRawPointer?) -> Void)?,
    _ progress: (@convention(c) (UnsafeMutableRawPointer?, UInt64, UInt64, UInt32) -> Void)?,
    _ callbackContext: UnsafeMutableRawPointer?,
    _ outSpeakerIds: UnsafeMutablePointer<UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?>?,
    _ outStartTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outEndTimes: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outQualityScores: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outCount: UnsafeMutablePointer<UInt32>?
) -> Int32 {
    guard let ptr = ptr, let audioPath = audioPath, let modelPath = modelPath else { return -1 }
    guard let units = parseDiarizeComputeUnits(computeUnits) else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()
    let token = cancelToken.map {
        Unmanaged<DiarizeCancelToken>.fromOpaque($0).takeUnretainedValue()
    }

    let callback: SortformerDiarizer.ProgressCallback? = progress.map { emit in
        { processed, total, chunks in
            emit(callbackContext, UInt64(max(0, processed)), UInt64(max(0, total)), UInt32(max(0, chunks)))
        }
    }
    let ready: (() -> Void)? = modelReady.map { emit in { emit(callbackContext) } }

    do {
        let segments = try bridge.diarizeFileWithModels(
            audioPath: String(cString: audioPath),
            modelPath: String(cString: modelPath),
            computeUnits: units,
            cancelToken: token,
            onModelReady: ready,
            progress: callback
        )
        emitDiarizationSegments(
            segments,
            outSpeakerIds: outSpeakerIds,
            outStartTimes: outStartTimes,
            outEndTimes: outEndTimes,
            outQualityScores: outQualityScores,
            outCount: outCount
        )
        return 0
    } catch is CancellationError {
        return -2
    } catch {
        diarizeLog("Diarize (model path) error: \(error)")
        return -1
    }
}

// MARK: - Warm-up: pre-compile the diarization model
//
// Compile the pre-staged Sortformer `.mlpackage` into the writable per-user cache and
// load it once, paying the one-time ~100s ANE compile up front (populates the e5rt
// cache). Callers warm this at install time so the first real diarize is fast.

@_cdecl("fluidaudio_compile_diarization_model")
public func fluidaudio_compile_diarization_model(
    _ ptr: UnsafeMutableRawPointer?,
    _ modelPath: UnsafePointer<CChar>?
) -> Int32 {
    guard let ptr = ptr, let modelPath = modelPath else { return -1 }
    let bridge = Unmanaged<FluidAudioBridgeInternal>.fromOpaque(ptr).takeUnretainedValue()

    do {
        try bridge.compileDiarizationModel(modelPath: String(cString: modelPath))
        return 0
    } catch {
        print("Compile diarization model error: \(error)")
        return -1
    }
}
