use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use tauri::{AppHandle, Manager};

const LOG_FILE: &str = "tyloinfer-server.log";
const LOG_ROTATE_BYTES: u64 = 16 * 1024 * 1024;

pub(super) fn check_local_ports(data_dir: &Path, openai_port: u16) -> Result<(), String> {
    let anthropic_port = fs::read(data_dir.join("listener.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| value.get("anthropic_port").and_then(|port| port.as_u64()))
        .and_then(|port| u16::try_from(port).ok())
        .filter(|port| *port > 0)
        .unwrap_or(8091);
    if openai_port == anthropic_port {
        return Err(format!(
            "OpenAI and Anthropic cannot share port {openai_port}. Choose different API ports."
        ));
    }
    for (protocol, port) in [("OpenAI", openai_port), ("Anthropic", anthropic_port)] {
        std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::AddrInUse => format!("The {protocol} API port {port} is already in use. Close the other service or choose a different port."),
                std::io::ErrorKind::PermissionDenied => format!("Permission denied for the {protocol} API port {port}. Choose a different port."),
                _ => format!("Could not use the {protocol} API port {port}: {error}"),
            })?;
    }
    Ok(())
}

pub(super) fn startup_failure(log_path: &Path, offset: u64, summary: &str) -> String {
    let detail = (|| -> std::io::Result<String> {
        let mut log = File::open(log_path)?;
        let end = log.metadata()?.len();
        log.seek(SeekFrom::Start(offset.max(end.saturating_sub(8192))))?;
        let mut bytes = Vec::new();
        log.take(8192).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).trim().to_string())
    })()
    .unwrap_or_default();
    if detail.is_empty() {
        format!("{summary}\nServer log: {}", log_path.display())
    } else {
        format!(
            "{summary}\n\n{detail}\n\nServer log: {}",
            log_path.display()
        )
    }
}

fn open_log(path: &Path) -> Result<File, String> {
    if path
        .metadata()
        .map(|metadata| metadata.len() >= LOG_ROTATE_BYTES)
        .unwrap_or(false)
    {
        let backup = path.with_extension("log.1");
        if backup.exists() {
            fs::remove_file(&backup).map_err(|error| error.to_string())?;
        }
        fs::rename(path, backup).map_err(|error| error.to_string())?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())
}

pub(super) fn configure_detached(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
}

pub(super) fn usable_program(path: &Path) -> bool {
    let Ok(candidate) = path.canonicalize() else {
        return false;
    };
    let Ok(current) = std::env::current_exe().and_then(fs::canonicalize) else {
        return false;
    };
    candidate.is_file() && candidate != current
}

pub(super) fn packaged_executable(app: &AppHandle, names: &[&str]) -> Option<PathBuf> {
    let mut directories = Vec::new();
    if let Ok(current) = std::env::current_exe() {
        if let Some(parent) = current.parent() {
            directories.push(parent.to_path_buf());
        }
    }
    if let Ok(resource) = app.path().resource_dir() {
        directories.push(resource);
    }
    for directory in directories {
        for name in names {
            let candidate = directory.join(name);
            if usable_program(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

fn packaged_runtime(app: &AppHandle) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    let names = ["mfq-decode-metal"];
    #[cfg(target_os = "windows")]
    let names = ["mfq-runtime.exe"];
    #[cfg(all(unix, not(target_os = "macos")))]
    let names = ["mfq-runtime"];
    packaged_executable(app, &names)
}

fn packaged_resource(app: &AppHandle, name: &str) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir().ok()?;
    let candidate = resource_dir.join(name);
    candidate.is_file().then_some(candidate)
}

#[cfg(target_os = "macos")]
fn packaged_framework(name: &str) -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    let contents = executable.parent()?.parent()?;
    let candidate = contents.join("Frameworks").join(name);
    candidate.is_file().then_some(candidate)
}

pub(super) fn packaged_server(app: &AppHandle) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "tyloinfer-server.exe"
    } else {
        "tyloinfer-server"
    };
    let program = app
        .path()
        .resource_dir()
        .ok()?
        .join("tyloinfer-server")
        .join(name);
    usable_program(&program).then_some(program)
}

pub(super) fn spawn_core(
    app: &AppHandle,
    mut command: Command,
    data_dir: &Path,
    port: u16,
    instance_id: &str,
) -> Result<(Child, PathBuf, u64), String> {
    let log_path = data_dir.join(LOG_FILE);
    let stdout = open_log(&log_path)?;
    let log_offset = stdout.metadata().map_err(|error| error.to_string())?.len();
    let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
    command
        .arg("--exit-on-stdin-close")
        .arg("--studio-instance-id")
        .arg(instance_id)
        .arg("--no-access-log")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--work-dir")
        .arg(data_dir)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string());
    if let Some(runtime) = packaged_runtime(app) {
        command.arg("--running-executable").arg(runtime);
    }
    #[cfg(windows)]
    if let Ok(resource_dir) = app.path().resource_dir() {
        // The CUDA and MSVC DLLs are bundled as resources rather
        // than alongside the desktop executable.  Put that directory
        // first in the worker's inherited DLL search path.
        let mut paths = vec![resource_dir];
        if let Some(existing) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(path) = std::env::join_paths(paths) {
            command.env("PATH", path);
        }
    }
    if let Some(metallib) = packaged_resource(app, "mlx.metallib") {
        command.env("TYLOINFER_MLX_METALLIB", metallib);
    }
    #[cfg(target_os = "macos")]
    if let Some(video_library) = packaged_framework("libmfq_avfoundation_video.dylib") {
        command.env("TYLOINFER_AVFOUNDATION_VIDEO_LIBRARY", video_library);
    }
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .current_dir(data_dir);
    configure_detached(&mut command);
    let child = command
        .spawn()
        .map_err(|error| format!("failed to start TyloInfer Server: {error}"))?;
    Ok((child, log_path, log_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn startup_test_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!("tyloinfer-core-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn startup_failure_reports_only_this_launch_and_bounds_log_output() {
        let directory = startup_test_dir();
        let path = directory.join(LOG_FILE);
        let previous = "previous-launch-private-text\n";
        fs::write(&path, format!("{previous}new launch: library not found\n")).unwrap();
        let error = startup_failure(&path, previous.len() as u64, "Server exited.");
        assert!(error.contains("Server exited."));
        assert!(error.contains("library not found"));
        assert!(!error.contains("previous-launch-private-text"));
        fs::write(&path, format!("{}\nlast error", "x".repeat(20000))).unwrap();
        let error = startup_failure(&path, 0, "Server exited.");
        assert!(error.contains("last error"));
        assert!(error.len() < 8500);
        assert!(startup_failure(&path, 30000, "Server exited.").contains("Server log:"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_failure_keeps_the_summary_when_the_log_is_missing() {
        let directory = startup_test_dir();
        let error = startup_failure(&directory.join(LOG_FILE), 0, "Server exited.");
        assert!(error.starts_with("Server exited.\nServer log:"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_failure_after_log_rotation_does_not_expose_the_previous_log() {
        use std::io::Write;
        let directory = startup_test_dir();
        let path = directory.join(LOG_FILE);
        fs::write(&path, "previous-launch-private-text").unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(LOG_ROTATE_BYTES)
            .unwrap();
        let mut log = open_log(&path).unwrap();
        let offset = log.metadata().unwrap().len();
        assert_eq!(offset, 0);
        writeln!(log, "current launch error").unwrap();
        drop(log);
        let error = startup_failure(&path, offset, "Server exited.");
        assert!(error.contains("current launch error"));
        assert!(!error.contains("previous-launch-private-text"));
        assert_eq!(
            path.with_extension("log.1").metadata().unwrap().len(),
            LOG_ROTATE_BYTES
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_rejects_shared_api_ports_without_spawning_a_server() {
        let directory = startup_test_dir();
        fs::write(
            directory.join("listener.json"),
            r#"{"anthropic_port":19290}"#,
        )
        .unwrap();
        let error = check_local_ports(&directory, 19290).unwrap_err();
        assert!(error.contains("cannot share port 19290"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn startup_identifies_the_occupied_api_port() {
        let directory = startup_test_dir();
        let occupied = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let occupied_port = occupied.local_addr().unwrap().port();
        let available = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let available_port = available.local_addr().unwrap().port();
        drop(available);
        fs::write(
            directory.join("listener.json"),
            format!(r#"{{"anthropic_port":{occupied_port}}}"#),
        )
        .unwrap();
        let error = check_local_ports(&directory, available_port).unwrap_err();
        assert!(error.contains(&format!(
            "Anthropic API port {occupied_port} is already in use"
        )));
        fs::write(
            directory.join("listener.json"),
            format!(r#"{{"anthropic_port":{available_port}}}"#),
        )
        .unwrap();
        let error = check_local_ports(&directory, occupied_port).unwrap_err();
        assert!(error.contains(&format!(
            "OpenAI API port {occupied_port} is already in use"
        )));
        fs::remove_dir_all(directory).unwrap();
    }
}
