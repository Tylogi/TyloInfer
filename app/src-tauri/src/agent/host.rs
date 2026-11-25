use std::{
    env, fs, io,
    io::{BufRead, BufReader},
    num::NonZeroU16,
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};
use url::Url;

pub(super) const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

pub(super) struct Host(Child);

#[derive(serde::Deserialize)]
pub(super) struct Ready {
    pub url: Url,
    pub core_port: Option<NonZeroU16>,
}

impl Host {
    pub(super) fn start(
        runtime: &Path,
        home: &Path,
        workspace: &Path,
    ) -> io::Result<(Self, Receiver<Ready>)> {
        // Tauri's canonicalized Windows resource path uses \\?\. Node 24
        // fails resolving a verbatim script entry (EISDIR on the drive), before
        // boot.mjs even runs. Keep OS-only prefixes out of Node args/env/PATH.
        let runtime = dunce::simplified(runtime);
        let home = dunce::simplified(home);
        let workspace = dunce::simplified(workspace);
        fs::create_dir_all(home)?;
        // Keep DSH's startup diagnostics instead of hiding every loader failure.
        let diagnostics = fs::File::create(home.join("agent-startup.log"))?;
        let node = runtime.join(if cfg!(windows) { "node.exe" } else { "node" });
        let path = env::var_os("PATH").unwrap_or_default();
        let path =
            env::join_paths(std::iter::once(runtime.to_owned()).chain(env::split_paths(&path)))
                .map_err(io::Error::other)?;
        let mut command = Command::new(node);
        command
            .arg(runtime.join("boot.mjs"))
            .arg("launch")
            .current_dir(workspace)
            .env("DSH_HOME", home)
            .env("PATH", path)
            .env_remove("NODE_PATH")
            .env_remove("NODE_OPTIONS")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(diagnostics));
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        let mut child = command.spawn()?;
        let stdout = child.stdout.take().unwrap();
        let (sender, ready) = mpsc::channel();
        thread::spawn(move || {
            let mut sender = Some(sender);
            // Drain stdout for the process lifetime, but never log the authentication URL.
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(launch) = ready_message(&line) {
                    if let Some(sender) = sender.take() {
                        let _ = sender.send(launch);
                    }
                }
            }
        });
        Ok((Self(child), ready))
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        stop_child(&mut self.0);
    }
}

pub(super) fn stop_child(child: &mut Child) {
    drop(child.stdin.take());
    let deadline = Instant::now() + Duration::from_secs(6);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(25)),
            _ => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn ready_message(line: &str) -> Option<Ready> {
    let ready: Ready = serde_json::from_str(line.strip_prefix("tyloinfer-agent ready: ")?).ok()?;
    let url = &ready.url;
    let mut query = url.query_pairs();
    let has_token = matches!(query.next(), Some((key, token)) if key == "token" && !token.is_empty())
        && query.next().is_none();
    (url.scheme() == "http"
        && url.host_str() == Some("127.0.0.1")
        && url.port().is_some_and(|port| port != 0)
        && url.path() == "/"
        && url.username().is_empty()
        && url.password().is_none()
        && url.fragment().is_none()
        && has_token)
        .then_some(ready)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_authenticated_local_boot_announcement_is_accepted() {
        let local = r#"tyloinfer-agent ready: {"url":"http://127.0.0.1:4321/?token=fixture","core_port":8090}"#;
        assert_eq!(ready_message(local).unwrap().core_port.unwrap().get(), 8090);
        assert!(ready_message(&local.replace("8090", "0")).is_none());
        assert!(ready_message(&local.replace("8090", "null"))
            .unwrap()
            .core_port
            .is_none());
        assert!(ready_message("dsh web: http://127.0.0.1:4321/?token=fixture").is_none());
        for url in [
            "http://127.0.0.2:4321/?token=fixture",
            "http://127.0.0.1:4321/",
            "http://127.0.0.1:4321/?token=",
            "http://127.0.0.1:4321/?token=fixture&token=other",
            "http://user@127.0.0.1:4321/?token=fixture",
            "http://127.0.0.1:0/?token=fixture",
        ] {
            assert!(
                ready_message(&format!(r#"tyloinfer-agent ready: {{"url":"{url}"}}"#)).is_none()
            );
        }
    }

    #[test]
    #[ignore = "requires a staged Agent runtime"]
    fn staged_runtime_starts_through_the_desktop_host() {
        let runtime = env::var_os("TYLOINFER_AGENT_TEST_RUNTIME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("agent/runtime"));
        // Tauri resource_dir() returns a canonicalized verbatim path on Windows.
        let runtime = fs::canonicalize(runtime).unwrap();
        let home = env::temp_dir().join(format!("Agent host 测试 {}", uuid::Uuid::new_v4()));
        let workspace = env::var_os("USERPROFILE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| home.clone());
        let started = Instant::now();
        let (host, ready) = Host::start(&runtime, &home, &workspace).unwrap();
        let result = ready.recv_timeout(STARTUP_TIMEOUT);
        drop(host);
        let diagnostics = fs::read_to_string(home.join("agent-startup.log")).unwrap_or_default();
        fs::remove_dir_all(&home).unwrap();
        let ready = result.unwrap_or_else(|error| {
            panic!(
                "DSH failed after {:?}: {error}\n{diagnostics}",
                started.elapsed()
            )
        });
        assert_eq!(ready.core_port.map(NonZeroU16::get), Some(8090));
        eprintln!("DSH desktop handoff in {:?}", started.elapsed());
    }

    #[cfg(unix)]
    #[test]
    fn host_hands_off_loopback_and_closes_its_child_on_drop() {
        use std::os::unix::fs::PermissionsExt;
        let runtime = env::temp_dir().join(format!("agent runtime {}", uuid::Uuid::new_v4()));
        fs::create_dir(&runtime).unwrap();
        let node = runtime.join("node");
        fs::write(&node, b"#!/bin/sh\nprintf 'tyloinfer-agent ready: {\"url\":\"http://127.0.0.1:4321/?token=fixture\"}\\n'\nread -r unused\nprintf 'closed' > \"$DSH_HOME/exited\"\n").unwrap();
        fs::set_permissions(node, fs::Permissions::from_mode(0o755)).unwrap();
        let home = runtime.join("data");
        let (host, ready) = Host::start(&runtime, &home, &runtime).unwrap();
        assert_eq!(
            ready
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .url
                .port(),
            Some(4321)
        );
        drop(host);
        assert_eq!(fs::read(home.join("exited")).unwrap(), b"closed");
        fs::remove_dir_all(runtime).unwrap();
    }
}
