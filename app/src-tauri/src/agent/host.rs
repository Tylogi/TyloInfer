use std::{
    env, fs, io,
    io::{BufRead, BufReader},
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};
use url::Url;

pub(super) struct Host(Child);

impl Host {
    pub(super) fn start(
        runtime: &Path,
        home: &Path,
        workspace: &Path,
    ) -> io::Result<(Self, Receiver<Url>)> {
        fs::create_dir_all(home)?;
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
            .stderr(Stdio::null());
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
                if let Some(url) = ready_url(&line) {
                    if let Some(sender) = sender.take() {
                        let _ = sender.send(url);
                    }
                }
            }
        });
        Ok((Self(child), ready))
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        drop(self.0.stdin.take());
        let deadline = Instant::now() + Duration::from_secs(6);
        loop {
            match self.0.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(25)),
                _ => break,
            }
        }
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn ready_url(line: &str) -> Option<Url> {
    let url = Url::parse(line.strip_prefix("tyloinfer-agent ready: ")?).ok()?;
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
        .then_some(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_authenticated_local_boot_announcement_is_accepted() {
        assert!(ready_url("tyloinfer-agent ready: http://127.0.0.1:4321/?token=fixture").is_some());
        assert!(ready_url("dsh web: http://127.0.0.1:4321/?token=fixture").is_none());
        for url in [
            "http://127.0.0.2:4321/?token=fixture",
            "http://127.0.0.1:4321/",
            "http://127.0.0.1:4321/?token=",
            "http://127.0.0.1:4321/?token=fixture&token=other",
            "http://user@127.0.0.1:4321/?token=fixture",
            "http://127.0.0.1:0/?token=fixture",
        ] {
            assert!(ready_url(&format!("tyloinfer-agent ready: {url}")).is_none());
        }
    }

    #[cfg(unix)]
    #[test]
    fn host_hands_off_loopback_and_closes_its_child_on_drop() {
        use std::os::unix::fs::PermissionsExt;
        let runtime = env::temp_dir().join(format!("agent runtime {}", uuid::Uuid::new_v4()));
        fs::create_dir(&runtime).unwrap();
        let node = runtime.join("node");
        fs::write(&node, b"#!/bin/sh\nprintf 'tyloinfer-agent ready: http://127.0.0.1:4321/?token=fixture\\n'\nread -r unused\nprintf 'closed' > \"$DSH_HOME/exited\"\n").unwrap();
        fs::set_permissions(node, fs::Permissions::from_mode(0o755)).unwrap();
        let home = runtime.join("data");
        let (host, ready) = Host::start(&runtime, &home, &runtime).unwrap();
        assert_eq!(
            ready.recv_timeout(Duration::from_secs(2)).unwrap().port(),
            Some(4321)
        );
        drop(host);
        assert_eq!(fs::read(home.join("exited")).unwrap(), b"closed");
        fs::remove_dir_all(runtime).unwrap();
    }
}
