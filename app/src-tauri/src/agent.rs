mod host;

use host::Host;
use std::sync::{Arc, Mutex, OnceLock};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindow, WebviewWindowBuilder};
use url::Origin;

type AgentState = Mutex<Option<Host>>;

pub(super) fn run() {
    tauri::Builder::default()
        .manage(AgentState::default())
        .setup(|app| {
            if app.config().identifier != "com.tylogi.tyloinfer.agent" {
                return Err("Build Agent with tauri.agent.conf.json".into());
            }
            let origin = Arc::new(OnceLock::<Origin>::new());
            let window_origin = origin.clone();
            let window =
                WebviewWindowBuilder::new(app, "agent", WebviewUrl::App("index.html".into()))
                    .title("TyloInfer Agent")
                    .inner_size(1280.0, 820.0)
                    .on_navigation(move |url| match window_origin.get() {
                        Some(origin) => url.origin() == *origin,
                        None => matches!(
                            (url.scheme(), url.host_str()),
                            ("tauri", Some("localhost")) | ("http", Some("tauri.localhost"))
                        ),
                    })
                    .build()?;
            let runtime = app.path().resource_dir()?.join("agent-runtime");
            let home = app.path().app_data_dir()?.join("dsh");
            let workspace = app.path().home_dir()?;
            let (host, ready) = match Host::start(&runtime, &home, &workspace) {
                Ok(launch) => launch,
                Err(error) => {
                    show_error(
                        &window,
                        &format!("Could not start the bundled DSH: {error}"),
                    );
                    return Ok(());
                }
            };
            *app.state::<AgentState>().lock().unwrap() = Some(host);
            std::thread::spawn(move || {
                let failure = match ready.recv_timeout(host::STARTUP_TIMEOUT) {
                    Ok(url) => {
                        let _ = origin.set(url.origin());
                        if window.navigate(url).is_ok() {
                            return;
                        }
                        "Could not open the local DSH window."
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        "The bundled DSH did not become ready within 30 seconds."
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        "The bundled DSH exited before becoming ready."
                    }
                };
                stop_host(window.app_handle());
                show_error(
                    &window,
                    &format!(
                        "{failure} Startup log: {}",
                        home.join("agent-startup.log").display()
                    ),
                );
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("TyloInfer Agent failed")
        .run(|app, event| {
            if matches!(event, RunEvent::Exit) {
                stop_host(app);
            }
        });
}

fn stop_host(app: &tauri::AppHandle) {
    let host = app.state::<AgentState>().lock().unwrap().take();
    drop(host);
}

fn show_error(window: &WebviewWindow, message: &str) {
    let message = serde_json::to_string(message).unwrap();
    let _ = window.eval(format!(
        "{{ const report = () => {{ document.getElementById('status').textContent = {message}; }}; \
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', report, {{once: true}}); else report(); }}"
    ));
}
