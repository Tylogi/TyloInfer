mod host;

use crate::core_process::{check_local_ports, packaged_server, spawn_core, startup_failure};
use host::Host;
use std::fs;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindow, WebviewWindowBuilder};
use url::Origin;

type AgentState = Mutex<Option<AgentProcesses>>;

struct AgentProcesses {
    _host: Host,
    _core: Option<CoreChild>,
}

struct CoreChild(Child);

impl Drop for CoreChild {
    fn drop(&mut self) {
        host::stop_child(&mut self.0);
    }
}

pub(super) fn run() {
    tauri::Builder::default()
        .manage(AgentState::default())
        .setup(|app| {
            if app.config().identifier != "com.tylogi.tyloinfer.agent" {
                return Err("Build Agent with tauri.agent.conf.json".into());
            }
            let origin = Arc::new(OnceLock::<Origin>::new());
            let window_origin = origin.clone();
            // The DSH login redirect needs its SameSite=Strict cookie, which WebKit omits after a cross-site navigation.
            let window = WebviewWindowBuilder::new(
                app,
                "agent",
                WebviewUrl::CustomProtocol("about:blank".parse().unwrap()),
            )
            .title("TyloInfer Agent")
            .inner_size(1280.0, 820.0)
            .on_navigation(move |url| match window_origin.get() {
                Some(origin) => url.origin() == *origin,
                None => url.as_str() == "about:blank",
            })
            .build()?;
            let runtime = app.path().resource_dir()?.join("agent-runtime");
            let home = app.path().app_data_dir()?.join("dsh");
            let workspace = app.path().home_dir()?;
            let (host, ready) = match Host::start(&runtime, &home, &workspace) {
                Ok(launch) => launch,
                Err(error) => {
                    show_error(&window, format!("Could not start the bundled DSH: {error}"));
                    return Ok(());
                }
            };
            *app.state::<AgentState>().lock().unwrap() = Some(AgentProcesses {
                _host: host,
                _core: None,
            });
            std::thread::spawn(move || {
                let failure = match ready.recv_timeout(host::STARTUP_TIMEOUT) {
                    Ok(launch) => {
                        if let Some(port) = launch.core_port {
                            match tauri::async_runtime::block_on(start_local_core(
                                window.app_handle(),
                                port.get(),
                            )) {
                                Ok(core) => {
                                    if let Some(processes) = window
                                        .app_handle()
                                        .state::<AgentState>()
                                        .lock()
                                        .unwrap()
                                        .as_mut()
                                    {
                                        processes._core = core;
                                    }
                                }
                                Err(error) => show_error(&window, error),
                            }
                        }
                        let _ = origin.set(launch.url.origin());
                        if window.navigate(launch.url).is_ok() {
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
                stop_processes(window.app_handle());
                show_error(
                    &window,
                    format!(
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
                stop_processes(app);
            }
        });
}

fn stop_processes(app: &tauri::AppHandle) {
    let processes = app.state::<AgentState>().lock().unwrap().take();
    drop(processes);
}

async fn start_local_core(app: &tauri::AppHandle, port: u16) -> Result<Option<CoreChild>, String> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_millis(500))
        .build()
        .map_err(|error| error.to_string())?;
    let health = format!("http://127.0.0.1:{port}/health");
    if core_ready(&client, &health).await {
        return Ok(None);
    }

    let data_dir = app
        .path()
        .local_data_dir()
        .map_err(|error| error.to_string())?
        .join("com.tylogi.tyloinfer");
    fs::create_dir_all(&data_dir).map_err(|error| error.to_string())?;
    check_local_ports(&data_dir, port)?;
    let program = packaged_server(app).ok_or("The Agent package is missing TyloInfer Server.")?;
    let mut command = Command::new(program);
    command.arg("--no-web-ui");
    let instance_id = uuid::Uuid::new_v4().to_string();
    let (child, log_path, log_offset) = spawn_core(app, command, &data_dir, port, &instance_id)?;
    let mut child = CoreChild(child);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let status = child.0.try_wait().map_err(|error| error.to_string())?;
        if core_ready(&client, &health).await {
            return Ok(status.is_none().then_some(child));
        }
        if let Some(status) = status {
            return Err(startup_failure(
                &log_path,
                log_offset,
                &format!("TyloInfer Server could not start ({status})."),
            ));
        }
        if Instant::now() >= deadline {
            return Err(startup_failure(
                &log_path,
                log_offset,
                "TyloInfer Server did not become ready within 30 seconds.",
            ));
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn core_ready(client: &reqwest::Client, health: &str) -> bool {
    let Ok(response) = client
        .get(health)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
    else {
        return false;
    };
    response
        .json::<serde_json::Value>()
        .await
        .is_ok_and(|health| {
            health["service"] == "tyloinfer-server" && health["protocol_version"] == "1.0"
        })
}

fn show_error(window: &WebviewWindow, message: String) {
    let window = window.clone();
    tauri::async_runtime::spawn(async move {
        rfd::AsyncMessageDialog::new()
            .set_title("TyloInfer Agent")
            .set_description(message)
            .set_parent(&window)
            .set_level(rfd::MessageLevel::Error)
            .show()
            .await;
    });
}
