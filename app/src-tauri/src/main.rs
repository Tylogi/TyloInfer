#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

#[cfg(feature = "agent")]
mod agent;
#[cfg(not(feature = "agent"))]
mod studio;

fn main() {
    #[cfg(feature = "agent")]
    agent::run();
    #[cfg(not(feature = "agent"))]
    studio::run();
}
