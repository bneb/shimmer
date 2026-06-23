//! Daemon Module
//!
//! Handles background execution via a Unix domain socket, allowing the
//! agent swarm to process generation requests as a system service.

use crate::agent::{Agent, AgentConfig};
use llama_cpp_2::llama_backend::LlamaBackend;
use std::io::{BufRead, BufReader};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::sync::mpsc;

/// Resolves the Unix domain socket path, checking:
/// 1. `SHIMMER_SOCKET` environment variable
/// 2. `$XDG_RUNTIME_DIR/shimmer.sock`
/// 3. `/tmp/shimmer.sock` as fallback
fn socket_path() -> String {
    if let Ok(path) = std::env::var("SHIMMER_SOCKET") {
        return path;
    }
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        return format!("{}/shimmer.sock", runtime_dir);
    }
    "/tmp/shimmer.sock".to_string()
}

/// A request payload received over the Unix socket.
pub struct DaemonRequest {
    pub prompt: String,
    pub stream: UnixStream,
}

pub fn start_daemon(
    agent: Arc<Agent>,
    backend: Arc<LlamaBackend>,
    config: Arc<AgentConfig>,
) -> anyhow::Result<()> {
    let path = socket_path();
    tracing::info!("Starting Shimmer UDS Daemon on {}", path);
    if std::path::Path::new(&path).exists() {
        std::fs::remove_file(&path)?;
    }

    let listener = UnixListener::bind(&path)?;
    let (tx, rx) = mpsc::channel::<DaemonRequest>();

    let a = agent.clone();
    let b = backend.clone();
    let c = config.clone();
    std::thread::spawn(move || {
        if let Err(e) = a.run_daemon_swarm(&b, &c, rx) {
            tracing::error!("Daemon thread error: {}", e);
        }
    });

    for stream in listener.incoming() {
        spawn_client_handler(stream, tx.clone());
    }
    Ok(())
}

fn spawn_client_handler(stream: std::io::Result<UnixStream>, tx: mpsc::Sender<DaemonRequest>) {
    match stream {
        Ok(stream) => {
            std::thread::spawn(move || {
                let _ = handle_client(stream, tx);
            });
        }
        Err(e) => tracing::error!("Connection failed: {}", e),
    }
}

fn handle_client(stream: UnixStream, tx: mpsc::Sender<DaemonRequest>) -> anyhow::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(());
    }

    let req: serde_json::Value = serde_json::from_str(&line)?;
    let prompt = req["prompt"].as_str().unwrap_or("").to_string();

    tx.send(DaemonRequest { prompt, stream })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_handle_client_valid_json() {
        let (tx, rx) = mpsc::channel();
        let (mut s1, s2) = std::os::unix::net::UnixStream::pair().unwrap();

        std::thread::spawn(move || {
            let _ = handle_client(s2, tx);
        });

        writeln!(s1, "{{\"prompt\": \"test prompt\"}}").unwrap();

        let req = rx.recv().unwrap();
        assert_eq!(req.prompt, "test prompt");
    }

    #[test]
    fn test_handle_client_invalid_json() {
        let (tx, rx) = mpsc::channel();
        let (mut s1, s2) = std::os::unix::net::UnixStream::pair().unwrap();

        std::thread::spawn(move || {
            let _ = handle_client(s2, tx);
        });

        writeln!(s1, "invalid json").unwrap();

        assert!(rx.recv().is_err());
    }

    #[test]
    fn test_handle_client_empty_json() {
        let (tx, rx) = mpsc::channel();
        let (mut s1, s2) = std::os::unix::net::UnixStream::pair().unwrap();

        std::thread::spawn(move || {
            let _ = handle_client(s2, tx);
        });

        writeln!(s1, "{{}}").unwrap();

        let req = rx.recv().unwrap();
        assert_eq!(req.prompt, "");
    }

    #[test]
    fn test_spawn_client_handler_ok() {
        let (tx, rx) = mpsc::channel();
        let (s1, s2) = std::os::unix::net::UnixStream::pair().unwrap();

        spawn_client_handler(Ok(s2), tx);

        let mut s1_clone = s1.try_clone().unwrap();
        writeln!(s1_clone, "{{\"prompt\": \"test prompt\"}}").unwrap();

        let req = rx.recv().unwrap();
        assert_eq!(req.prompt, "test prompt");
    }

    #[test]
    fn test_spawn_client_handler_err() {
        let (tx, rx) = mpsc::channel();
        let err = std::io::Error::other("test error");
        spawn_client_handler(Err(err), tx);

        // Ensure no message is sent on the channel.
        assert!(rx.try_recv().is_err());
    }
}
