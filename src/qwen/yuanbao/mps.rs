//! Persistent native Metal worker. Rust owns all tokenization and ranking.
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use crate::Result;
use crate::inference::inference_error;

pub(super) struct Mps {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
}

impl Mps {
    pub(super) fn open(dir: &Path) -> Result<Self> {
        let mut child = Command::new(dir.join("yuanbao-mps"))
            .arg(dir.join("yuanbao-mps.pt"))
            .env("PYTORCH_ENABLE_MPS_FALLBACK", "0")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let mut worker = Self {
            input: child.stdin.take().unwrap(),
            output: child.stdout.take().unwrap(),
            child,
        };
        let mut hello = [0; 4];
        worker.output.read_exact(&mut hello)?;
        if &hello != b"MPS1" {
            return Err(inference_error("invalid Yuanbao MPS handshake"));
        }
        Ok(worker)
    }

    pub(super) fn logits(
        &mut self,
        shape: [usize; 2],
        ids: &[i64],
        mask: &[i64],
    ) -> Result<Vec<f32>> {
        let mut request = Vec::with_capacity(8 + 16 * ids.len());
        for dim in shape {
            request.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        for value in ids.iter().chain(mask) {
            request.extend_from_slice(&value.to_le_bytes());
        }
        self.input.write_all(&request)?;
        self.input.flush()?;
        let mut bytes = vec![0; shape[0] * 8];
        self.output.read_exact(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }
}

impl Drop for Mps {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
