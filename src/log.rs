use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use crate::format::{FRAME_HEADER_SIZE, SEGMENT_HEADER, frame_header, frame_info};
use crate::{Error, Result};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub(crate) struct Location {
    pub(crate) segment: u64,
    pub(crate) offset: u64,
}

pub(crate) struct Log {
    directory: PathBuf,
    active: File,
    segment: u64,
    length: u64,
    target_bytes: u64,
    uncached_reads: bool,
    dirty: bool,
    pub(crate) data_syncs: u64,
    #[cfg(test)]
    pub(crate) fail_next_sync: bool,
    _lock: File,
}

impl Log {
    pub(crate) fn open(
        path: &Path,
        target_bytes: u64,
        mut replay: impl FnMut(Location, &[u8]) -> Result<()>,
    ) -> Result<Self> {
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        builder.mode(0o700);
        match builder.create(path) {
            Ok(()) => {
                let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
                sync_directory(parent.unwrap_or_else(|| Path::new(".")))?;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists && path.is_dir() => {}
            Err(error) => return Err(error.into()),
        }
        let directory = fs::canonicalize(path)?;
        let lock = private_file_options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(directory.join("LOCK"))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Error::Locked),
            Err(TryLockError::Error(error)) => return Err(error.into()),
        }

        let segments = list_segments(&directory)?;
        let mut active = None;
        for (index, segment) in segments.iter().copied().enumerate() {
            let last = index + 1 == segments.len();
            let file = OpenOptions::new()
                .read(true)
                .write(last)
                .open(segment_path(&directory, segment))?;
            let (file, length) = replay_segment(file, segment, last, &mut replay)?;
            if last {
                active = Some((segment, file, length));
            }
        }
        let (segment, active, length) = match active {
            Some(active) => active,
            None => (
                1,
                create_segment(&directory, 1)?,
                SEGMENT_HEADER.len() as u64,
            ),
        };
        Ok(Self {
            directory,
            active,
            segment,
            length,
            target_bytes,
            uncached_reads: false,
            dirty: false,
            data_syncs: 0,
            #[cfg(test)]
            fail_next_sync: false,
            _lock: lock,
        })
    }

    pub(crate) fn append_frame(&mut self, payload: &[u8]) -> Result<Location> {
        let frame_size = (FRAME_HEADER_SIZE + payload.len()) as u64;
        let next_length = self.length.saturating_add(frame_size);
        if self.length > SEGMENT_HEADER.len() as u64 && next_length > self.target_bytes {
            self.sync_pending()?;
            let next_segment = self
                .segment
                .checked_add(1)
                .ok_or(Error::SequenceExhausted)?;
            self.active = create_segment(&self.directory, next_segment)?;
            self.segment = next_segment;
            self.length = SEGMENT_HEADER.len() as u64;
        }
        let location = Location {
            segment: self.segment,
            offset: self.length + FRAME_HEADER_SIZE as u64,
        };
        self.active.write_all(&frame_header(payload))?;
        self.active.write_all(payload)?;
        self.dirty = true;
        self.length += frame_size;
        Ok(location)
    }

    pub(crate) fn sync_pending(&mut self) -> Result<()> {
        if self.dirty {
            #[cfg(test)]
            if std::mem::take(&mut self.fail_next_sync) {
                return Err(io::Error::other("injected data sync failure").into());
            }
            self.active.sync_all()?;
            self.dirty = false;
            self.data_syncs += 1;
        }
        Ok(())
    }

    pub(crate) fn read(&self, location: Location, len: usize) -> Result<Vec<u8>> {
        let mut file = self.open_at(location)?;
        let mut data = vec![0; len];
        file.read_exact(&mut data)?;
        Ok(data)
    }

    pub(crate) fn read_block(&self, location: Location, len: usize) -> Result<Vec<u8>> {
        let mut file = self.open_at(location)?;
        let segment_end = if location.segment == self.segment {
            self.length
        } else {
            file.metadata()?.len()
        };
        let available = segment_end.saturating_sub(location.offset);
        let mut data = vec![0; available.min(len as u64) as usize];
        file.read_exact(&mut data)?;
        Ok(data)
    }

    fn open_at(&self, location: Location) -> Result<File> {
        let mut file = File::open(segment_path(&self.directory, location.segment))?;
        if self.uncached_reads {
            disable_read_cache(&file)?;
        }
        file.seek(SeekFrom::Start(location.offset))?;
        Ok(file)
    }

    pub(crate) fn set_uncached_reads(&mut self, enabled: bool) -> io::Result<()> {
        if enabled && !cfg!(target_os = "macos") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "uncached reads require macOS",
            ));
        }
        self.uncached_reads = enabled;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn disable_read_cache(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;

    // The descriptor is open for the lifetime of the call. F_NOCACHE takes an int.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn disable_read_cache(_: &File) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "uncached reads require macOS",
    ))
}

fn segment_path(directory: &Path, id: u64) -> PathBuf {
    directory.join(format!("{id:016x}.mhl"))
}

fn list_segments(directory: &Path) -> Result<Vec<u64>> {
    let mut segments = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(stem) = name.to_str().and_then(|name| name.strip_suffix(".mhl")) else {
            continue;
        };
        let id = u64::from_str_radix(stem, 16)
            .map_err(|_| Error::Corrupt("invalid segment name".into()))?;
        if stem != format!("{id:016x}") || !entry.file_type()?.is_file() {
            return Err(Error::Corrupt("invalid segment file".into()));
        }
        segments.push(id);
    }
    segments.sort_unstable();
    for (index, segment) in segments.iter().enumerate() {
        if *segment != index as u64 + 1 {
            return Err(Error::Corrupt("gap in segment sequence".into()));
        }
    }
    Ok(segments)
}

fn create_segment(directory: &Path, id: u64) -> Result<File> {
    let path = segment_path(directory, id);
    if path.try_exists()? {
        return Err(Error::Corrupt(format!("segment {id} already exists")));
    }
    // Only the current lock holder can publish a segment. A pending file from
    // an interrupted publication can be removed without touching any segment.
    let temporary = directory.join(".memoryh-pending-segment");
    match fs::remove_file(&temporary) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let mut file = private_file_options()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(SEGMENT_HEADER)?;
    file.sync_all()?;
    fs::rename(temporary, &path)?;
    sync_directory(directory)?;
    Ok(OpenOptions::new().append(true).open(path)?)
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

fn private_file_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    #[cfg(unix)]
    options.mode(0o600);
    options
}

fn replay_segment(
    file: File,
    segment: u64,
    last: bool,
    replay: &mut impl FnMut(Location, &[u8]) -> Result<()>,
) -> Result<(File, u64)> {
    let length = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut segment_header = [0; SEGMENT_HEADER.len()];
    if length < SEGMENT_HEADER.len() as u64 {
        return Err(Error::Corrupt(format!("short header in segment {segment}")));
    }
    reader.read_exact(&mut segment_header)?;
    if segment_header != *SEGMENT_HEADER {
        return Err(Error::Corrupt(format!(
            "unknown format in segment {segment}"
        )));
    }
    let mut offset = SEGMENT_HEADER.len() as u64;
    let mut payload = Vec::new();
    while length - offset >= FRAME_HEADER_SIZE as u64 {
        let mut header = [0; FRAME_HEADER_SIZE];
        reader.read_exact(&mut header)?;
        let (len, checksum) =
            frame_info(header).map_err(|error| at_batch(error, segment, offset))?;
        let start = offset + FRAME_HEADER_SIZE as u64;
        if length - start < len as u64 {
            break;
        }
        payload.resize(len, 0);
        reader.read_exact(&mut payload)?;
        if crc32fast::hash(&payload) != checksum {
            return Err(Error::Corrupt(format!(
                "checksum mismatch in segment {segment} at byte {offset}"
            )));
        }
        replay(
            Location {
                segment,
                offset: start,
            },
            &payload,
        )
        .map_err(|error| at_batch(error, segment, offset))?;
        offset = start + len as u64;
    }
    let mut file = reader.into_inner();
    if offset != length {
        if !last {
            return Err(Error::Corrupt(format!(
                "incomplete batch in sealed segment {segment}"
            )));
        }
        file.set_len(offset)?;
        file.sync_all()?;
    }
    if last {
        file.seek(SeekFrom::Start(offset))?;
    }
    Ok((file, offset))
}

fn at_batch(error: Error, segment: u64, offset: u64) -> Error {
    Error::Corrupt(format!("segment {segment} at byte {offset}: {error}"))
}
