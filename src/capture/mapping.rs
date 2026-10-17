use std::os::fd::{AsRawFd, OwnedFd};

use anyhow::{Context, Result};

const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x4008_6200;
const DMA_BUF_SYNC_READ: u64 = 1;
const DMA_BUF_SYNC_END: u64 = 4;

pub(super) struct Mapping {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: The mapping is read-only and unmapped only in `Drop`, so moving it to another thread
// cannot invalidate it.
unsafe impl Send for Mapping {}
// SAFETY: Only shared, read-only slices are handed out. A buffer is read only after its capture
// has completed, and `State::take_buffer()` reuses it only once its `Arc` count is one, so
// nothing writes it while a slice is live. `ReadGuard`'s DMA_BUF_IOCTL_SYNC only keeps CPU
// caches coherent.
unsafe impl Sync for Mapping {}

impl Mapping {
    pub(super) fn new(fd: &OwnedFd, len: usize) -> Result<Self> {
        // SAFETY: A null address lets the kernel pick the placement, `fd` is a live buffer fd for
        // the call, and the result is compared against MAP_FAILED below.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error()).context("mmap capture buffer");
        }
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `len` come from the successful mmap in `new`, and this is the only
        // place that unmaps them.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

/// CPU pixels borrowed from a linear mapping or GBM's detiled view.
pub struct ReadGuard<'a> {
    bytes: &'a [u8],
    /// `None` for wl_shm memory and GBM views, which need no DMA_BUF sync here.
    fd: Option<&'a OwnedFd>,
    pub stride: usize,
}

impl<'a> ReadGuard<'a> {
    pub(super) fn new(
        map: &'a Mapping,
        fd: Option<&'a OwnedFd>,
        offset: usize,
        stride: usize,
    ) -> Result<Self> {
        if let Some(fd) = fd {
            sync(fd, DMA_BUF_SYNC_READ)?;
        }
        // SAFETY: `offset <= map.len` (the only caller sizes the mapping as `offset + stride *
        // height`), so the slice stays inside the mapping, and the `'a` borrow of `map` keeps
        // it mapped. `sync` above brackets CPU access to the dmabuf.
        let bytes = unsafe { std::slice::from_raw_parts(map.ptr.add(offset), map.len - offset) };
        Ok(Self { bytes, fd, stride })
    }

    pub(super) fn borrowed(bytes: &'a [u8], stride: usize) -> Self {
        Self {
            bytes,
            fd: None,
            stride,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }

    pub fn row(&self, y: usize, width_bytes: usize) -> &[u8] {
        &self.bytes[y * self.stride..y * self.stride + width_bytes]
    }
}

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        if let Some(fd) = self.fd {
            let _ = sync(fd, DMA_BUF_SYNC_READ | DMA_BUF_SYNC_END);
        }
    }
}

fn sync(fd: &OwnedFd, flags: u64) -> Result<()> {
    loop {
        // SAFETY: `fd` is live, and DMA_BUF_IOCTL_SYNC takes a pointer to a single u64 of flags
        // (struct dma_buf_sync), which `flags` provides for the duration of the call.
        let r = unsafe { libc::ioctl(fd.as_raw_fd(), DMA_BUF_IOCTL_SYNC, &flags) };
        if r == 0 {
            return Ok(());
        }
        let e = std::io::Error::last_os_error();
        if !matches!(e.raw_os_error(), Some(libc::EINTR | libc::EAGAIN)) {
            return Err(e).context("DMA_BUF_IOCTL_SYNC");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::os::unix::fs::FileExt;

    #[test]
    fn shm_memfd_reads_pixels_without_dmabuf_sync() -> Result<()> {
        let alloc = super::super::alloc::Allocator::allocate_shm(2, 2)?;
        assert!(alloc.gbm.is_none());
        assert_eq!(alloc.modifier, super::super::alloc::MOD_LINEAR);
        assert_eq!(alloc.planes.len(), 1);
        let p = &alloc.planes[0];
        assert_eq!((p.offset, p.stride), (0, 8));
        let file = File::from(p.fd.try_clone()?);
        assert_eq!(file.metadata()?.len(), 16);
        let pixels = [1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255];
        file.write_all_at(&pixels, 0)?;
        let map = Mapping::new(&p.fd, pixels.len())?;
        // A memfd rejects DMA_BUF_IOCTL_SYNC. The dmabuf path must still issue it.
        assert!(ReadGuard::new(&map, Some(&p.fd), 0, p.stride as usize).is_err());
        for _ in 0..2 {
            let guard = ReadGuard::new(&map, None, 0, p.stride as usize)?;
            assert_eq!(guard.row(0, 8), &pixels[..8]);
            assert_eq!(guard.row(1, 8), &pixels[8..]);
        }
        Ok(())
    }
}
