use anyhow::{bail, ensure, Result};
use erupt::vk1_0 as vk;
use gpu_alloc::{MemoryBlock, Request};
use gpu_alloc_erupt::EruptMemoryDevice;
use vk_core::SharedCore;

pub struct Matrix {
    rows: usize,
    cols: usize,
    layers: usize,

    allocation: Option<MemoryBlock<vk::DeviceMemory>>,
    buffer: vk::Buffer,

    core: SharedCore,
    name: String,
    cpu_visible: bool,
}

impl Matrix {
    pub fn new(
        rows: usize,
        cols: usize,
        layers: usize,
        cpu_visible: bool,
        name: impl Into<String>,
        core: SharedCore,
    ) -> Result<Self> {
        ensure!(rows > 0, "Rows must be nonzero!");
        ensure!(cols > 0, "Cols must be nonzero!");
        ensure!(layers > 0, "Layers must be nonzero!");
        let name = name.into();
        let buffer_size = layers * cols * rows * std::mem::size_of::<f32>();
        let create_info = vk::BufferCreateInfoBuilder::new()
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(buffer_size as u64);

        // Create a buffer
        let buffer = unsafe { core.device.create_buffer(&create_info, None, None) }.result()?;

        // Allocate memory for it
        use gpu_alloc::UsageFlags;
        let usage = match cpu_visible {
            true => UsageFlags::DOWNLOAD | UsageFlags::UPLOAD | gpu_alloc::UsageFlags::HOST_ACCESS,
            false => UsageFlags::FAST_DEVICE_ACCESS,
        };

        let request = Request {
            size: buffer_size as _,
            align_mask: std::mem::align_of::<f32>() as _,
            usage,
            memory_types: !0,
        };

        let allocation = unsafe {
            core.allocator()?
                .alloc(EruptMemoryDevice::wrap(&core.device), request)?
        };

        // Bind that memory
        unsafe {
            core.device
                .bind_buffer_memory(buffer, *allocation.memory(), allocation.offset())
                .result()?;
        }

        Ok(Self {
            rows,
            cols,
            layers,
            allocation: Some(allocation),
            buffer,
            core,
            name,
            cpu_visible,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn size(&self) -> usize {
        self.rows() * self.cols() * self.layers()
    }

    pub fn size_bytes(&self) -> usize {
        self.rows() * self.cols() * self.layers() * std::mem::size_of::<f32>()
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn chk_buf_mismatch(&self, buf: &[f32]) -> Result<()> {
        if buf.len() != self.size() {
            bail!(
                "Buffer of size {} does not match the length of data in matrix \"{}\", {}",
                buf.len(),
                self.name,
                self.size(),
            )
        } else {
            Ok(())
        }
    }

    pub fn read(&mut self, buf: &mut [f32]) -> Result<()> {
        if !self.cpu_visible {
            bail!("Cannot read from GPU-only matrix \"{}\"", self.name());
        }
        self.chk_buf_mismatch(buf)?;
        unsafe {
            self.allocation.as_mut().unwrap().read_bytes(
                EruptMemoryDevice::wrap(&self.core.device),
                0,
                bytemuck::cast_slice_mut(buf),
            )?;
        }
        Ok(())
    }

    pub fn write(&mut self, buf: &[f32]) -> Result<()> {
        if !self.cpu_visible {
            bail!("Cannot write to GPU-only matrix \"{}\"", self.name());
        }
        self.chk_buf_mismatch(buf)?;
        unsafe {
            self.allocation.as_mut().unwrap().write_bytes(
                EruptMemoryDevice::wrap(&self.core.device),
                0,
                &bytemuck::cast_slice(buf),
            )?;
        }
        Ok(())
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            self.core.allocator().unwrap().dealloc(
                EruptMemoryDevice::wrap(&self.core.device),
                self.allocation.take().unwrap(),
            );
            self.core.device.destroy_buffer(Some(self.buffer), None);
        }
    }
}
