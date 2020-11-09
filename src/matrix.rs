use crate::engine::SharedCore;
use anyhow::{bail, Result};
use erupt::{
    utils::allocator::{Allocation, MappedMemory, MemoryTypeFinder},
    vk1_0 as vk,
};

pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Option<Allocation<vk::Buffer>>,
    core: SharedCore,
    name: String,
}

impl Matrix {
    pub fn new(
        rows: usize,
        cols: usize,
        name: impl Into<String>,
        core: SharedCore,
    ) -> Result<Self> {
        let name = name.into();
        let buffer_size = cols * rows * std::mem::size_of::<f32>();
        let create_info = vk::BufferCreateInfoBuilder::new()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(buffer_size as u64);

        // TODO: This should really use a staging buffer... Perhaps make host-visible mats an
        // option? Still need to be able to write random numbers but a compute shader could be used
        // for that...
        let buffer = unsafe { core.device.create_buffer(&create_info, None, None) }.result()?;
        let data = core
            .allocator()?
            .allocate(&core.device, buffer, MemoryTypeFinder::dynamic())
            .result()?;
        let data = Some(data);

        Ok(Self {
            rows,
            cols,
            data,
            core,
            name,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn allocation<'a>(&'a self) -> &'a Allocation<vk::Buffer> {
        self.data.as_ref().unwrap()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn map(&mut self) -> Result<MappedMemory> {
        let mapping = self.allocation().map(&self.core.device, ..).result()?;
        Ok(mapping)
    }

    fn chk_buf_mismatch(&self, buf: &[f32]) -> Result<()> {
        let mat_len = self.rows() * self.cols();
        if buf.len() != mat_len {
            bail!(
                "Buffer of size {} does not match the length of data in matrix \"{}\", {}",
                buf.len(),
                self.name,
                mat_len
            )
        } else {
            Ok(())
        }
    }

    pub fn read(&mut self, buf: &mut [f32]) -> Result<()> {
        self.chk_buf_mismatch(buf)?;
        let mapping = self.map()?;
        buf.copy_from_slice(&bytemuck::cast_slice(mapping.read())[..buf.len()]);
        mapping.unmap(&self.core.device).result()?;
        Ok(())
    }

    pub fn write(&mut self, buf: &[f32]) -> Result<()> {
        self.chk_buf_mismatch(buf)?;
        let mut mapping = self.map()?;
        mapping.import(bytemuck::cast_slice(buf));
        mapping.unmap(&self.core.device).result()?;
        Ok(())
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        self.core
            .allocator()
            .unwrap()
            .free(&self.core.device, self.data.take().unwrap());
    }
}
