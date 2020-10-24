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
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, core: SharedCore) -> Result<Self> {
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

    fn map(&mut self) -> Result<MappedMemory> {
        let mapping = self.allocation().map(&self.core.device, ..).result()?;
        Ok(mapping)
    }

    pub fn read(&mut self, buf: &mut [f32]) -> Result<()> {
        if self.rows() * self.cols() != buf.len() {
            bail!("Mismatched buffer sizes");
        }
        let mapping = self.map()?;
        buf.copy_from_slice(bytemuck::cast_slice(mapping.read()));
        mapping.unmap(&self.core.device).result()?;
        Ok(())
    }

    pub fn write(&mut self, buf: &[f32]) -> Result<()> {
        if self.rows() * self.cols() != buf.len() {
            bail!("Mismatched buffer sizes");
        }
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
