use crate::engine::SharedCore;
use anyhow::Result;
use erupt::{
    utils::allocator::{Allocation, MemoryTypeFinder, MappedMemory},
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

    fn map(&mut self) -> Result<MappedMemory> {
        let mapping = self
            .data
            .as_ref()
            .unwrap()
            .map(&self.core.device, ..)
            .result()?;
        Ok(mapping)
    }

    pub fn read(&mut self, buf: &mut [f32]) -> Result<()> {
        let mapping = self.map()?;
        buf.copy_from_slice(bytemuck::cast_slice(mapping.read()));
        mapping.unmap(&self.core.device).result()?;
        Ok(())
    }

    pub fn write(&mut self, buf: &[f32]) -> Result<()> {
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
