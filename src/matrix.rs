use crate::engine::SharedCore;
use anyhow::Result;
use erupt::{
    utils::allocator::{Allocation, MemoryTypeFinder},
    vk1_0 as vk,
};

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Option<Allocation<vk::Buffer>>,
    core: SharedCore,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, core: SharedCore) -> Result<Self> {
        let buffer_size = cols * rows * std::mem::size_of::<f32>();
        let create_info = vk::BufferCreateInfoBuilder::new()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(buffer_size as u64);

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
}

impl Drop for Matrix {
    fn drop(&mut self) {
        self.core
            .allocator()
            .unwrap()
            .free(&self.core.device, self.data.take().unwrap());
    }
}
