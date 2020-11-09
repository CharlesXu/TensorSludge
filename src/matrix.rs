use crate::engine::SharedCore;
use anyhow::{bail, Result};
use erupt::{
    utils::allocator::{Allocation, MappedMemory, MemoryTypeFinder},
    vk1_0 as vk,
};

pub struct Matrix {
    rows: usize,
    cols: usize,
    layers: usize,
    name: String,
    data: Option<Allocation<vk::Image>>,
    core: SharedCore,
}

const IMAGE_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

impl Matrix {
    pub fn new(
        rows: usize,
        cols: usize,
        layers: usize,
        name: impl Into<String>,
        core: SharedCore,
    ) -> Result<Self> {
        let name = name.into();
        let buffer_size = cols * rows * std::mem::size_of::<f32>();
        let create_info = vk::BufferCreateInfoBuilder::new()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(buffer_size as u64);

        let extent_3d = vk::Extent3DBuilder::new()
            .width(rows as _)
            .height(cols as _)
            .depth(1)
            .build();

        let create_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(extent_3d)
            .mip_levels(1)
            .array_layers(layers as _)
            .format(IMAGE_FORMAT)
            .tiling(vk::ImageTiling::LINEAR)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlagBits::_1);

        let image = unsafe { core.device.create_image(&create_info, None, None) }.result()?;

        let data = core.allocator()?
            .allocate(&core.device, image, MemoryTypeFinder::dynamic())
            .result()?;

        let data = Some(data);

        Ok(Self {
            rows,
            cols,
            layers,
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

    pub fn allocation<'a>(&'a self) -> &'a Allocation<vk::Image> {
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
        buf.copy_from_slice(bytemuck::cast_slice(mapping.read()));
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
