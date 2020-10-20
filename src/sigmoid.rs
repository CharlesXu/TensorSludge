use anyhow::Result;
use erupt::{vk1_0 as vk, DeviceLoader};
use std::sync::Arc;

pub struct Sigmoid {
    descriptor_set_layout: vk::DescriptorSetLayout,
    device: Arc<DeviceLoader>,
}

impl Sigmoid {
    pub fn new(device: Arc<DeviceLoader>) -> Result<Self> {
        // Layout:
        let bindings = [vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&create_info, None, None) }.result()?;

        Ok(Self {
            descriptor_set_layout,
            device,
        })
    }
}

impl Drop for Sigmoid {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(Some(self.descriptor_set_layout), None);
        }
    }
}
