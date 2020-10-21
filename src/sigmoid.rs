use anyhow::{Result, Context};
use erupt::{vk1_0 as vk, DeviceLoader, utils::decode_spv};
use std::sync::Arc;
use std::ffi::CString;
use crate::engine::SharedCore;

pub struct Sigmoid {
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: SharedCore,
}

impl Sigmoid {
    pub fn new(core: SharedCore) -> Result<Self> {
        // Layout:
        let bindings = [vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout =
            unsafe { core.device.create_descriptor_set_layout(&create_info, None, None) }.result()?;

        // Load shader
        let shader_spirv =
            std::fs::read("shaders/sigmoid.comp.spv").context("Sigmoid shader failed to load")?;
        let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
        let shader_module =
            unsafe { core.device.create_shader_module(&create_info, None, None) }.result()?;

        // Pipeline
        let descriptor_set_layouts = [descriptor_set_layout];
        let create_info =
            vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&descriptor_set_layouts);
        let pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        let entry_point = CString::new("main")?;
        let stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(shader_module)
            .name(&entry_point)
            .build();
        let create_info = vk::ComputePipelineCreateInfoBuilder::new()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline =
            unsafe { core.device.create_compute_pipelines(None, &[create_info], None) }.result()?[0];

        unsafe {
            core.device.destroy_shader_module(Some(shader_module), None);
        }

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            core,
        })
    }

    pub fn desc_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn desc_pool_sizes(sizes: &mut Vec<vk::DescriptorPoolSizeBuilder>) {
        sizes.push(vk::DescriptorPoolSizeBuilder::new()
        ._type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1))
    }
}

impl Drop for Sigmoid {
    fn drop(&mut self) {
        unsafe {
            self.core.device.destroy_pipeline(Some(self.pipeline), None);
            self.core.device
                .destroy_descriptor_set_layout(Some(self.descriptor_set_layout), None);
        }
    }
}
