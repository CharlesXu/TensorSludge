use crate::desc_set_allocator::DescriptorSetAllocator;
use crate::matrix::Matrix;
use anyhow::{Context, Result};
use erupt::{utils::decode_spv, vk1_0 as vk, DeviceLoader};
use std::ffi::CString;
use vk_core::SharedCore;

pub struct Sigmoid {
    pipeline: vk::Pipeline,
    deriv_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: SharedCore,
    ds_allocator: DescriptorSetAllocator,
}

pub struct Invocation {
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    invocations: u32,
    pipeline: vk::Pipeline,
}

const SHADER_SPV_PATH: &str = "shaders/sigmoid.comp.spv";
const DERIV_SHADER_SPV_PATH: &str = "shaders/sigmoid_deriv.comp.spv";
const LOCAL_SIZE_X: u32 = 16;

impl Sigmoid {
    pub fn new(core: SharedCore) -> Result<Self> {
        // Layout:
        let bindings = [vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&create_info, None, None)
        }
        .result()?;

        // Load shader
        let shader_spirv =
            std::fs::read(SHADER_SPV_PATH).context("Sigmoid shader failed to load")?;
        let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
        let shader_module =
            unsafe { core.device.create_shader_module(&create_info, None, None) }.result()?;

        let shader_spirv =
            std::fs::read(DERIV_SHADER_SPV_PATH).context("Sigmoid shader failed to load")?;
        let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
        let deriv_shader_module =
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

        let stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(deriv_shader_module)
            .name(&entry_point)
            .build();
        let deriv_create_info = vk::ComputePipelineCreateInfoBuilder::new()
            .stage(stage)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            core.device
                .create_compute_pipelines(None, &[create_info, deriv_create_info], None)
        }
        .result()?;

        let pipeline = pipelines[0];
        let deriv_pipeline = pipelines[1];

        unsafe {
            core.device.destroy_shader_module(Some(shader_module), None);
            core.device
                .destroy_shader_module(Some(deriv_shader_module), None);
        }

        // Create descriptor set allocator
        let dpsbs = vec![vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let ds_allocator = DescriptorSetAllocator::new(dpsbs, descriptor_set_layout, core.clone());

        Ok(Self {
            pipeline,
            deriv_pipeline,
            pipeline_layout,
            descriptor_set_layout,
            ds_allocator,
            core,
        })
    }

    pub fn invoke_deriv(&mut self, matrix: &Matrix) -> Result<Invocation> {
        self.invoke_internal(matrix, self.deriv_pipeline)
    }

    pub fn invoke(&mut self, matrix: &Matrix) -> Result<Invocation> {
        self.invoke_internal(matrix, self.pipeline)
    }

    pub fn invoke_internal(
        &mut self,
        matrix: &Matrix,
        pipeline: vk::Pipeline,
    ) -> Result<Invocation> {
        let descriptor_set = self.ds_allocator.pop()?;
        unsafe {
            self.core.device.update_descriptor_sets(
                &[vk::WriteDescriptorSetBuilder::new()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                        .buffer(matrix.buffer())
                        .offset(0)
                        .range(vk::WHOLE_SIZE)])],
                &[],
            )
        };

        let invocations = (matrix.size() as u32 / LOCAL_SIZE_X) + 1;

        Ok(Invocation {
            pipeline,
            pipeline_layout: self.pipeline_layout,
            descriptor_set,
            invocations,
        })
    }
}

impl crate::engine::Invocation for Invocation {
    fn dispatch(&self, device: &DeviceLoader, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            device.cmd_dispatch(command_buffer, self.invocations, 1, 1);
        }
    }
}

impl Drop for Sigmoid {
    fn drop(&mut self) {
        unsafe {
            self.core.device.destroy_pipeline(Some(self.pipeline), None);
            self.core
                .device
                .destroy_pipeline_layout(Some(self.pipeline_layout), None);
            self.core
                .device
                .destroy_pipeline(Some(self.deriv_pipeline), None);
            self.core
                .device
                .destroy_descriptor_set_layout(Some(self.descriptor_set_layout), None);
        }
    }
}
