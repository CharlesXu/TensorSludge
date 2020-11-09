use crate::desc_set_allocator::DescriptorSetAllocator;
use crate::engine::SharedCore;
use crate::matrix::Matrix;
use anyhow::{ensure, Context, Result};
use erupt::{utils::decode_spv, vk1_0 as vk, DeviceLoader};
use std::ffi::CString;

pub struct ElementwiseArithmetic {
    mult_pipeline: vk::Pipeline,
    add_pipeline: vk::Pipeline,
    sub_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: SharedCore,
    ds_allocator: DescriptorSetAllocator,
    local_size_x: u32,
}

pub struct Invocation {
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    invocations: u32,
    pipeline: vk::Pipeline,
}

const MULT_SHADER_PATH: &str = "shaders/elem_mult.comp.spv";
const ADD_SHADER_PATH: &str = "shaders/elem_add.comp.spv";
const SUB_SHADER_PATH: &str = "shaders/elem_sub.comp.spv";

impl ElementwiseArithmetic {
    pub fn new(core: SharedCore, local_size_x: u32) -> Result<Self> {
        // Layout:
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&create_info, None, None)
        }
        .result()?;

        // Load shaders
        let shader_spirv = std::fs::read(MULT_SHADER_PATH)
            .context("Elementwise multiply shader failed to load")?;
        let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
        let mult_shader_module =
            unsafe { core.device.create_shader_module(&create_info, None, None) }.result()?;

        let shader_spirv =
            std::fs::read(ADD_SHADER_PATH).context("Elementwise add shader failed to load")?;
        let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
        let add_shader_module =
            unsafe { core.device.create_shader_module(&create_info, None, None) }.result()?;

        let shader_spirv =
            std::fs::read(SUB_SHADER_PATH).context("Elementwise subtract shader failed to load")?;
        let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
        let sub_shader_module =
            unsafe { core.device.create_shader_module(&create_info, None, None) }.result()?;

        // Pipelines
        let descriptor_set_layouts = [descriptor_set_layout];
        let create_info =
            vk::PipelineLayoutCreateInfoBuilder::new().set_layouts(&descriptor_set_layouts);
        let pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        let entry_point = CString::new("main")?;

        let mult_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(mult_shader_module)
            .name(&entry_point)
            .build();
        let mult_create_info = vk::ComputePipelineCreateInfoBuilder::new()
            .stage(mult_stage)
            .layout(pipeline_layout);

        let add_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(add_shader_module)
            .name(&entry_point)
            .build();
        let add_create_info = vk::ComputePipelineCreateInfoBuilder::new()
            .stage(add_stage)
            .layout(pipeline_layout);

        let sub_stage = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(sub_shader_module)
            .name(&entry_point)
            .build();
        let sub_create_info = vk::ComputePipelineCreateInfoBuilder::new()
            .stage(sub_stage)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            core.device.create_compute_pipelines(
                None,
                &[mult_create_info, add_create_info, sub_create_info],
                None,
            )
        }
        .result()?;
        let mult_pipeline = pipelines[0];
        let add_pipeline = pipelines[1];
        let sub_pipeline = pipelines[2];

        unsafe {
            core.device
                .destroy_shader_module(Some(mult_shader_module), None);
            core.device
                .destroy_shader_module(Some(add_shader_module), None);
        }

        // Create descriptor set allocator
        let dpsbs = vec![
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        ];
        let ds_allocator = DescriptorSetAllocator::new(dpsbs, descriptor_set_layout, core.clone());

        Ok(Self {
            mult_pipeline,
            add_pipeline,
            sub_pipeline,
            pipeline_layout,
            descriptor_set_layout,
            ds_allocator,
            core,
            local_size_x,
        })
    }

    pub fn invoke_sub(&mut self, product: &Matrix, scalars: &Matrix) -> Result<Invocation> {
        self.invoke(product, scalars, self.sub_pipeline)
    }

    pub fn invoke_add(&mut self, product: &Matrix, scalars: &Matrix) -> Result<Invocation> {
        self.invoke(product, scalars, self.add_pipeline)
    }

    pub fn invoke_mult(&mut self, product: &Matrix, scalars: &Matrix) -> Result<Invocation> {
        self.invoke(product, scalars, self.mult_pipeline)
    }

    fn invoke(
        &mut self,
        product: &Matrix,
        scalars: &Matrix,
        pipeline: vk::Pipeline,
    ) -> Result<Invocation> {
        let warning = "Dimensions must match for elementwise ops";
        ensure!(product.cols() == scalars.cols(), warning);
        ensure!(product.rows() == scalars.rows(), warning);

        let descriptor_set = self.ds_allocator.pop()?;
        unsafe {
            self.core.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(*product.allocation().object())
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(descriptor_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(*scalars.allocation().object())
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                ],
                &[],
            )
        };

        let invocations = ((product.rows() * scalars.cols()) as u32 / self.local_size_x) + 1;

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

impl Drop for ElementwiseArithmetic {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device
                .destroy_pipeline(Some(self.mult_pipeline), None);
            self.core
                .device
                .destroy_pipeline(Some(self.add_pipeline), None);
            self.core
                .device
                .destroy_descriptor_set_layout(Some(self.descriptor_set_layout), None);
        }
    }
}
