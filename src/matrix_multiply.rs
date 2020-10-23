use crate::desc_set_allocator::DescriptorSetAllocator;
use crate::engine::SharedCore;
use crate::matrix::Matrix;
use anyhow::{ensure, Context, Result};
use erupt::{utils::decode_spv, vk1_0 as vk, DeviceLoader};
use std::ffi::CString;

const LOCAL_SIZE_X: u32 = 16;
const LOCAL_SIZE_Y: u32 = 16;

pub struct MatrixMultiply {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: SharedCore,
    ds_allocator: DescriptorSetAllocator,
}

pub struct Invocation {
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    a_cols: u32,   // Number of cols in a
    b_cols: u32,   // Number of cols in b
    out_rows: u32, // Rows of in_a, product
    out_cols: u32, // Columns of in_b, product
    inner_rc: u32, // Columns of in_a, Rows of in_B
    a_transpose: bool,
    b_transpose: bool,
    pipeline: vk::Pipeline,
}

impl MatrixMultiply {
    pub fn new(core: SharedCore) -> Result<Self> {
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
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(2)
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

        // Load shader
        let shader_spirv = std::fs::read("shaders/matrix_multiply.comp.spv")
            .context("MatrixMultiply shader failed to load")?;
        let shader_decoded = decode_spv(&shader_spirv).context("Shader decode failed")?;
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&shader_decoded);
        let shader_module =
            unsafe { core.device.create_shader_module(&create_info, None, None) }.result()?;

        // Pipeline
        let push_constant_ranges = [vk::PushConstantRangeBuilder::new()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<[u32; 7]>() as u32)];

        let descriptor_set_layouts = [descriptor_set_layout];
        let create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_ranges);
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
        let pipeline = unsafe {
            core.device
                .create_compute_pipelines(None, &[create_info], None)
        }
        .result()?[0];

        unsafe {
            core.device.destroy_shader_module(Some(shader_module), None);
        }

        // Create descriptor set allocator
        let dpsbs = vec![
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1);
            3
        ];
        let ds_allocator = DescriptorSetAllocator::new(dpsbs, descriptor_set_layout, core.clone());

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            ds_allocator,
            core,
        })
    }

    pub fn invoke(
        &mut self,
        a: &Matrix,
        a_transpose: bool,
        b: &Matrix,
        b_transpose: bool,
        dst: &Matrix,
    ) -> Result<Invocation> {
        let descriptor_set = self.ds_allocator.pop()?;
        unsafe {
            self.core.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(*a.allocation().object())
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(descriptor_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(*b.allocation().object())
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(descriptor_set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfoBuilder::new()
                            .buffer(*dst.allocation().object())
                            .offset(0)
                            .range(vk::WHOLE_SIZE)]),
                ],
                &[],
            )
        };

        let invalid_msg = "Input matrix dimensions invalid for multiplication";
        match (a_transpose, b_transpose) {
            (false, false) => ensure!(a.cols() == b.rows(), invalid_msg),
            (true, false) => ensure!(a.rows() == b.rows(), invalid_msg),
            (false, true) => ensure!(a.cols() == b.cols(), invalid_msg),
            (true, true) => ensure!(a.rows() == b.cols(), invalid_msg),
        }

        let inner_rc = if b_transpose { b.cols() } else { b.rows() } as u32;

        let out_rows = if a_transpose { a.cols() } else { a.rows() } as u32;

        let out_cols = if b_transpose { b.rows() } else { b.cols() } as u32;

        let invalid_msg = "Output matrix dimensions invalid for input sizes in multiplication";
        ensure!(
            dst.cols() as u32 == out_cols && dst.rows() as u32 == out_rows,
            invalid_msg
        );

        Ok(Invocation {
            a_cols: a.cols() as u32,
            b_cols: b.cols() as u32,
            pipeline: self.pipeline,
            pipeline_layout: self.pipeline_layout,
            descriptor_set,
            a_transpose,
            b_transpose,
            out_rows,
            out_cols,
            inner_rc,
        })
    }
}

impl crate::engine::Invocation for Invocation {
    fn dispatch(&self, device: &DeviceLoader, command_buffer: vk::CommandBuffer) {
        unsafe {
            let constants = [
                self.a_cols,
                self.b_cols,
                self.out_rows,
                self.out_cols,
                self.inner_rc,
                if self.a_transpose { 1 } else { 0 },
                if self.b_transpose { 1 } else { 0 },
            ];
            let constants: &[u8] = bytemuck::cast_slice(&constants[..]);
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                constants.len() as u32,
                constants.as_ptr() as _,
            );

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

            let invocations_x = (self.out_cols as u32 / LOCAL_SIZE_X) + 1;
            let invocations_y = (self.out_rows as u32 / LOCAL_SIZE_Y) + 1;

            device.cmd_dispatch(command_buffer, invocations_x, invocations_y, 1);
        }
    }
}

impl Drop for MatrixMultiply {
    fn drop(&mut self) {
        unsafe {
            self.core.device.destroy_pipeline(Some(self.pipeline), None);
            self.core
                .device
                .destroy_descriptor_set_layout(Some(self.descriptor_set_layout), None);
        }
    }
}
