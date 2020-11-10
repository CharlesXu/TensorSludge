use crate::desc_set_allocator::DescriptorSetAllocator;
use crate::engine::SharedCore;
use crate::matrix::Matrix;
use anyhow::{bail, Context, Result};
use erupt::{utils::decode_spv, vk1_0 as vk, DeviceLoader};
use std::ffi::CString;

const LOCAL_SIZE_X: u32 = 16;
const LOCAL_SIZE_Y: u32 = 16;
const LOCAL_SIZE_Z: u32 = 4;

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
    pipeline: vk::Pipeline,
    push_constant_sizes: SizeConstants,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct MatrixDims {
    cols: u32,
    rows: u32,
    layers: u32,
    trans: u32,
}

#[repr(C)]
#[derive(Default, Copy, Clone)]
struct SizeConstants {
    a_dim: MatrixDims,
    b_dim: MatrixDims,
    out_dim: MatrixDims,
}

unsafe impl bytemuck::Zeroable for MatrixDims {}
unsafe impl bytemuck::Pod for MatrixDims {}

unsafe impl bytemuck::Zeroable for SizeConstants {}
unsafe impl bytemuck::Pod for SizeConstants {}

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
        const PUSH_CONSTANT_SIZES: usize = std::mem::size_of::<SizeConstants>();
        assert_eq!(PUSH_CONSTANT_SIZES, std::mem::size_of::<[u32; 4 * 3]>());
        let push_constant_ranges = [vk::PushConstantRangeBuilder::new()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_SIZES as u32)];

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

        let okay = match (a_transpose, b_transpose) {
            (false, false) => a.cols() == b.rows(),
            (false, true) => a.cols() == b.cols(),
            (true, false) => a.rows() == b.rows(),
            (true, true) => a.rows() == b.cols(),
        };

        if !okay {
            bail!("Cannot multiply; Matrix dimension mismatch between \"{}\" and \"{}\"",
                a.name(),
                b.name(),
            );
        }

        let outer_dims = match (a_transpose, b_transpose) {
            (false, false) => (a.rows(), b.cols()),
            (false, true) => (a.rows(), b.rows()),
            (true, false) => (a.cols(), b.cols()),
            (true, true) => (a.cols(), b.rows()),
        };

        let out_trans = false;
        let output_dims = if out_trans {
            (dst.cols(), dst.rows())
        } else {
            (dst.rows(), dst.cols())
        };

        if outer_dims != output_dims {
            bail!("Output matrix \"{}\" size incompatible.", dst.name());
        }

        fn matrix_dims(matrix: &Matrix, trans: bool) -> MatrixDims {
            MatrixDims {
                rows: matrix.rows() as _,
                cols: matrix.cols() as _,
                layers: matrix.layers() as _,
                trans: if trans { 1 } else { 0 },
            }
        }

        let mut push_constant_sizes = SizeConstants {
            a_dim: matrix_dims(a, a_transpose),
            b_dim: matrix_dims(b, b_transpose),
            out_dim: matrix_dims(dst, false),
        };
        //push_constant_sizes.b_dim.trans ^= 1;

        Ok(Invocation {
            pipeline: self.pipeline,
            pipeline_layout: self.pipeline_layout,
            descriptor_set,
            push_constant_sizes,
        })
    }
}

impl crate::engine::Invocation for Invocation {
    fn dispatch(&self, device: &DeviceLoader, command_buffer: vk::CommandBuffer) {
        unsafe {
            let slice = [self.push_constant_sizes];
            let constants: &[u8] = bytemuck::cast_slice(&slice);
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

            let abbrev = &self.push_constant_sizes.out_dim;
            let (x, y) = if self.push_constant_sizes.out_dim.trans == 0 {
                (abbrev.rows, abbrev.cols)
            } else {
                (abbrev.cols, abbrev.rows)
            };

            let invocations_x = (x / LOCAL_SIZE_X) + 1;
            let invocations_y = (y / LOCAL_SIZE_Y) + 1;
            let invocations_z = (abbrev.layers / LOCAL_SIZE_Z) + 1;

            device.cmd_dispatch(command_buffer, invocations_x, invocations_y, invocations_z);
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
