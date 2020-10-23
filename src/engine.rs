use crate::matrix::Matrix;
use crate::matrix_multiply::MatrixMultiply;
use crate::sigmoid::Sigmoid;
use crate::Operation;
use anyhow::{bail, format_err, Context, Result};
use erupt::{
    cstr,
    utils::{
        allocator::{Allocator, AllocatorCreateInfo},
        loading::DefaultEntryLoader,
    },
    vk1_0 as vk, DeviceLoader, EntryLoader, InstanceLoader,
};
use genmap::{GenMap, Handle};
use std::collections::HashSet;
use std::ffi::CString;
use std::sync::MutexGuard;
use std::sync::{Arc, Mutex};

/// The TensorSludge engine
pub struct TensorSludge {
    command_pool: vk::CommandPool,
    passes: GenMap<Pass>,
    matrices: GenMap<Matrix>,
    core: SharedCore,
    sigmoid: Sigmoid,
    matrix_multiply: MatrixMultiply,
    queue: vk::Queue,
}

pub struct Core {
    pub allocator: Mutex<Allocator>,
    pub device: DeviceLoader,
    pub instance: InstanceLoader,
    _entry: DefaultEntryLoader,
}

pub type SharedCore = Arc<Core>;

impl TensorSludge {
    /// Create a new TensorSludge instance
    pub fn new() -> Result<Self> {
        let entry = EntryLoader::new()?;

        // Instance
        let name = CString::new("TensorSludge")?;
        let app_info = vk::ApplicationInfoBuilder::new()
            .application_name(&name)
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(&name)
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));

        // Instance and device layers and extensions
        let mut instance_layers = Vec::new();
        let mut instance_extensions = Vec::new();
        let mut device_layers = Vec::new();
        let device_extensions = Vec::new();

        // Vulkan layers and extensions
        if cfg!(debug_assertions) {
            const LAYER_KHRONOS_VALIDATION: *const i8 = cstr!("VK_LAYER_KHRONOS_validation");
            instance_extensions
                .push(erupt::extensions::ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
            instance_layers.push(LAYER_KHRONOS_VALIDATION);
            device_layers.push(LAYER_KHRONOS_VALIDATION);
        }

        // Instance creation
        let create_info = vk::InstanceCreateInfoBuilder::new()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&instance_layers);

        let instance = InstanceLoader::new(&entry, &create_info, None)?;

        // Hardware selection
        let (queue_family_index, physical_device) = select_device(&instance)?;

        // Create logical device and queues
        let create_info = [vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0])];

        let physical_device_features = vk::PhysicalDeviceFeaturesBuilder::new();
        let create_info = vk::DeviceCreateInfoBuilder::new()
            .queue_create_infos(&create_info)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&device_extensions)
            .enabled_layer_names(&device_layers);

        let device = DeviceLoader::new(&instance, physical_device, &create_info, None)?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0, None) };

        // Allocator
        let allocator =
            Allocator::new(&instance, physical_device, AllocatorCreateInfo::default()).result()?;

        // Create command buffer
        // Command pool:
        let create_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::empty());
        let command_pool =
            unsafe { device.create_command_pool(&create_info, None, None) }.result()?;

        let core = Arc::new(Core {
            allocator: Mutex::new(allocator),
            device,
            instance,
            _entry: entry,
        });

        let sigmoid = Sigmoid::new(core.clone())?;
        let matrix_multiply = MatrixMultiply::new(core.clone())?;

        Ok(Self {
            command_pool,
            matrix_multiply,
            matrices: GenMap::with_capacity(10),
            passes: GenMap::with_capacity(10),
            sigmoid,
            queue,
            core,
        })
    }

    /// Create a new matrix with the specified dimensions
    pub fn matrix(&mut self, rows: usize, cols: usize) -> Result<crate::Matrix> {
        Ok(crate::Matrix(self.matrices.insert(Matrix::new(
            rows,
            cols,
            self.core.clone(),
        )?)))
    }

    fn get_matrix_mut<'a>(&'a mut self, matrix: crate::Matrix) -> Result<&'a mut Matrix> {
        self.matrices
            .get_mut(matrix.0)
            .context("Matrix was deleted")
    }

    fn get_matrix<'a>(&'a self, matrix: crate::Matrix) -> Result<&'a Matrix> {
        self.matrices.get(matrix.0).context("Matrix was deleted")
    }

    /// Write data to a matrix in row-major order
    pub fn write(&mut self, matrix: crate::Matrix, data: &[f32]) -> Result<()> {
        self.get_matrix_mut(matrix)?.write(data)
    }

    /// Read data from a matrix in row-major order
    pub fn read(&mut self, matrix: crate::Matrix, data: &mut [f32]) -> Result<()> {
        self.get_matrix_mut(matrix)?.read(data)
    }

    /// Create a pass from a sequence of operations
    pub fn create_pass(&mut self, ops: &[crate::Operation]) -> Result<crate::Pass> {
        // Allocate command buffer
        let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer =
            unsafe { self.core.device.allocate_command_buffers(&allocate_info) }.result()?[0];

        // Create dispatch information and required matrix list
        // This must be a seperate loop because the descriptor set update happens in invoke() for
        // each type of operation
        let mut required_mats = Vec::new();
        let mut dispatch_list: Vec<(Box<dyn Invocation>, Vec<BufferAction>)> = Vec::new();
        for op in ops {
            match op {
                Operation::Sigmoid(mat) => {
                    required_mats.push(mat.0);
                    let invocation = self
                        .sigmoid
                        .invoke(self.matrices.get(mat.0).context("Matrix was deleted")?)?;
                    let actions = vec![BufferAction {
                        matrix: mat.0,
                        read: true,
                        write: true,
                    }];
                    dispatch_list.push((Box::new(invocation), actions));
                }
                Operation::MatrixMultiply {
                    left,
                    right,
                    left_transpose,
                    right_transpose,
                    dst,
                } => {
                    required_mats.push(left.0);
                    required_mats.push(right.0);
                    required_mats.push(dst.0);
                    let invocation = self.matrix_multiply.invoke(
                        self.matrices.get(left.0).context("Left mat was deleted")?,
                        *left_transpose,
                        self.matrices
                            .get(right.0)
                            .context("Right mat was deleted")?,
                        *right_transpose,
                        self.matrices
                            .get(dst.0)
                            .context("Destination mat was deleted")?,
                    )?;
                    let actions = vec![
                        BufferAction {
                            matrix: left.0,
                            read: true,
                            write: false,
                        },
                        BufferAction {
                            matrix: right.0,
                            read: true,
                            write: false,
                        },
                        BufferAction {
                            matrix: dst.0,
                            read: false,
                            write: true,
                        },
                    ];
                    dispatch_list.push((Box::new(invocation), actions));
                }
                _ => todo!("Some ops not implemented"),
            }
        }

        // Write command buffer
        let mut dirty_list = HashSet::new(); // Naughy buffers!
        unsafe {
            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            self.core
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .result()?;

            for (invocation, actions) in dispatch_list {
                let mut barriers = Vec::new();
                for action in actions {
                    if (action.read || action.write) && dirty_list.remove(&action.matrix) {
                        let buffer = self.matrices.get(action.matrix).unwrap();
                        let dst_flags = match (action.read, action.write) {
                            (true, false) => vk::AccessFlags::SHADER_READ,
                            (true, true) => {
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
                            }
                            (false, true) => vk::AccessFlags::SHADER_WRITE,
                            (false, false) => unreachable!(),
                        };
                        let buf_mem_barrier = vk::BufferMemoryBarrierBuilder::new()
                            .buffer(*buffer.allocation().object())
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(dst_flags)
                            .offset(0)
                            .size(vk::WHOLE_SIZE);

                        barriers.push(buf_mem_barrier);
                    }

                    if action.write {
                        dirty_list.insert(action.matrix);
                    }
                }

                self.core.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    None,
                    &[],
                    &barriers,
                    &[],
                );

                invocation.dispatch(&self.core.device, command_buffer);
            }

            self.core
                .device
                .end_command_buffer(command_buffer)
                .result()?;
        }

        let pass = Pass {
            command_buffer,
            required_mats,
        };

        Ok(crate::Pass(self.passes.insert(pass)))
    }

    /// Run the specified pass on the TensorSludge engine
    pub fn flow(&mut self, pass: crate::Pass) -> Result<()> {
        let pass = self.passes.get(pass.0).context("Pass was deleted")?;
        for matrix in &pass.required_mats {
            if self.matrices.get(*matrix).is_none() {
                bail!("A matrix used in this pass was deleted since its creation");
            }
        }

        unsafe {
            let command_buffers = [pass.command_buffer];
            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&command_buffers);
            self.core
                .device
                .queue_submit(self.queue, &[submit_info], None)
                .result()?;
            self.core.device.queue_wait_idle(self.queue).result()?;
        }

        Ok(())
    }
}

struct Pass {
    command_buffer: vk::CommandBuffer,
    required_mats: Vec<Handle>,
}

pub trait Invocation {
    fn dispatch(&self, device: &DeviceLoader, command_buffer: vk::CommandBuffer);
}

struct BufferAction {
    matrix: Handle,
    read: bool,
    write: bool,
}

fn select_device(instance: &InstanceLoader) -> Result<(u32, vk::PhysicalDevice)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;
    for device in physical_devices {
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(device, None) };
        for (family, properites) in families.iter().enumerate() {
            if properites.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok((family as u32, device));
            }
        }
    }
    Err(format_err!("No suitable device found"))
}

impl Core {
    pub fn allocator(&self) -> Result<MutexGuard<Allocator>> {
        self.allocator
            .lock()
            .map_err(|_| format_err!("Allocator mutex poisoned"))
    }
}

impl Drop for TensorSludge {
    fn drop(&mut self) {}
}
