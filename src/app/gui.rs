use std::{
    convert::TryInto,
    f32::consts::PI,
    iter,
    sync::Arc,
    time::{Duration, Instant},
};

use camera::CameraUniform;
use cgmath::Rotation3;
use hdr::HdrPipeline;
use instance::{Instance, INSTANCE_RAW_LAYOUT};
use light_uniform::LightUniform;
use log::{error, info, warn};
use model::{DrawLight, DrawModel, Model, MODEL_VERTEX_LAYOUT};
use resolution::Resolution;
use texture::Texture;
#[cfg(feature = "debug")]
use wgpu::TextureFormat;
use wgpu::{
    util::DeviceExt, Adapter, Backends, BindGroup, BindGroupLayout, Buffer, CommandEncoder, Device,
    DeviceType, Queue, RenderPipeline, Surface, SurfaceConfiguration,
};
use winit::{
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::PhysicalKey,
    window::Window,
};

const NUM_INSTANCES_PER_ROW: u32 = 10;

mod camera;
mod hdr;
mod instance;
mod model;
mod resources;
mod texture;

pub(super) mod resolution;

#[cfg(feature = "debug")]
mod debug;

pub(super) struct GuiConfig {
    pub(super) title: String,
}

pub(super) enum MaybeGui {
    Unitialized(GuiConfig),
    Initialized(Gui),
}

impl MaybeGui {
    pub(super) fn surely_get(&mut self) -> &mut Gui {
        let MaybeGui::Initialized(gui) = self else {
            panic!("expected gui to be already initialized")
        };
        gui
    }

    pub(super) async fn get_or_initialize(&mut self, event_loop: &ActiveEventLoop) -> &mut Gui {
        match self {
            MaybeGui::Unitialized(gui_config) => {
                let gui = Gui::new(event_loop, gui_config).await.unwrap();
                *self = MaybeGui::Initialized(gui);

                let gui = self.surely_get();
                //TODO: No title on Linux?
                info!("created gui for '{}'", gui.window().title());
                gui
            }
            MaybeGui::Initialized(state) => state,
        }
    }
}

mod light_uniform;

pub(super) struct Gui {
    window_arc: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub(super) config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    obj_model: model::Model,
    camera: camera::Camera,
    projection: camera::Projection,
    pub(super) camera_controller: camera::CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
    pub(super) mouse_pressed: bool,
    hdr: hdr::HdrPipeline,
    environment_bind_group: wgpu::BindGroup,
    sky_pipeline: wgpu::RenderPipeline,
    #[cfg(feature = "debug")]
    debug: debug::Debug,
    last_render_time: Instant,
}

fn create_winit_related_fields(
    event_loop: &ActiveEventLoop,
    title: &str,
) -> (Arc<Window>, Resolution) {
    let window = Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_title(title))
            .unwrap(),
    );
    let resolution = window.inner_size().try_into().unwrap();
    info!("created window with title '{title}' and resolution {resolution}",);
    (window, resolution)
}

fn create_surface_and_select_best_adapter(window: Arc<Window>) -> (Surface<'static>, Adapter) {
    // The wgpu_instance is used to get an adapter and a surface.
    let wgpu_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let surface = wgpu_instance.create_surface(window).unwrap();

    let mut adaters = wgpu_instance.enumerate_adapters(Backends::all());
    adaters.retain(|adapter| adapter.is_surface_supported(&surface));

    if adaters.is_empty() {
        error!("found no adapter compatible with the surface");
        panic!()
    }

    let mut adapters_and_infos: Vec<_> = adaters
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            (adapter, info)
        })
        .collect();
    adapters_and_infos.sort_by_key(|(_, info)| match info.device_type {
        wgpu::DeviceType::Other => 4,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::VirtualGpu => 3,
        wgpu::DeviceType::Cpu => 2,
    });

    let mut number_of_adapters_for_discrete_gpus = 0;
    for (_, info) in &adapters_and_infos {
        if info.device_type != DeviceType::DiscreteGpu {
            break;
        }
        number_of_adapters_for_discrete_gpus += 1;
    }

    if number_of_adapters_for_discrete_gpus > 1 {
        warn!("found multiple adapters for discrete GPUs, will choose arbitrarily:");
        for (i, (_, info)) in adapters_and_infos[..number_of_adapters_for_discrete_gpus]
            .iter()
            .enumerate()
        {
            warn!(
                "\tPossibility {}: backend '{}' for GPU '{}'",
                i + 1,
                info.backend,
                info.name
            );
        }
    } else if number_of_adapters_for_discrete_gpus == 0 {
        warn!(
            "found no adapters for discrete GPUs, have to fall back to device type '{:?}'",
            adapters_and_infos[0].1.device_type
        );
    }

    // Handle to our graphics card.
    let (adapter, info) = adapters_and_infos.swap_remove(0);
    info!("using backend '{}' for GPU '{}'", info.backend, info.name);
    (surface, adapter)
}

async fn create_wgpu_related_fields(
    window: Arc<Window>,
    resolution: Resolution,
) -> (Surface<'static>, Device, Queue, SurfaceConfiguration) {
    let required_limits = wgpu::Limits::default();

    let (surface, adapter) = create_surface_and_select_best_adapter(window);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: Default::default(),
            },
            None, // Trace path
        )
        .await
        .unwrap();

    let surface_caps = surface.get_capabilities(&adapter);
    // Shader code in this tutorial assumes an Srgb surface texture. Using a different
    // one will result all the colors comming out darker. If you want to support non
    // Srgb surfaces, you'll need to account for that when drawing to the frame.
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: resolution.width(),
        height: resolution.height(),
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![surface_format.add_srgb_suffix()],
        desired_maximum_frame_latency: 2,
    };

    info!("initial configuration for {}", resolution);
    surface.configure(&device, &config);

    (surface, device, queue, config)
}

fn create_texture_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // normal map
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("texture_bind_group_layout"),
    })
}

fn create_camera_stuff(
    config: &SurfaceConfiguration,
    device: &Device,
) -> (
    camera::Camera,
    camera::Projection,
    camera::CameraController,
    wgpu::Buffer,
    wgpu::BindGroup,
    camera::CameraUniform,
    wgpu::BindGroupLayout,
) {
    let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
    let projection =
        camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
    let camera_controller = camera::CameraController::new(4.0, 0.4);

    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera, &projection);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
        label: Some("camera_bind_group"),
    });

    (
        camera,
        projection,
        camera_controller,
        camera_buffer,
        camera_bind_group,
        camera_uniform,
        camera_bind_group_layout,
    )
}

fn create_instance_stuff(device: &Device) -> (std::vec::Vec<instance::Instance>, wgpu::Buffer) {
    const SPACE_BETWEEN: f32 = 2.0;
    let instances = (0..NUM_INSTANCES_PER_ROW)
        .flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = cgmath::Quaternion::from_axis_angle(
                    cgmath::Vector3::unit_z(),
                    cgmath::Deg(0.0),
                );

                Instance { position, rotation }
            })
        })
        .collect::<Vec<_>>();

    let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(&instance_data),
        usage: wgpu::BufferUsages::VERTEX,
    });

    (instances, instance_buffer)
}

async fn create_obj_model(
    device: &Device,
    queue: &Queue,
    texture_bind_group_layout: &BindGroupLayout,
) -> Model {
    resources::load_model("cube.obj", device, queue, texture_bind_group_layout)
        .await
        .unwrap()
}

fn create_light_stuff(device: &Device) -> (LightUniform, Buffer, BindGroup, BindGroupLayout) {
    let light_uniform = LightUniform {
        position: [2.0, 2.0, 2.0],
        _padding: 0,
        color: [1.0, 1.0, 1.0],
        _padding2: 0,
    };

    let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Light VB"),
        contents: bytemuck::cast_slice(&[light_uniform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let light_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
        });

    let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &light_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: light_buffer.as_entire_binding(),
        }],
        label: None,
    });

    (
        light_uniform,
        light_buffer,
        light_bind_group,
        light_bind_group_layout,
    )
}

fn create_depth_texture(device: &Device, config: &SurfaceConfiguration) -> Texture {
    texture::Texture::create_depth_texture(device, config, "depth_texture")
}

async fn create_hdr_stuff(
    device: &Device,
    config: &SurfaceConfiguration,
    queue: &Queue,
) -> Result<(HdrPipeline, BindGroupLayout, BindGroup), anyhow::Error> {
    let hdr = hdr::HdrPipeline::new(device, config);

    let hdr_loader = resources::HdrLoader::new(device);
    let sky_bytes = resources::load_binary("pure-sky.hdr").await?;
    let sky_texture = hdr_loader.create_texture_from_equirectangular_bytes(
        device,
        queue,
        &sky_bytes,
        1080,
        Some("Sky Texture"),
    )?;

    let environment_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("environment_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });

    let environment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("environment_bind_group"),
        layout: &environment_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(sky_texture.view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sky_texture.sampler()),
            },
        ],
    });

    Ok((hdr, environment_layout, environment_bind_group))
}

#[cfg(feature = "debug")]
fn create_debug_stuff(
    device: &Device,
    camera_bind_group_layout: &BindGroupLayout,
    surface_format: TextureFormat,
) -> debug::Debug {
    debug::Debug::new(device, camera_bind_group_layout, surface_format)
}

fn create_render_pipelines(
    device: &Device,
    hdr: &HdrPipeline,
    texture_bind_group_layout: &BindGroupLayout,
    camera_bind_group_layout: &BindGroupLayout,
    light_bind_group_layout: &BindGroupLayout,
    environment_layout: &BindGroupLayout,
) -> (RenderPipeline, RenderPipeline, RenderPipeline) {
    let render_pipeline = {
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    texture_bind_group_layout,
                    camera_bind_group_layout,
                    light_bind_group_layout,
                    environment_layout,
                ],
                push_constant_ranges: &[],
            });

        let shader = wgpu::include_wgsl!("shader.wgsl");
        create_render_pipeline(
            device,
            &render_pipeline_layout,
            hdr.format(),
            Some(texture::Texture::DEPTH_FORMAT),
            &[MODEL_VERTEX_LAYOUT, INSTANCE_RAW_LAYOUT],
            wgpu::PrimitiveTopology::TriangleList,
            shader,
        )
    };

    let light_render_pipeline = {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, light_bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = wgpu::include_wgsl!("light.wgsl");
        create_render_pipeline(
            device,
            &layout,
            hdr.format(),
            Some(texture::Texture::DEPTH_FORMAT),
            &[MODEL_VERTEX_LAYOUT],
            wgpu::PrimitiveTopology::TriangleList,
            shader,
        )
    };

    let sky_pipeline = {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, environment_layout],
            push_constant_ranges: &[],
        });
        let shader = wgpu::include_wgsl!("sky.wgsl");
        create_render_pipeline(
            device,
            &layout,
            hdr.format(),
            Some(texture::Texture::DEPTH_FORMAT),
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            shader,
        )
    };

    (render_pipeline, light_render_pipeline, sky_pipeline)
}

impl Gui {
    pub(super) async fn new(
        event_loop: &ActiveEventLoop,
        gui_config: &GuiConfig,
    ) -> anyhow::Result<Self> {
        info!("creating gui for '{}'", gui_config.title);

        let (window, resolution) = create_winit_related_fields(event_loop, &gui_config.title);

        let (surface, device, queue, config) =
            create_wgpu_related_fields(window.clone(), resolution).await;

        let texture_bind_group_layout = create_texture_bind_group_layout(&device);

        let (
            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            camera_bind_group_layout,
        ) = create_camera_stuff(&config, &device);

        let (instances, instance_buffer) = create_instance_stuff(&device);

        let obj_model = create_obj_model(&device, &queue, &texture_bind_group_layout).await;

        let (light_uniform, light_buffer, light_bind_group, light_bind_group_layout) =
            create_light_stuff(&device);

        let depth_texture = create_depth_texture(&device, &config);

        let (hdr, environment_layout, environment_bind_group) =
            create_hdr_stuff(&device, &config, &queue).await.unwrap();

        let (render_pipeline, light_render_pipeline, sky_pipeline) = create_render_pipelines(
            &device,
            &hdr,
            &texture_bind_group_layout,
            &camera_bind_group_layout,
            &light_bind_group_layout,
            &environment_layout,
        );

        #[cfg(feature = "debug")]
        let debug = create_debug_stuff(&device, &camera_bind_group_layout, config.format);

        Ok(Self {
            window_arc: window,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            obj_model,
            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            instances,
            instance_buffer,
            depth_texture,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            mouse_pressed: false,
            hdr,
            environment_bind_group,
            sky_pipeline,
            last_render_time: Instant::now(),
            #[cfg(feature = "debug")]
            debug,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window_arc
    }

    pub(super) fn resize(&mut self, new_resolution: Resolution) {
        if new_resolution.width() == self.config.width
            && new_resolution.height() == self.config.height
        {
            info!("skipping resizing to the same size ({})", new_resolution);
            return;
        }

        info!("resizing to {}", new_resolution);

        self.projection.resize(new_resolution);
        self.hdr.resize(&self.device, new_resolution);
        self.config.width = new_resolution.width();
        self.config.height = new_resolution.height();
        self.surface.configure(&self.device, &self.config);
        self.depth_texture =
            texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }

    pub(super) fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => self.camera_controller.process_keyboard(*key, *state),
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            _ => false,
        }
    }

    pub(super) fn update(&mut self, now: Instant) {
        let dt = now - self.last_render_time;

        const MINIMUM_EXPECTED_FPS: u32 = 30;

        if Duration::from_secs(1) / MINIMUM_EXPECTED_FPS < dt {
            warn!("updating gui after {dt:?}");
        }

        self.last_render_time = now;

        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update the light
        let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        self.light_uniform.position = (cgmath::Quaternion::from_axis_angle(
            (0.0, 1.0, 0.0).into(),
            cgmath::Deg(PI * dt.as_secs_f32()),
        ) * old_position)
            .into();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    fn do_render_pass(&self, encoder: &mut CommandEncoder) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.hdr.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_pipeline(&self.light_render_pipeline);
        render_pass.draw_light_model(
            &self.obj_model,
            &self.camera_bind_group,
            &self.light_bind_group,
        );

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.draw_model_instanced(
            &self.obj_model,
            0..self.instances.len() as u32,
            &self.camera_bind_group,
            &self.light_bind_group,
            &self.environment_bind_group,
        );

        render_pass.set_pipeline(&self.sky_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.environment_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    pub(super) fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.add_srgb_suffix()),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // The main render pass, I think.
        self.do_render_pass(&mut encoder);

        // Apply tonemapping
        self.hdr.process(&mut encoder, &view);

        #[cfg(feature = "debug")]
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Debug"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.debug.draw_axis(&mut pass, &self.camera_bind_group);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    topology: wgpu::PrimitiveTopology,
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    info!("create render pipeline '{:?}'", shader.label);

    let module = device.create_shader_module(shader);

    //How to move vertices on the surface.
    let vertex_state = wgpu::VertexState {
        module: &module,
        entry_point: "vs_main",
        buffers: vertex_layouts,
        compilation_options: Default::default(),
    };

    //How to store color data to the surface.
    let fragment_state = wgpu::FragmentState {
        module: &module,
        entry_point: "fs_main",
        targets: &[Some(wgpu::ColorTargetState {
            format: color_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        compilation_options: Default::default(),
    };

    //How to interpret vertices as corners of triangles.
    let primitive_state = wgpu::PrimitiveState {
        topology,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
        polygon_mode: wgpu::PolygonMode::Fill,
        // Requires Features::DEPTH_CLIP_CONTROL
        unclipped_depth: false,
        // Requires Features::CONSERVATIVE_RASTERIZATION
        conservative: false,
    };

    let maybe_depth_stencil_state = depth_format.map(|format| wgpu::DepthStencilState {
        format,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::LessEqual, // UDPATED!
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });

    let multisample_state = wgpu::MultisampleState {
        count: 1,
        mask: !0,
        alpha_to_coverage_enabled: false,
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{:?}", module)),
        layout: Some(layout),
        vertex: vertex_state,
        fragment: Some(fragment_state),
        primitive: primitive_state,
        depth_stencil: maybe_depth_stencil_state,
        multisample: multisample_state,
        multiview: None,
        cache: None,
    })
}
