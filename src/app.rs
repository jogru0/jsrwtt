use std::convert::TryInto;

use gui::{resolution::Resolution, Gui, GuiConfig, MaybeGui};
use log::info;
use pollster::FutureExt;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
};
use world::World;

mod gui;
mod world {
    pub(super) struct World {
        _size: u32,
    }
    impl World {
        pub(super) fn new() -> Self {
            Self { _size: 7 }
        }
    }
}

pub(super) struct StateApplication {
    maybe_gui: MaybeGui,
    _world: World,
}

pub(super) struct AppConfig {
    pub(super) title: String,
}

impl StateApplication {
    fn surely_gui(&mut self) -> &mut Gui {
        self.maybe_gui.surely_get()
    }

    async fn gui(&mut self, event_loop: &ActiveEventLoop) -> &mut Gui {
        self.maybe_gui.get_or_initialize(event_loop).await
    }

    pub fn new(app_config: AppConfig) -> Self {
        Self {
            maybe_gui: MaybeGui::Unitialized(GuiConfig {
                title: app_config.title,
            }),
            _world: World::new(),
        }
    }
}

impl ApplicationHandler for StateApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        info!("Application resumed.");
        self.gui(event_loop).block_on();
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let gui = self.surely_gui();

        if let DeviceEvent::MouseMotion { delta } = event {
            if gui.mouse_pressed {
                gui.camera_controller.process_mouse(delta.0, delta.1)
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let gui = self.surely_gui();

        if window_id != gui.window().id() {
            panic!("We currently expect only one window")
        }

        // See if event gets consumed by the gui.
        if gui.input(&event) {
            return;
        }

        match event {
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                gui.resize(physical_size.try_into().unwrap());
            }
            // UPDATED!
            WindowEvent::RedrawRequested => {
                gui.window().request_redraw();
                gui.update(instant::Instant::now());
                match gui.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        gui.resize(Resolution::new(gui.config.width, gui.config.height).unwrap())
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            _ => {}
        }
    }
}
