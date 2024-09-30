use app::{AppConfig, StateApplication};
use winit::event_loop::EventLoop;

mod app;

pub async fn run() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("jsrwtt", log::LevelFilter::Info)
        .filter_module("wgpu_hal::vulkan::instance", log::LevelFilter::Error)
        .init();

    let event_loop = EventLoop::new().unwrap();

    let app_config = AppConfig {
        title: "Jonathan's Sausage Roll With Time Travel".into(),
    };

    let mut window_state = StateApplication::new(app_config);
    event_loop.run_app(&mut window_state).unwrap();
}
