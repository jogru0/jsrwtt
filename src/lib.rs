use app::{AppConfig, StateApplication};
use winit::event_loop::EventLoop;

mod app;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Could't initialize logger");
        } else {
            env_logger::Builder::new()
                .filter_level(log::LevelFilter::Warn)
                .filter_module("jsrwtt", log::LevelFilter::Info)
                .filter_module("wgpu_hal::vulkan::instance", log::LevelFilter::Error)
                .init();
        }
    }

    let event_loop = EventLoop::new().unwrap();

    let app_config = AppConfig {
        title: "Jonathan's Sausage Roll With Time Travel".into(),
    };

    let mut window_state = StateApplication::new(app_config);
    event_loop.run_app(&mut window_state).unwrap();
}
