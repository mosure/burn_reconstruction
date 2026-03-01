use std::marker::PhantomData;
use std::path::Path;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy_tasks::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy_window::FileDragAndDrop;
#[cfg(not(target_arch = "wasm32"))]
use bevy_winit::{EventLoopProxy, EventLoopProxyWrapper, WakeUp};
use crossbeam_channel::{bounded, Receiver, Sender};

#[cfg(target_arch = "wasm32")]
use js_sys::Uint8Array;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{closure::Closure, JsCast};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{spawn_local, JsFuture};
#[cfg(target_arch = "wasm32")]
use web_sys::{Document, DragEvent};

use super::FileDialogPlugin;
#[cfg(not(target_arch = "wasm32"))]
use super::WakeUpOnDrop;

/// Marker trait saying that data can be loaded from dropped files.
pub trait DropFileContents: Send + Sync + 'static {}

impl<T> DropFileContents for T where T: Send + Sync + 'static {}

/// Event that gets sent when file contents get dropped into the Bevy window.
#[derive(Message)]
pub struct DialogFileDropped<T: DropFileContents> {
    /// Name of dropped file.
    pub file_name: String,

    /// Byte contents of dropped file.
    pub contents: Vec<u8>,

    /// Path to dropped file.
    ///
    /// Does not exist in wasm, you can use this on native platforms only.
    #[cfg(not(target_arch = "wasm32"))]
    pub path: std::path::PathBuf,

    marker: PhantomData<T>,
}

impl<T: DropFileContents> DialogFileDropped<T> {
    /// Returns the native path when available.
    pub fn path(&self) -> Option<&Path> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Some(self.path.as_path())
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
    }
}

/// Event that gets sent when no dropped files were accepted.
#[derive(Message)]
pub struct DialogFileDropCanceled<T: DropFileContents>(PhantomData<T>);

impl<T: DropFileContents> Default for DialogFileDropCanceled<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Resource)]
struct DropStreamReceiver<T: DropFileContents>(Receiver<DialogFileDropped<T>>);

#[derive(Resource)]
struct DropStreamSender<T: DropFileContents>(Sender<DialogFileDropped<T>>);

fn handle_drop_stream<T: DropFileContents>(
    receiver: Res<DropStreamReceiver<T>>,
    mut ev_done: MessageWriter<DialogFileDropped<T>>,
) {
    for event in receiver.0.try_iter() {
        ev_done.write(event);
    }
}

impl FileDialogPlugin {
    /// Allow loading files via drag-and-drop on both native and web.
    ///
    /// For every dropped file, [`DialogFileDropped<T>`] is emitted.
    pub fn with_drop_file<T: DropFileContents>(mut self) -> Self {
        self.0.push(Box::new(|app| {
            let (tx, rx) = bounded::<DialogFileDropped<T>>(64);
            app.insert_resource(DropStreamSender(tx));
            app.insert_resource(DropStreamReceiver(rx));
            app.add_message::<DialogFileDropped<T>>();
            app.add_message::<DialogFileDropCanceled<T>>();
            app.add_systems(First, handle_drop_stream::<T>);
            #[cfg(not(target_arch = "wasm32"))]
            app.add_systems(Update, handle_native_file_drop::<T>);
            #[cfg(target_arch = "wasm32")]
            install_web_drop_listeners::<T>(app);
        }));
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn handle_native_file_drop<T: DropFileContents>(
    mut events: MessageReader<FileDragAndDrop>,
    sender: Res<DropStreamSender<T>>,
    event_loop_proxy: Option<Res<EventLoopProxyWrapper<WakeUp>>>,
) {
    let event_loop_proxy = event_loop_proxy
        .as_ref()
        .map(|proxy| EventLoopProxy::clone(&**proxy));
    for event in events.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = event else {
            continue;
        };
        let path = path_buf.clone();
        let sender = sender.0.clone();
        let event_loop_proxy = event_loop_proxy.clone();
        AsyncComputeTaskPool::get()
            .spawn(async move {
                let _wake_up = event_loop_proxy.as_ref().map(WakeUpOnDrop);
                match std::fs::read(&path) {
                    Ok(contents) => {
                        let file_name = path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("file")
                            .to_string();
                        let _ = sender.send(DialogFileDropped {
                            file_name,
                            contents,
                            path,
                            marker: PhantomData,
                        });
                    }
                    Err(err) => {
                        log::warn!("Failed to read dropped file {}: {err}", path.display());
                    }
                }
            })
            .detach();
    }
}

#[cfg(target_arch = "wasm32")]
struct WebDropListeners<T: DropFileContents> {
    document: Document,
    dragover: Closure<dyn FnMut(DragEvent)>,
    drop: Closure<dyn FnMut(DragEvent)>,
    marker: PhantomData<T>,
}

#[cfg(target_arch = "wasm32")]
impl<T: DropFileContents> Drop for WebDropListeners<T> {
    fn drop(&mut self) {
        let _ = self.document.remove_event_listener_with_callback(
            "dragover",
            self.dragover.as_ref().unchecked_ref(),
        );
        let _ = self
            .document
            .remove_event_listener_with_callback("drop", self.drop.as_ref().unchecked_ref());
    }
}

#[cfg(target_arch = "wasm32")]
fn install_web_drop_listeners<T: DropFileContents>(app: &mut App) {
    let sender = match app.world().get_resource::<DropStreamSender<T>>() {
        Some(sender) => sender.0.clone(),
        None => {
            log::warn!("File drop listeners could not start: missing drop sender resource.");
            return;
        }
    };
    let Some(window) = web_sys::window() else {
        log::warn!("File drop listeners could not start: web window is unavailable.");
        return;
    };
    let Some(document) = window.document() else {
        log::warn!("File drop listeners could not start: web document is unavailable.");
        return;
    };

    let dragover = Closure::<dyn FnMut(DragEvent)>::new(move |event: DragEvent| {
        event.prevent_default();
    });
    if let Err(err) =
        document.add_event_listener_with_callback("dragover", dragover.as_ref().unchecked_ref())
    {
        log::warn!("Failed to register web dragover listener: {err:?}");
        return;
    }

    let drop_sender = sender.clone();
    let drop = Closure::<dyn FnMut(DragEvent)>::new(move |event: DragEvent| {
        event.prevent_default();
        event.stop_propagation();
        let Some(data_transfer) = event.data_transfer() else {
            return;
        };
        let Some(files) = data_transfer.files() else {
            return;
        };
        for index in 0..files.length() {
            let Some(file) = files.get(index) else {
                continue;
            };
            let sender = drop_sender.clone();
            spawn_local(async move {
                let file_name = file.name();
                match read_web_file_bytes(file).await {
                    Ok(contents) => {
                        let _ = sender.send(DialogFileDropped {
                            file_name,
                            contents,
                            marker: PhantomData,
                        });
                    }
                    Err(err) => {
                        log::warn!("Failed to read dropped browser file '{file_name}': {err}");
                    }
                }
            });
        }
    });
    if let Err(err) =
        document.add_event_listener_with_callback("drop", drop.as_ref().unchecked_ref())
    {
        log::warn!("Failed to register web drop listener: {err:?}");
        return;
    }

    app.insert_non_send_resource(WebDropListeners::<T> {
        document,
        dragover,
        drop,
        marker: PhantomData,
    });
}

#[cfg(target_arch = "wasm32")]
async fn read_web_file_bytes(file: web_sys::File) -> Result<Vec<u8>, String> {
    let promise = file.array_buffer();
    let buffer = JsFuture::from(promise)
        .await
        .map_err(|err| format!("array_buffer await failed: {err:?}"))?;
    let bytes = Uint8Array::new(&buffer);
    let mut out = vec![0u8; bytes.length() as usize];
    bytes.copy_to(&mut out);
    Ok(out)
}
