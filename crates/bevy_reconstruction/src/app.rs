use std::cmp::Ordering;
#[cfg(not(target_arch = "wasm32"))]
use std::ffi::OsString;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
use std::path::Path;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

#[cfg(not(target_arch = "wasm32"))]
use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::image::{CompressedImageFormats, Image, ImageSampler, ImageType};
use bevy::log::{Level, LogPlugin};
use bevy::prelude::*;
#[cfg(target_arch = "wasm32")]
use bevy::tasks::{futures_lite::future, AsyncComputeTaskPool, Task};
#[cfg(target_arch = "wasm32")]
use bevy::window::RequestRedraw;
use bevy::window::{Window, WindowPlugin};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
#[cfg(target_arch = "wasm32")]
use burn_reconstruction::backend::{default_device, ensure_wasm_wgpu_runtime};
use burn_reconstruction::canonicalize_gaussian_transform_cv;
#[cfg(all(test, not(target_arch = "wasm32")))]
use burn_reconstruction::cv_xyzw_to_canonical_wxyz;
#[cfg(not(target_arch = "wasm32"))]
use burn_reconstruction::pack_gaussian_rows_full;
#[cfg(target_arch = "wasm32")]
use burn_reconstruction::pack_gaussian_rows_full_async;
#[cfg(test)]
use burn_reconstruction::sanitize_scale_for_viewer;
use burn_reconstruction::PipelineInputImage;
use burn_reconstruction::{GlbExportOptions, GlbSortMode, PipelineGaussians};
#[cfg(target_arch = "wasm32")]
use burn_reconstruction::{ImageToGaussianPipeline, PipelineConfig};
#[cfg(not(target_arch = "wasm32"))]
use burn_reconstruction::{
    ImageToGaussianPipeline, PipelineConfig, PipelineQuality, PipelineWeights, YonoWeightFormat,
    YonoWeightPrecision,
};

use bevy_gaussian_splatting::{
    gaussian::formats::planar_3d::{Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle},
    material::spherical_harmonics::SphericalHarmonicCoefficients,
    sort::SortMode,
    CloudSettings, GaussianCamera, GaussianSplattingPlugin,
};
#[cfg(not(target_arch = "wasm32"))]
use crossbeam_channel::{Receiver, Sender};
#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Reflect};
#[cfg(target_arch = "wasm32")]
use serde::Deserialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

use crate::bevy_file_dialog::prelude::{
    DialogFileDropped, DialogFileLoaded, FileDialogExt, FileDialogPlugin,
};

const APP_TITLE: &str = "bevy_reconstruction";
const MAX_STAGING_LABEL_IMAGES: usize = 10;
const MENU_HEIGHT: f32 = 44.0;
const PANEL_WIDTH: f32 = 360.0;
const MENU_BG: Color = Color::srgb(0.08, 0.09, 0.11);
const PANEL_BG: Color = Color::srgb(0.06, 0.07, 0.09);
const PANEL_BORDER: Color = Color::srgb(0.13, 0.14, 0.18);
const BUTTON_BG: Color = Color::srgb(0.1, 0.11, 0.14);
const BUTTON_BG_HOVER: Color = Color::srgb(0.13, 0.14, 0.18);
const BUTTON_BG_PRESSED: Color = Color::srgb(0.17, 0.19, 0.24);
const BUTTON_BORDER: Color = Color::srgb(0.28, 0.3, 0.35);
const BUTTON_BORDER_HOVER: Color = Color::srgb(0.36, 0.4, 0.5);
const BUTTON_BORDER_PRESSED: Color = Color::srgb(0.46, 0.52, 0.64);
const BUTTON_BG_DISABLED: Color = Color::srgb(0.08, 0.09, 0.11);
const BUTTON_BORDER_DISABLED: Color = Color::srgb(0.2, 0.22, 0.26);
const BUTTON_TEXT: Color = Color::srgb(0.86, 0.88, 0.94);
const BUTTON_TEXT_DISABLED: Color = Color::srgb(0.45, 0.48, 0.56);
const BUTTON_OPEN_BG: Color = Color::srgb(0.14, 0.18, 0.3);
const BUTTON_OPEN_BG_HOVER: Color = Color::srgb(0.18, 0.24, 0.39);
const BUTTON_OPEN_BG_PRESSED: Color = Color::srgb(0.22, 0.29, 0.47);
const BUTTON_OPEN_BORDER: Color = Color::srgb(0.32, 0.4, 0.62);
const BUTTON_OPEN_BORDER_HOVER: Color = Color::srgb(0.43, 0.53, 0.8);
const BUTTON_OPEN_BORDER_PRESSED: Color = Color::srgb(0.55, 0.66, 0.95);
const STATUS_BADGE_BG: Color = Color::srgb(0.1, 0.11, 0.14);
const STATUS_BADGE_BORDER: Color = Color::srgb(0.24, 0.27, 0.33);
// Match YoNoSplat evaluation prune setting (+encoder.gaussian_adapter_cfg.prune_probs=[0.005]).
const VIEWER_OPACITY_THRESHOLD: f32 = 0.005;
const SCENE_RADIUS_QUANTILE: f32 = 0.9;
const SCENE_RADIUS_FLOOR: f32 = 0.25;
const SCENE_RADIUS_PADDING_SCALE: f32 = 2.0;
const FRUSTUM_DEFAULT_NEAR: f32 = 0.06;
const FRUSTUM_DEFAULT_FAR: f32 = 0.36;
const FRUSTUM_DEFAULT_NEAR_HALF_BASE: f32 = 0.03;
const FRUSTUM_DEFAULT_FAR_HALF_BASE: f32 = 0.16;
const FRUSTUM_MIN_FAR: f32 = 0.03;
const FRUSTUM_MAX_FAR: f32 = 0.6;
const FRUSTUM_NEIGHBOR_FAR_SCALE: f32 = 0.35;
const FRUSTUM_SCENE_RADIUS_FAR_CAP_SCALE: f32 = 0.2;
const FRUSTUM_NEAR_TO_FAR_RATIO: f32 = FRUSTUM_DEFAULT_NEAR / FRUSTUM_DEFAULT_FAR;
const FRUSTUM_NEAR_HALF_RATIO: f32 = FRUSTUM_DEFAULT_NEAR_HALF_BASE / FRUSTUM_DEFAULT_NEAR;
const FRUSTUM_FAR_HALF_RATIO: f32 = FRUSTUM_DEFAULT_FAR_HALF_BASE / FRUSTUM_DEFAULT_FAR;
#[cfg(target_arch = "wasm32")]
const WASM_DEFAULT_MODEL_BASE_URL: &str = "https://aberration.technology/model";
#[cfg(target_arch = "wasm32")]
const WASM_DEFAULT_MODEL_REMOTE_ROOT: &str = "yono";
#[cfg(target_arch = "wasm32")]
const WASM_BACKBONE_BURNPACK_FILE: &str = "yono_backbone_f16.bpk";
#[cfg(target_arch = "wasm32")]
const WASM_HEAD_BURNPACK_FILE: &str = "yono_head_f16.bpk";

#[derive(Debug)]
struct ImageSelectionDialog;

#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct LoadedImage {
    name: String,
    bytes: Vec<u8>,
    thumbnail: Option<Handle<Image>>,
}

#[derive(Resource, Default)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct UiState {
    status: String,
    selected_images: Vec<LoadedImage>,
    selected_image_index: Option<usize>,
    pending_camera_focus_index: Option<usize>,
    staging_revision: u64,
    queued_run: bool,
    active_cloud: Option<Entity>,
    active_scene_bounds: Option<SceneBounds>,
}

#[derive(Debug, Clone, Copy)]
struct SceneBounds {
    center: Vec3,
    radius: f32,
}

#[derive(Resource, Debug, Clone, Default)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct LaunchOptions {
    startup_images: Vec<PathBuf>,
    auto_run: bool,
    output_glb: Option<PathBuf>,
    exit_after_run: bool,
    #[cfg(not(target_arch = "wasm32"))]
    print_help: bool,
    #[cfg(not(target_arch = "wasm32"))]
    print_rev: bool,
}

#[derive(Resource, Default)]
struct FrustumDebug {
    poses_world_from_camera: Vec<Mat4>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource)]
struct NativeInferenceWorker {
    command_tx: Sender<NativeWorkerCommand>,
    event_rx: Receiver<NativeWorkerEvent>,
    run_in_flight: bool,
}

#[cfg(not(target_arch = "wasm32"))]
enum NativeWorkerCommand {
    Run(NativeRunRequest),
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
struct NativeRunRequest {
    images: Vec<LoadedImage>,
    output_glb: Option<PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
enum NativeWorkerEvent {
    Status(String),
    Completed(NativeRunResult),
    Failed(String),
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeRunResult {
    cloud: PlanarGaussian3d,
    selected_gaussians: usize,
    total_gaussians: usize,
    timings: burn_reconstruction::ForwardTimings,
    camera_poses_world_from_camera: Vec<Mat4>,
    scene_bounds: Option<SceneBounds>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct WasmInferenceRuntime {
    pipeline: Option<ImageToGaussianPipeline>,
    model_load_task: Option<Task<Result<ImageToGaussianPipeline, String>>>,
    run_task: Option<Task<(ImageToGaussianPipeline, Result<WasmRunResult, String>)>>,
    pending_run: Option<WasmRunRequest>,
    run_requested: bool,
    progress: WasmProgress,
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct WasmStartupImages {
    checked: bool,
    task: Option<WasmStartupImageTask>,
    auto_run: bool,
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone)]
struct WasmRunRequest {
    images: Vec<LoadedImage>,
}

#[cfg(target_arch = "wasm32")]
struct WasmRunResult {
    cloud: PlanarGaussian3d,
    selected_gaussians: usize,
    total_gaussians: usize,
    timings: burn_reconstruction::ForwardTimings,
    camera_poses_world_from_camera: Vec<Mat4>,
    scene_bounds: Option<SceneBounds>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Deserialize)]
struct WasmPartsManifest {
    parts: Vec<WasmPartEntry>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Deserialize)]
struct WasmPartEntry {
    path: String,
}

#[cfg(target_arch = "wasm32")]
type WasmProgress = Rc<RefCell<Option<String>>>;
#[cfg(target_arch = "wasm32")]
type WasmStartupImageTask = Task<Result<Vec<(String, Vec<u8>)>, String>>;

#[derive(Component)]
struct SelectImagesButton;

#[derive(Component)]
struct RunInferenceButton;

#[derive(Component)]
struct ClearImagesButton;

#[derive(Component)]
struct MainOrbitCamera;

#[derive(Component)]
struct StatusLabel;

#[derive(Component)]
struct SelectionLabel;

#[derive(Component)]
struct StagingListRoot;

#[derive(Component, Clone, Copy)]
struct StagingImageButton {
    index: usize,
}

#[derive(Component)]
struct QueueBadge;

#[derive(Component)]
struct QueueBadgeDot;

#[derive(Component)]
struct QueueBadgeText;

#[derive(Component)]
struct ButtonLabel;

#[derive(Component, Clone, Copy)]
struct ControlButton(ControlButtonKind);

#[derive(Clone, Copy)]
enum ControlButtonKind {
    Primary,
    Accent,
    Danger,
}

type ControlButtonVisualQuery<'a> = (
    &'a Interaction,
    &'a ControlButton,
    Option<&'a RunInferenceButton>,
    Option<&'a ClearImagesButton>,
    &'a Children,
    &'a mut BackgroundColor,
    &'a mut BorderColor,
);

type StagingButtonVisualQuery<'a> = (
    &'a Interaction,
    &'a StagingImageButton,
    &'a mut BackgroundColor,
    &'a mut BorderColor,
);

type StagingButtonInteractionQuery<'a> = (&'a Interaction, &'a StagingImageButton);

type QueueBadgeVisualQuery<'a> = (&'a mut BackgroundColor, &'a mut BorderColor);
type QueueBadgeVisualFilter = (With<QueueBadge>, Without<QueueBadgeDot>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StatusBadgeTone {
    Idle,
    Pending,
    Busy,
    Success,
    Error,
}

pub fn run() {
    burn_reconstruction::setup_hooks();
    let launch_options = launch_options_from_env();
    #[cfg(not(target_arch = "wasm32"))]
    if launch_options.print_help {
        print_native_help();
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if launch_options.print_rev {
        println!("{}", burn_reconstruction::git_revision_short());
        return;
    }

    log::info!(
        "bevy_reconstruction: starting app ({})",
        burn_reconstruction::app_banner(APP_TITLE)
    );

    let mut app = App::new();
    app.insert_resource(UiState {
        status: "select at least two images of one static scene".to_string(),
        ..UiState::default()
    });
    app.insert_resource(launch_options);
    app.insert_resource(FrustumDebug::default());
    #[cfg(not(target_arch = "wasm32"))]
    app.insert_resource(spawn_native_inference_worker());
    #[cfg(target_arch = "wasm32")]
    app.insert_non_send_resource(WasmInferenceRuntime::default());
    #[cfg(target_arch = "wasm32")]
    app.insert_non_send_resource(WasmStartupImages::default());

    #[cfg(target_arch = "wasm32")]
    let primary_window = Window {
        title: APP_TITLE.to_string(),
        fit_canvas_to_parent: true,
        ..Window::default()
    };
    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Window {
        title: APP_TITLE.to_string(),
        ..Window::default()
    };

    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(primary_window),
                ..WindowPlugin::default()
            })
            .set(LogPlugin {
                level: Level::INFO,
                filter: "wgpu=error,naga=warn,bevy_reconstruction=info,burn_reconstruction=info"
                    .to_string(),
                ..LogPlugin::default()
            }),
    );
    app.add_plugins(InfiniteGridPlugin);

    app.add_plugins(GaussianSplattingPlugin);
    app.add_plugins((
        PanOrbitCameraPlugin,
        FileDialogPlugin::new()
            .with_load_file::<ImageSelectionDialog>()
            .with_drop_file::<ImageSelectionDialog>(),
    ));

    app.add_systems(
        Startup,
        (setup_scene, setup_ui, ingest_startup_images).chain(),
    );
    app.add_systems(
        Update,
        (
            handle_ui_buttons,
            handle_staging_image_buttons,
            handle_file_dialog_loads,
            handle_dropped_files,
            rebuild_staging_list,
            update_status_label,
            update_selection_label,
            update_status_badge,
            update_control_button_visuals,
            update_staging_button_visuals,
            focus_camera_on_selected_view,
            draw_camera_frustums,
        ),
    );
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(
        Update,
        (
            queue_native_inference_if_requested,
            poll_native_worker_events,
        )
            .chain(),
    );
    #[cfg(target_arch = "wasm32")]
    app.add_systems(
        Update,
        (
            poll_wasm_startup_images,
            queue_wasm_inference_if_requested,
            poll_wasm_inference_worker,
            keep_wasm_runtime_redrawing,
        )
            .chain(),
    );

    app.run();
}

fn setup_scene(mut commands: Commands) {
    log::info!("bevy_reconstruction: setting up scene");
    commands.spawn((
        Camera2d,
        IsDefaultUiCamera,
        Camera {
            order: 10,
            clear_color: ClearColorConfig::None,
            ..Camera::default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        Tonemapping::None,
        Transform::from_xyz(0.0, 1.25, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
        GaussianCamera { warmup: true },
        MainOrbitCamera,
        PanOrbitCamera {
            allow_upside_down: true,
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            ..PanOrbitCamera::default()
        },
    ));

    commands.spawn(InfiniteGridBundle::default());

    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadows_enabled: false,
            ..DirectionalLight::default()
        },
        Transform::from_xyz(2.0, 4.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 400.0,
        ..AmbientLight::default()
    });
}

fn setup_ui(mut commands: Commands) {
    log::info!("bevy_reconstruction: setting up UI");
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                flex_direction: FlexDirection::Column,
                ..Node::default()
            },
            ZIndex(100),
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.0)),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(MENU_HEIGHT),
                    padding: UiRect::horizontal(Val::Px(14.0)),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..Node::default()
                },
                BackgroundColor(MENU_BG),
            ))
            .with_children(|menu| {
                menu.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..Node::default()
                })
                .with_children(|left| {
                    left.spawn((
                        Text::new("bevy_reconstruction"),
                        TextFont::from_font_size(16.0),
                        TextColor(Color::srgb(0.92, 0.94, 0.98)),
                    ));

                    left.spawn((
                        Button,
                        SelectImagesButton,
                        ControlButton(ControlButtonKind::Accent),
                        Node {
                            padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            ..Node::default()
                        },
                        BorderColor::all(BUTTON_OPEN_BORDER),
                        BackgroundColor(BUTTON_OPEN_BG),
                        BorderRadius::all(Val::Px(7.0)),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new("open images"),
                            TextFont::from_font_size(13.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });

                    left.spawn((
                        Button,
                        RunInferenceButton,
                        ControlButton(ControlButtonKind::Primary),
                        Node {
                            padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            ..Node::default()
                        },
                        BorderColor::all(BUTTON_BORDER),
                        BackgroundColor(BUTTON_BG),
                        BorderRadius::all(Val::Px(7.0)),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new("run"),
                            TextFont::from_font_size(13.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });

                    left.spawn((
                        Button,
                        ClearImagesButton,
                        ControlButton(ControlButtonKind::Danger),
                        Node {
                            padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                            border: UiRect::all(Val::Px(1.0)),
                            ..Node::default()
                        },
                        BorderColor::all(BUTTON_BORDER),
                        BackgroundColor(BUTTON_BG),
                        BorderRadius::all(Val::Px(7.0)),
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new("clear"),
                            TextFont::from_font_size(13.0),
                            TextColor(BUTTON_TEXT),
                            ButtonLabel,
                        ));
                    });
                });

                menu.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..Node::default()
                })
                .with_children(|right| {
                    right
                        .spawn((
                            QueueBadge,
                            Node {
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                column_gap: Val::Px(7.0),
                                padding: UiRect::axes(Val::Px(10.0), Val::Px(5.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                ..Node::default()
                            },
                            BackgroundColor(STATUS_BADGE_BG),
                            BorderColor::all(STATUS_BADGE_BORDER),
                            BorderRadius::all(Val::Px(999.0)),
                        ))
                        .with_children(|badge| {
                            badge.spawn((
                                QueueBadgeDot,
                                Node {
                                    width: Val::Px(8.0),
                                    height: Val::Px(8.0),
                                    ..Node::default()
                                },
                                BackgroundColor(Color::srgb(0.52, 0.56, 0.64)),
                                BorderRadius::all(Val::Px(999.0)),
                            ));
                            badge.spawn((
                                Text::new("idle"),
                                TextFont::from_font_size(13.0),
                                TextColor(Color::srgb(0.72, 0.76, 0.84)),
                                QueueBadgeText,
                            ));
                        });
                });
            });

            root.spawn(Node {
                width: Val::Percent(100.0),
                flex_grow: 1.0,
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::FlexStart,
                ..Node::default()
            })
            .with_children(|body| {
                body.spawn((
                    Node {
                        width: Val::Px(PANEL_WIDTH),
                        height: Val::Percent(100.0),
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Stretch,
                        row_gap: Val::Px(10.0),
                        padding: UiRect::all(Val::Px(14.0)),
                        border: UiRect::right(Val::Px(1.0)),
                        ..Node::default()
                    },
                    BackgroundColor(PANEL_BG),
                    BorderColor::all(PANEL_BORDER),
                ))
                .with_children(|panel| {
                    panel.spawn((
                        Text::new("image staging"),
                        TextFont::from_font_size(15.0),
                        TextColor(Color::srgb(0.9, 0.92, 0.97)),
                    ));

                    panel.spawn((
                        Text::new("drop files into the viewport or click open images."),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.68, 0.72, 0.8)),
                    ));

                    panel.spawn((
                        Text::new("selected images: 0"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.85, 0.88, 0.92)),
                        SelectionLabel,
                    ));

                    panel
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                flex_grow: 1.0,
                                min_height: Val::Px(220.0),
                                flex_direction: FlexDirection::Column,
                                padding: UiRect::all(Val::Px(8.0)),
                                border: UiRect::all(Val::Px(1.0)),
                                overflow: Overflow::clip_y(),
                                ..Node::default()
                            },
                            BorderColor::all(Color::srgb(0.24, 0.28, 0.35)),
                            BackgroundColor(Color::srgb(0.1, 0.12, 0.16)),
                            StagingListRoot,
                        ))
                        .with_children(|staging| {
                            staging.spawn((
                                Text::new("no images selected"),
                                TextFont::from_font_size(12.0),
                                TextColor(Color::srgb(0.85, 0.88, 0.92)),
                            ));
                        });

                    panel.spawn((
                        Text::new("status"),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgb(0.82, 0.86, 0.94)),
                    ));

                    panel.spawn((
                        Text::new("select at least two images of one static scene"),
                        TextFont::from_font_size(12.0),
                        TextColor(Color::srgb(0.78, 0.82, 0.92)),
                        StatusLabel,
                    ));

                    panel
                        .spawn(Node {
                            width: Val::Percent(100.0),
                            margin: UiRect::top(Val::Auto),
                            ..Node::default()
                        })
                        .with_children(|footer| {
                            footer.spawn((
                                Text::new(format!("build {}", burn_reconstruction::build_label())),
                                TextFont::from_font_size(11.0),
                                TextColor(Color::srgb(0.6, 0.64, 0.72)),
                            ));
                        });
                });
            });
        });
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_native_inference_worker() -> NativeInferenceWorker {
    let (command_tx, command_rx) = crossbeam_channel::unbounded::<NativeWorkerCommand>();
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<NativeWorkerEvent>();
    thread::spawn(move || native_worker_loop(command_rx, event_tx));
    NativeInferenceWorker {
        command_tx,
        event_rx,
        run_in_flight: false,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn send_worker_status(event_tx: &Sender<NativeWorkerEvent>, status: impl Into<String>) {
    let _ = event_tx.send(NativeWorkerEvent::Status(status.into()));
}

#[cfg(not(target_arch = "wasm32"))]
fn native_load_pipeline(
    event_tx: &Sender<NativeWorkerEvent>,
) -> Result<ImageToGaussianPipeline, String> {
    let start = Instant::now();
    log::info!("bevy_reconstruction: resolving model weights...");
    send_worker_status(event_tx, "resolving model files (cache + remote)...");

    let cfg = PipelineConfig {
        quality: PipelineQuality::Balanced,
        ..PipelineConfig::default()
    };

    let progress_tx = event_tx.clone();
    let weights = PipelineWeights::resolve_or_bootstrap_yono_with_precision_and_progress(
        YonoWeightFormat::Burnpack,
        YonoWeightPrecision::F16,
        move |status| {
            let _ = progress_tx.send(NativeWorkerEvent::Status(format!(
                "loading model weights: {status}"
            )));
        },
    )
    .map_err(|err| format!("failed to resolve model weights: {err}"))?;
    send_worker_status(event_tx, "initializing yono pipeline modules...");
    log::info!(
        "bevy_reconstruction: model weights resolved in {:.2}s; initializing pipeline...",
        start.elapsed().as_secs_f64()
    );

    let progress_tx = event_tx.clone();
    let (pipeline, _load_report) =
        ImageToGaussianPipeline::load_default_with_progress(cfg, weights, move |status| {
            let _ = progress_tx.send(NativeWorkerEvent::Status(format!(
                "loading model weights: {status}"
            )));
        })
        .map_err(|err| format!("failed to initialize inference pipeline: {err}"))?;
    send_worker_status(event_tx, "yono modules initialized; preparing inference...");
    log::info!(
        "bevy_reconstruction: model initialized in {:.2}s total",
        start.elapsed().as_secs_f64()
    );
    Ok(pipeline)
}

#[cfg(not(target_arch = "wasm32"))]
fn native_run_request(
    pipeline: &ImageToGaussianPipeline,
    request: &NativeRunRequest,
) -> Result<NativeRunResult, String> {
    let inference_start = Instant::now();
    log::info!(
        "bevy_reconstruction: starting inference on {} images...",
        request.images.len()
    );

    let inputs = request
        .images
        .iter()
        .map(|image| PipelineInputImage {
            name: image.name.as_str(),
            bytes: image.bytes.as_slice(),
        })
        .collect::<Vec<_>>();

    let run_output = pipeline
        .run_image_bytes_timed_with_cameras(inputs.as_slice(), true)
        .map_err(|err| format!("inference failed: {err}"))?;

    log::info!(
        "bevy_reconstruction: inference finished in {:.2}s (image_load {:.2} ms, backbone {:.2} ms, head {:.2} ms, total {:.2} ms).",
        inference_start.elapsed().as_secs_f64(),
        run_output.timings.image_load.as_secs_f64() * 1000.0,
        run_output.timings.backbone.as_secs_f64() * 1000.0,
        run_output.timings.head.as_secs_f64() * 1000.0,
        run_output.timings.total.as_secs_f64() * 1000.0
    );

    let [batch, gaussians_per_batch, _] = run_output.gaussians.means.shape().dims::<3>();
    let total_gaussians = batch * gaussians_per_batch;
    let export_options = GlbExportOptions {
        max_gaussians: total_gaussians.max(1),
        opacity_threshold: VIEWER_OPACITY_THRESHOLD,
        sort_mode: GlbSortMode::Index,
    };

    if let Some(path) = request.output_glb.as_ref() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create GLB output directory: {err}"))?;
            }
        }
        let report = pipeline
            .save_glb(path, &run_output.gaussians, &export_options)
            .map_err(|err| format!("failed to write GLB: {err}"))?;
        log::info!(
            "bevy_reconstruction: wrote GLB {} ({} gaussians, select {:.2} ms, write {:.2} ms)",
            path.display(),
            report.selected_gaussians,
            report.select_millis,
            report.write_millis
        );
    }

    let cloud_build = build_planar_cloud_from_pipeline(&run_output.gaussians, &export_options)
        .map_err(|err| format!("failed to prepare gaussians for viewer: {err}"))?;
    log::info!(
        "bevy_reconstruction: selected {} / {} gaussians for cloud spawn (opacity_threshold {:.4}, sort_mode {:?}).",
        cloud_build.selected_gaussians,
        total_gaussians,
        export_options.opacity_threshold,
        export_options.sort_mode
    );

    let camera_poses_world_from_camera = run_output
        .camera_poses
        .iter()
        .copied()
        .map(row_major_pose_to_mat4)
        .collect::<Vec<_>>();

    Ok(NativeRunResult {
        cloud: cloud_build.cloud,
        selected_gaussians: cloud_build.selected_gaussians,
        total_gaussians,
        timings: run_output.timings,
        camera_poses_world_from_camera,
        scene_bounds: cloud_build.scene_bounds,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn native_worker_loop(
    command_rx: Receiver<NativeWorkerCommand>,
    event_tx: Sender<NativeWorkerEvent>,
) {
    let mut pipeline: Option<ImageToGaussianPipeline> = None;

    while let Ok(command) = command_rx.recv() {
        match command {
            NativeWorkerCommand::Run(request) => {
                if pipeline.is_none() {
                    send_worker_status(&event_tx, "loading model weights...");
                    match native_load_pipeline(&event_tx) {
                        Ok(loaded) => {
                            pipeline = Some(loaded);
                        }
                        Err(err) => {
                            let _ = event_tx.send(NativeWorkerEvent::Failed(err));
                            continue;
                        }
                    }
                }

                send_worker_status(
                    &event_tx,
                    format!("running inference on {} images...", request.images.len()),
                );

                let result = pipeline
                    .as_ref()
                    .ok_or_else(|| "pipeline not available".to_string())
                    .and_then(|pipeline| native_run_request(pipeline, &request));

                match result {
                    Ok(done) => {
                        let _ = event_tx.send(NativeWorkerEvent::Completed(done));
                    }
                    Err(err) => {
                        let _ = event_tx.send(NativeWorkerEvent::Failed(err));
                    }
                }
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn handle_ui_buttons(
    mut commands: Commands,
    select_interactions: Query<&Interaction, (With<Button>, With<SelectImagesButton>)>,
    run_interactions: Query<&Interaction, (With<Button>, With<RunInferenceButton>)>,
    clear_interactions: Query<&Interaction, (With<Button>, With<ClearImagesButton>)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut ui: ResMut<UiState>,
    mut frustums: ResMut<FrustumDebug>,
) {
    let just_pressed_left = mouse_buttons.just_pressed(MouseButton::Left);
    let just_released_left = mouse_buttons.just_released(MouseButton::Left);

    for interaction in select_interactions.iter() {
        if !ui_button_activated(*interaction, just_pressed_left, just_released_left) {
            continue;
        }
        log::info!("bevy_reconstruction: opening image selection dialog");
        commands
            .dialog()
            .set_title("Select source images")
            .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
            .load_multiple_files::<ImageSelectionDialog>();
        ui.status = "waiting for selected images (new files are appended)...".to_string();
    }

    for interaction in run_interactions.iter() {
        if !ui_button_activated(*interaction, just_pressed_left, just_released_left) {
            continue;
        }
        ui.queued_run = true;
        log::info!("bevy_reconstruction: run requested via UI.");
    }

    for interaction in clear_interactions.iter() {
        if !ui_button_activated(*interaction, just_pressed_left, just_released_left) {
            continue;
        }
        let had_staged_images = !ui.selected_images.is_empty();
        let had_frustums = !frustums.poses_world_from_camera.is_empty();
        let cleared_cloud = ui.active_cloud.take();

        if let Some(entity) = cleared_cloud {
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
            }
        }

        ui.selected_images.clear();
        ui.selected_image_index = None;
        ui.pending_camera_focus_index = None;
        ui.staging_revision = ui.staging_revision.wrapping_add(1);
        ui.queued_run = false;
        ui.active_scene_bounds = None;
        frustums.poses_world_from_camera.clear();

        if had_staged_images || had_frustums || cleared_cloud.is_some() {
            ui.status = "cleared staged images and reconstruction output".to_string();
            log::info!(
                "bevy_reconstruction: cleared staged images and reconstruction output via UI."
            );
        } else {
            ui.status = "nothing to clear".to_string();
        }
    }
}

fn handle_staging_image_buttons(
    interactions: Query<StagingButtonInteractionQuery<'_>, With<Button>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut ui: ResMut<UiState>,
) {
    let just_pressed_left = mouse_buttons.just_pressed(MouseButton::Left);
    let just_released_left = mouse_buttons.just_released(MouseButton::Left);
    for (interaction, button) in interactions.iter() {
        if !ui_button_activated(*interaction, just_pressed_left, just_released_left) {
            continue;
        }
        if button.index >= ui.selected_images.len() {
            continue;
        }
        ui.selected_image_index = Some(button.index);
        ui.pending_camera_focus_index = Some(button.index);
        ui.staging_revision = ui.staging_revision.wrapping_add(1);
        if let Some(image) = ui.selected_images.get(button.index) {
            ui.status = format!("selected view {}: {}", button.index + 1, image.name);
        }
    }
}

fn ui_button_activated(
    interaction: Interaction,
    _just_pressed_left: bool,
    just_released_left: bool,
) -> bool {
    interaction == Interaction::Hovered && just_released_left
}

fn staging_item_palette(selected: bool, interaction: Interaction) -> (Color, Color) {
    match (selected, interaction) {
        (true, Interaction::Pressed) => {
            (Color::srgb(0.24, 0.31, 0.48), Color::srgb(0.6, 0.72, 0.98))
        }
        (true, Interaction::Hovered) => {
            (Color::srgb(0.2, 0.27, 0.42), Color::srgb(0.5, 0.62, 0.88))
        }
        (true, Interaction::None) => (Color::srgb(0.16, 0.22, 0.34), Color::srgb(0.4, 0.52, 0.76)),
        (false, Interaction::Pressed) => {
            (Color::srgb(0.17, 0.2, 0.28), Color::srgb(0.44, 0.51, 0.64))
        }
        (false, Interaction::Hovered) => {
            (Color::srgb(0.13, 0.15, 0.2), Color::srgb(0.3, 0.35, 0.44))
        }
        (false, Interaction::None) => (Color::srgb(0.1, 0.11, 0.14), Color::srgb(0.2, 0.22, 0.26)),
    }
}

fn rebuild_staging_list(
    mut commands: Commands,
    ui: Res<UiState>,
    roots: Query<Entity, With<StagingListRoot>>,
    mut last_revision: Local<Option<u64>>,
) {
    if last_revision.as_ref() == Some(&ui.staging_revision) {
        return;
    }
    *last_revision = Some(ui.staging_revision);

    let Ok(root) = roots.single() else {
        return;
    };

    let mut root_commands = commands.entity(root);
    root_commands.despawn_related::<Children>();
    root_commands.with_children(|staging| {
        if ui.selected_images.is_empty() {
            staging.spawn((
                Text::new("no images selected"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.85, 0.88, 0.92)),
            ));
            return;
        }

        for (index, image) in ui
            .selected_images
            .iter()
            .take(MAX_STAGING_LABEL_IMAGES)
            .enumerate()
        {
            let selected = ui.selected_image_index == Some(index);
            let (bg, border) = staging_item_palette(selected, Interaction::None);
            staging
                .spawn((
                    Button,
                    StagingImageButton { index },
                    Node {
                        width: Val::Percent(100.0),
                        min_height: Val::Px(74.0),
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(8.0),
                        padding: UiRect::all(Val::Px(6.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        ..Node::default()
                    },
                    BorderRadius::all(Val::Px(6.0)),
                    BackgroundColor(bg),
                    BorderColor::all(border),
                ))
                .with_children(|row| {
                    row.spawn((
                        Node {
                            width: Val::Px(60.0),
                            height: Val::Px(60.0),
                            border: UiRect::all(Val::Px(1.0)),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..Node::default()
                        },
                        BorderRadius::all(Val::Px(4.0)),
                        BorderColor::all(Color::srgb(0.26, 0.31, 0.4)),
                        BackgroundColor(Color::srgb(0.04, 0.05, 0.08)),
                    ))
                    .with_children(|thumb| {
                        if let Some(handle) = image.thumbnail.clone() {
                            thumb.spawn((
                                ImageNode::new(handle),
                                Node {
                                    width: Val::Percent(100.0),
                                    height: Val::Percent(100.0),
                                    ..Node::default()
                                },
                            ));
                        } else {
                            thumb.spawn((
                                Text::new("preview"),
                                TextFont::from_font_size(10.0),
                                TextColor(Color::srgb(0.72, 0.76, 0.84)),
                            ));
                        }
                    });

                    row.spawn(Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(3.0),
                        flex_grow: 1.0,
                        ..Node::default()
                    })
                    .with_children(|meta| {
                        meta.spawn((
                            Text::new(image.name.clone()),
                            TextFont::from_font_size(12.0),
                            TextColor(Color::srgb(0.9, 0.92, 0.97)),
                        ));
                        meta.spawn((
                            Text::new(format!(
                                "{} KB",
                                ((image.bytes.len() as f32) / 1024.0).round() as usize
                            )),
                            TextFont::from_font_size(11.0),
                            TextColor(Color::srgb(0.68, 0.72, 0.8)),
                        ));
                    });
                });
        }

        if ui.selected_images.len() > MAX_STAGING_LABEL_IMAGES {
            staging.spawn((
                Text::new(format!(
                    "... and {} more",
                    ui.selected_images.len() - MAX_STAGING_LABEL_IMAGES
                )),
                TextFont::from_font_size(11.0),
                TextColor(Color::srgb(0.72, 0.76, 0.84)),
            ));
        }
    });
}

fn update_staging_button_visuals(
    ui: Res<UiState>,
    mut buttons: Query<StagingButtonVisualQuery<'_>, With<Button>>,
) {
    for (interaction, staging, mut bg, mut border) in &mut buttons {
        let selected = ui.selected_image_index == Some(staging.index);
        let (background, border_color) = staging_item_palette(selected, *interaction);
        bg.0 = background;
        *border = BorderColor::all(border_color);
    }
}

fn focus_camera_on_selected_view(
    mut ui: ResMut<UiState>,
    frustums: Res<FrustumDebug>,
    mut cameras: Query<&mut PanOrbitCamera, With<MainOrbitCamera>>,
) {
    let Some(index) = ui.pending_camera_focus_index else {
        return;
    };

    let Some(world_from_camera) = frustums.poses_world_from_camera.get(index).copied() else {
        return;
    };

    let Ok(mut orbit) = cameras.single_mut() else {
        return;
    };

    let fallback_radius = orbit.target_radius.max(0.35);
    let scene_bounds = ui.active_scene_bounds;
    let Some((focus, yaw, pitch, radius)) =
        orbit_targets_for_view(world_from_camera, fallback_radius, scene_bounds)
    else {
        ui.pending_camera_focus_index = None;
        return;
    };

    orbit.target_focus = focus;
    orbit.target_yaw = yaw;
    orbit.target_pitch = pitch;
    orbit.target_radius = radius;
    orbit.force_update = true;
    ui.pending_camera_focus_index = None;
}

fn orbit_targets_for_view(
    world_from_camera: Mat4,
    fallback_radius: f32,
    scene_bounds: Option<SceneBounds>,
) -> Option<(Vec3, f32, f32, f32)> {
    let position = world_from_camera.transform_point3(Vec3::ZERO);
    let forward = world_from_camera
        .transform_vector3(Vec3::new(0.0, 0.0, -1.0))
        .normalize_or_zero();
    if !forward.is_finite() || forward.length_squared() <= f32::EPSILON {
        return None;
    }
    let fallback_radius = fallback_radius.max(0.35);
    let axis_distance = scene_bounds
        .and_then(|bounds| {
            optical_axis_focus_distance(position, forward, bounds.center, bounds.radius)
        })
        .unwrap_or(fallback_radius);
    let focus = position + forward * axis_distance;
    let delta = position - focus;
    let radius = delta.length().max(0.35);
    let yaw = delta.x.atan2(delta.z);
    let pitch = (delta.y / radius).clamp(-1.0, 1.0).asin();
    Some((focus, yaw, pitch, radius))
}

fn optical_axis_focus_distance(
    position: Vec3,
    forward: Vec3,
    scene_center: Vec3,
    scene_radius: f32,
) -> Option<f32> {
    if !position.is_finite() || !forward.is_finite() || !scene_center.is_finite() {
        return None;
    }

    let radius = scene_radius.max(SCENE_RADIUS_FLOOR);
    if !radius.is_finite() {
        return None;
    }

    let min_distance = (radius * 0.2).max(0.2);
    let max_distance = (radius * 8.0).max(min_distance + 0.5);

    // Try to place focus near the front intersection of the scene sphere along the camera ray.
    let oc = position - scene_center;
    let projection = oc.dot(forward);
    let c = oc.length_squared() - radius * radius;
    let discriminant = projection * projection - c;
    if discriminant.is_finite() && discriminant >= 0.0 {
        let root = discriminant.sqrt();
        let t_enter = -projection - root;
        let t_exit = -projection + root;
        if t_enter.is_finite() && t_enter > min_distance {
            return Some((t_enter + radius * 0.25).clamp(min_distance, max_distance));
        }
        if t_exit.is_finite() && t_exit > min_distance {
            return Some((t_exit * 0.6).clamp(min_distance, max_distance));
        }
    }

    // Fallback to the projected scene center along optical axis.
    let projected_center_t = (scene_center - position).dot(forward);
    if !projected_center_t.is_finite() {
        return None;
    }
    let near_center_t = projected_center_t - radius * 0.35;
    Some(near_center_t.clamp(min_distance, max_distance))
}

fn handle_file_dialog_loads(
    mut loaded_events: MessageReader<DialogFileLoaded<ImageSelectionDialog>>,
    mut ui: ResMut<UiState>,
    mut thumbnails: ResMut<Assets<Image>>,
    mut frustums: ResMut<FrustumDebug>,
) {
    let mut loaded_count = 0usize;
    for event in loaded_events.read() {
        let thumbnail =
            decode_thumbnail_handle(thumbnails.as_mut(), &event.file_name, &event.contents);
        add_loaded_image(
            &mut ui,
            event.file_name.clone(),
            event.contents.clone(),
            thumbnail,
        );
        loaded_count += 1;
    }

    if loaded_count > 0 {
        frustums.poses_world_from_camera.clear();
        ui.pending_camera_focus_index = None;
        log::info!("bevy_reconstruction: loaded {loaded_count} images from file dialog");
        set_loaded_images_status(&mut ui, "loaded");
    }
}

fn handle_dropped_files(
    mut dropped_events: MessageReader<DialogFileDropped<ImageSelectionDialog>>,
    mut ui: ResMut<UiState>,
    mut thumbnails: ResMut<Assets<Image>>,
    mut frustums: ResMut<FrustumDebug>,
) {
    let mut loaded_count = 0usize;
    for event in dropped_events.read() {
        let thumbnail =
            decode_thumbnail_handle(thumbnails.as_mut(), &event.file_name, &event.contents);
        add_loaded_image(
            &mut ui,
            event.file_name.clone(),
            event.contents.clone(),
            thumbnail,
        );
        loaded_count += 1;
    }

    if loaded_count > 0 {
        frustums.poses_world_from_camera.clear();
        ui.pending_camera_focus_index = None;
        log::info!("bevy_reconstruction: loaded {loaded_count} dropped images");
        set_loaded_images_status(&mut ui, "added");
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn queue_native_inference_if_requested(
    mut ui: ResMut<UiState>,
    mut worker: ResMut<NativeInferenceWorker>,
    launch: Res<LaunchOptions>,
    mut exit: MessageWriter<AppExit>,
) {
    if !ui.queued_run {
        return;
    }
    ui.queued_run = false;

    if worker.run_in_flight {
        ui.status = "a reconstruction is already in progress".to_string();
        return;
    }

    if ui.selected_images.len() < 2 {
        ui.status = "select at least 2 images of the same scene".to_string();
        if launch.exit_after_run {
            exit.write(AppExit::error());
        }
        return;
    }

    let request = NativeRunRequest {
        images: ui.selected_images.clone(),
        output_glb: launch.output_glb.clone(),
    };
    if worker
        .command_tx
        .send(NativeWorkerCommand::Run(request))
        .is_err()
    {
        ui.status = "failed to queue reconstruction worker job".to_string();
        if launch.exit_after_run {
            exit.write(AppExit::error());
        }
        return;
    }

    worker.run_in_flight = true;
    ui.status = format!(
        "queued reconstruction for {} images...",
        ui.selected_images.len()
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn poll_native_worker_events(
    mut commands: Commands,
    mut ui: ResMut<UiState>,
    mut worker: ResMut<NativeInferenceWorker>,
    launch: Res<LaunchOptions>,
    mut exit: MessageWriter<AppExit>,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut frustums: ResMut<FrustumDebug>,
) {
    while let Ok(event) = worker.event_rx.try_recv() {
        match event {
            NativeWorkerEvent::Status(status) => {
                ui.status = status;
            }
            NativeWorkerEvent::Completed(done) => {
                let handle = clouds.add(done.cloud);

                if let Some(entity) = ui.active_cloud.take() {
                    if let Ok(mut ec) = commands.get_entity(entity) {
                        ec.despawn();
                    }
                }

                let entity = spawn_reconstruction_cloud(&mut commands, handle);
                ui.active_cloud = Some(entity);
                frustums.poses_world_from_camera = done.camera_poses_world_from_camera;
                ui.active_scene_bounds = done.scene_bounds;
                if ui.selected_image_index.is_none() && !ui.selected_images.is_empty() {
                    ui.selected_image_index = Some(0);
                    ui.staging_revision = ui.staging_revision.wrapping_add(1);
                }
                if let Some(index) = ui.selected_image_index {
                    if index < frustums.poses_world_from_camera.len() {
                        ui.pending_camera_focus_index = Some(index);
                    }
                }
                worker.run_in_flight = false;

                ui.status = format_inference_complete_status(
                    &done.timings,
                    done.selected_gaussians,
                    done.total_gaussians,
                    None,
                );
                log::info!(
                    "bevy_reconstruction: spawned cloud entity {:?} with {} gaussians",
                    entity,
                    done.selected_gaussians
                );

                if launch.exit_after_run {
                    exit.write(AppExit::Success);
                }
            }
            NativeWorkerEvent::Failed(message) => {
                ui.status = message;
                worker.run_in_flight = false;
                if launch.exit_after_run {
                    exit.write(AppExit::error());
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn queue_wasm_inference_if_requested(
    mut ui: ResMut<UiState>,
    mut runtime: NonSendMut<WasmInferenceRuntime>,
) {
    if !ui.queued_run {
        return;
    }
    ui.queued_run = false;
    if ui.selected_images.len() < 2 {
        ui.status = "select at least 2 images of the same scene".to_string();
        return;
    }
    runtime.run_requested = true;
    runtime.pending_run = Some(WasmRunRequest {
        images: ui.selected_images.clone(),
    });

    if runtime.model_load_task.is_some() {
        ui.status = "loading model weights; reconstruction queued...".to_string();
        return;
    }
    if runtime.pipeline.is_none() {
        ui.status = "loading model weights; reconstruction queued...".to_string();
        let progress = runtime.progress.clone();
        runtime.model_load_task = Some(
            AsyncComputeTaskPool::get().spawn_local(async { wasm_load_pipeline(progress).await }),
        );
        return;
    }
    if runtime.run_task.is_some() {
        ui.status = "reconstruction already running; queued another run...".to_string();
        return;
    }

    start_wasm_run_if_ready(ui.as_mut(), runtime.as_mut());
}

#[cfg(target_arch = "wasm32")]
fn poll_wasm_startup_images(
    mut ui: ResMut<UiState>,
    mut startup: NonSendMut<WasmStartupImages>,
    mut thumbnails: ResMut<Assets<Image>>,
    mut frustums: ResMut<FrustumDebug>,
) {
    if !startup.checked {
        startup.checked = true;
        let urls = wasm_startup_image_urls();
        if !urls.is_empty() {
            let count = urls.len();
            startup.auto_run = js_window_bool("BEVY_RECONSTRUCTION_AUTO_RUN").unwrap_or(true);
            ui.status = format!("loading {count} startup images from web...");
            startup.task = Some(
                AsyncComputeTaskPool::get()
                    .spawn_local(async move { fetch_startup_images(urls).await }),
            );
        }
    }

    if let Some(mut task) = startup.task.take() {
        if let Some(result) = future::block_on(future::poll_once(&mut task)) {
            match result {
                Ok(images) => {
                    if images.is_empty() {
                        ui.status = "no startup images could be loaded from web".to_string();
                        return;
                    }

                    ui.selected_images.clear();
                    ui.selected_image_index = None;
                    ui.pending_camera_focus_index = None;
                    ui.staging_revision = ui.staging_revision.wrapping_add(1);
                    frustums.poses_world_from_camera.clear();

                    for (name, bytes) in images {
                        let thumbnail = decode_thumbnail_handle(thumbnails.as_mut(), &name, &bytes);
                        add_loaded_image(&mut ui, name, bytes, thumbnail);
                    }
                    set_loaded_images_status(&mut ui, "loaded");

                    if startup.auto_run {
                        ui.queued_run = true;
                        ui.status = format!(
                            "loaded {} startup images. running reconstruction...",
                            ui.selected_images.len()
                        );
                        log::info!(
                            "bevy_reconstruction: wasm auto-run queued from startup images."
                        );
                    }
                }
                Err(message) => {
                    ui.status = message;
                }
            }
        } else {
            startup.task = Some(task);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn poll_wasm_inference_worker(
    mut commands: Commands,
    mut ui: ResMut<UiState>,
    mut runtime: NonSendMut<WasmInferenceRuntime>,
    mut frustums: ResMut<FrustumDebug>,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
) {
    if runtime.model_load_task.is_some() {
        if let Some(progress) = runtime.progress.borrow().clone() {
            ui.status = progress;
        }
    }

    if let Some(mut task) = runtime.model_load_task.take() {
        if let Some(result) = future::block_on(future::poll_once(&mut task)) {
            match result {
                Ok(pipeline) => {
                    runtime.pipeline = Some(pipeline);
                    runtime.progress.borrow_mut().take();
                    ui.status = "model initialized".to_string();
                    start_wasm_run_if_ready(ui.as_mut(), runtime.as_mut());
                }
                Err(message) => {
                    runtime.progress.borrow_mut().take();
                    runtime.pending_run = None;
                    runtime.run_requested = false;
                    ui.status = message;
                }
            }
        } else {
            runtime.model_load_task = Some(task);
        }
    }

    if let Some(mut task) = runtime.run_task.take() {
        if let Some((pipeline, result)) = future::block_on(future::poll_once(&mut task)) {
            runtime.pipeline = Some(pipeline);
            match result {
                Ok(done) => {
                    let handle = clouds.add(done.cloud);
                    if let Some(entity) = ui.active_cloud.take() {
                        if let Ok(mut ec) = commands.get_entity(entity) {
                            ec.despawn();
                        }
                    }
                    let entity = spawn_reconstruction_cloud(&mut commands, handle);
                    ui.active_cloud = Some(entity);
                    let frustum_count = done.camera_poses_world_from_camera.len();
                    frustums.poses_world_from_camera = done.camera_poses_world_from_camera;
                    ui.active_scene_bounds = done.scene_bounds;
                    if ui.selected_image_index.is_none() && !ui.selected_images.is_empty() {
                        ui.selected_image_index = Some(0);
                        ui.staging_revision = ui.staging_revision.wrapping_add(1);
                    }
                    if let Some(index) = ui.selected_image_index {
                        if index < frustums.poses_world_from_camera.len() {
                            ui.pending_camera_focus_index = Some(index);
                        }
                    }
                    ui.status = format_inference_complete_status(
                        &done.timings,
                        done.selected_gaussians,
                        done.total_gaussians,
                        Some(frustum_count),
                    );
                    log::info!(
                        "bevy_reconstruction: spawned wasm cloud entity {:?} with {} gaussians.",
                        entity,
                        done.selected_gaussians
                    );
                    log::info!(
                        "bevy_reconstruction: wasm inference complete ({} / {} gaussians, {} frustums).",
                        done.selected_gaussians,
                        done.total_gaussians,
                        frustum_count
                    );
                }
                Err(message) => {
                    ui.status = message;
                }
            }

            // If user queued another run while one was in-flight, launch it now.
            start_wasm_run_if_ready(ui.as_mut(), runtime.as_mut());
        } else {
            runtime.run_task = Some(task);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn keep_wasm_runtime_redrawing(
    ui: Res<UiState>,
    runtime: NonSend<WasmInferenceRuntime>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    let runtime_busy = ui.queued_run
        || runtime.run_requested
        || runtime.pending_run.is_some()
        || runtime.model_load_task.is_some()
        || runtime.run_task.is_some();
    if !runtime_busy {
        return;
    }
    redraw.write(RequestRedraw);
}

#[cfg(target_arch = "wasm32")]
fn start_wasm_run_if_ready(ui: &mut UiState, runtime: &mut WasmInferenceRuntime) {
    if runtime.model_load_task.is_some() || runtime.run_task.is_some() {
        return;
    }
    if runtime.pending_run.is_none() && runtime.run_requested && ui.selected_images.len() >= 2 {
        runtime.pending_run = Some(WasmRunRequest {
            images: ui.selected_images.clone(),
        });
    }

    let Some(request) = runtime.pending_run.take() else {
        return;
    };
    let Some(pipeline) = runtime.pipeline.take() else {
        runtime.pending_run = Some(request);
        return;
    };
    runtime.run_requested = false;
    let image_count = request.images.len();
    ui.status = format!("running inference on {image_count} images...");
    runtime.run_task = Some(AsyncComputeTaskPool::get().spawn_local(async move {
        let result = wasm_run_request(&pipeline, &request).await;
        (pipeline, result)
    }));
}

#[cfg(target_arch = "wasm32")]
async fn wasm_load_pipeline(progress: WasmProgress) -> Result<ImageToGaussianPipeline, String> {
    let model_root = wasm_model_root_url()?;
    set_wasm_model_progress(
        &progress,
        format!("resolving wasm model files from {model_root}"),
    );
    log::info!(
        "bevy_reconstruction: loading wasm model parts from {}",
        model_root
    );

    let backbone_url = join_url(model_root.as_str(), WASM_BACKBONE_BURNPACK_FILE);
    let head_url = join_url(model_root.as_str(), WASM_HEAD_BURNPACK_FILE);
    let backbone_parts = fetch_parts_bundle(backbone_url.as_str(), "backbone", &progress).await?;
    let head_parts = fetch_parts_bundle(head_url.as_str(), "head", &progress).await?;

    set_wasm_model_progress(&progress, "initializing yono pipeline modules...");
    let cfg = PipelineConfig::default();
    let device = default_device();
    ensure_wasm_wgpu_runtime(&device).await;
    let progress_for_load = progress.clone();
    let (pipeline, load_report) = ImageToGaussianPipeline::load_from_yono_parts_with_progress(
        device,
        cfg,
        backbone_parts.as_slice(),
        head_parts.as_slice(),
        move |status| set_wasm_model_progress(&progress_for_load, status),
    )
    .map_err(|err| format!("failed to initialize wasm inference pipeline: {err}"))?;
    log::info!(
        "bevy_reconstruction: wasm model initialized (backbone applied {}, head applied {}).",
        load_report.backbone.applied,
        load_report.head.applied
    );
    set_wasm_model_progress(&progress, "model initialization complete");
    Ok(pipeline)
}

#[cfg(target_arch = "wasm32")]
async fn wasm_run_request(
    pipeline: &ImageToGaussianPipeline,
    request: &WasmRunRequest,
) -> Result<WasmRunResult, String> {
    let inputs = request
        .images
        .iter()
        .map(|image| PipelineInputImage {
            name: image.name.as_str(),
            bytes: image.bytes.as_slice(),
        })
        .collect::<Vec<_>>();

    let run_output = pipeline
        .run_image_bytes_timed_with_cameras_async(inputs.as_slice(), true)
        .await
        .map_err(|err| format!("inference failed: {err}"))?;

    let [batch, gaussians_per_batch, _] = run_output.gaussians.means.shape().dims::<3>();
    let total_gaussians = batch * gaussians_per_batch;
    let export_options = GlbExportOptions {
        max_gaussians: total_gaussians.max(1),
        opacity_threshold: VIEWER_OPACITY_THRESHOLD,
        sort_mode: GlbSortMode::Index,
    };
    let cloud_build =
        build_planar_cloud_from_pipeline_async(&run_output.gaussians, &export_options).await?;

    let camera_poses_world_from_camera = run_output
        .camera_poses
        .iter()
        .copied()
        .map(row_major_pose_to_mat4)
        .collect::<Vec<_>>();

    Ok(WasmRunResult {
        cloud: cloud_build.cloud,
        selected_gaussians: cloud_build.selected_gaussians,
        total_gaussians,
        timings: run_output.timings,
        camera_poses_world_from_camera,
        scene_bounds: cloud_build.scene_bounds,
    })
}

#[cfg(target_arch = "wasm32")]
fn wasm_model_root_url() -> Result<String, String> {
    let model_base = js_window_string("BURN_RECONSTRUCTION_MODEL_BASE_URL")
        .unwrap_or_else(|| WASM_DEFAULT_MODEL_BASE_URL.to_string());
    let remote_root = js_window_string("BURN_RECONSTRUCTION_YONO_REMOTE_ROOT")
        .unwrap_or_else(|| WASM_DEFAULT_MODEL_REMOTE_ROOT.to_string());

    if remote_root.starts_with("http://") || remote_root.starts_with("https://") {
        return Ok(remote_root);
    }
    Ok(join_url(model_base.as_str(), remote_root.as_str()))
}

#[cfg(target_arch = "wasm32")]
fn wasm_startup_image_urls() -> Vec<String> {
    let query_urls = wasm_query_image_urls();
    if !query_urls.is_empty() {
        return query_urls;
    }

    if let Some(values) = js_window_string_array("BEVY_RECONSTRUCTION_STARTUP_IMAGES") {
        return values
            .into_iter()
            .filter_map(|raw| normalize_wasm_startup_image_url(raw.as_str()))
            .collect();
    }
    js_window_string("BEVY_RECONSTRUCTION_STARTUP_IMAGES")
        .map(|raw| {
            raw.split(',')
                .filter_map(normalize_wasm_startup_image_url)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
fn wasm_query_image_urls() -> Vec<String> {
    let Some(window) = web_sys::window() else {
        return Vec::new();
    };
    let Ok(search) = window.location().search() else {
        return Vec::new();
    };
    if search.trim().is_empty() || search == "?" {
        return Vec::new();
    }
    let Ok(params) = web_sys::UrlSearchParams::new_with_str(search.as_str()) else {
        return Vec::new();
    };

    let mut out = Vec::<String>::new();
    let mut add_value = |raw: String| {
        let Some(value) = normalize_wasm_startup_image_url(raw.as_str()) else {
            return;
        };
        if !out.iter().any(|existing| existing == &value) {
            out.push(value);
        }
    };

    for key in ["image", "img"] {
        let values = params.get_all(key);
        for entry in values.iter() {
            if let Some(value) = entry.as_string() {
                add_value(value);
            }
        }
    }

    let image_lists = params.get_all("images");
    for entry in image_lists.iter() {
        if let Some(value) = entry.as_string() {
            for piece in value.split(',') {
                add_value(piece.to_string());
            }
        }
    }

    out
}

#[cfg(target_arch = "wasm32")]
fn normalize_wasm_startup_image_url(raw: &str) -> Option<String> {
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }

    let is_explicit_path = value.starts_with("http://")
        || value.starts_with("https://")
        || value.starts_with("blob:")
        || value.starts_with("data:")
        || value.starts_with('/')
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with("assets/")
        || value.starts_with("www/");
    if is_explicit_path {
        return Some(value.to_string());
    }

    Some(format!("assets/images/{value}"))
}

#[cfg(target_arch = "wasm32")]
fn js_window_string(key: &str) -> Option<String> {
    let window = web_sys::window()?;
    let window_js = JsValue::from(window);
    let value = Reflect::get(&window_js, &JsValue::from_str(key)).ok()?;
    value
        .as_string()
        .map(|entry| entry.trim().to_string())
        .filter(|entry| !entry.is_empty())
}

#[cfg(target_arch = "wasm32")]
fn js_window_bool(key: &str) -> Option<bool> {
    let window = web_sys::window()?;
    let window_js = JsValue::from(window);
    let value = Reflect::get(&window_js, &JsValue::from_str(key)).ok()?;
    if value.is_null() || value.is_undefined() {
        return None;
    }
    if let Some(boolean) = value.as_bool() {
        return Some(boolean);
    }
    value
        .as_string()
        .map(|entry| entry.trim().to_ascii_lowercase())
        .and_then(|entry| match entry.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

#[cfg(target_arch = "wasm32")]
fn js_window_string_array(key: &str) -> Option<Vec<String>> {
    let window = web_sys::window()?;
    let window_js = JsValue::from(window);
    let value = Reflect::get(&window_js, &JsValue::from_str(key)).ok()?;
    if value.is_null() || value.is_undefined() {
        return None;
    }
    if let Some(single) = value.as_string() {
        return Some(
            single
                .split(',')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(str::to_string)
                .collect(),
        );
    }
    if !value.is_instance_of::<Array>() {
        return None;
    }
    let array = Array::from(&value);
    let mut out = Vec::new();
    for idx in 0..array.length() {
        let item = array.get(idx);
        if let Some(entry) = item.as_string() {
            let entry = entry.trim().to_string();
            if !entry.is_empty() {
                out.push(entry);
            }
        }
    }
    Some(out)
}

#[cfg(target_arch = "wasm32")]
fn join_url(base: &str, child: &str) -> String {
    if child.starts_with("http://") || child.starts_with("https://") {
        return child.to_string();
    }
    let left = base.trim_end_matches('/');
    let right = child.trim_start_matches('/');
    if left.is_empty() {
        return format!("/{right}");
    }
    format!("{left}/{right}")
}

#[cfg(target_arch = "wasm32")]
async fn fetch_startup_images(urls: Vec<String>) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::with_capacity(urls.len());
    for url in urls {
        let bytes = fetch_url_bytes(url.as_str()).await?;
        let name = url
            .rsplit('/')
            .next()
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .unwrap_or("image.png")
            .to_string();
        out.push((name, bytes));
    }
    Ok(out)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_parts_bundle(
    base_burnpack_url: &str,
    component: &str,
    progress: &WasmProgress,
) -> Result<Vec<Vec<u8>>, String> {
    let manifest_url = format!("{base_burnpack_url}.parts.json");
    set_wasm_model_progress(progress, format!("downloading {component} manifest"));
    let manifest_bytes = fetch_url_bytes(manifest_url.as_str()).await?;
    let manifest: WasmPartsManifest = serde_json::from_slice(manifest_bytes.as_slice())
        .map_err(|err| format!("failed to parse parts manifest {manifest_url}: {err}"))?;
    if manifest.parts.is_empty() {
        return Err(format!("parts manifest is empty: {manifest_url}"));
    }

    let mut parts = Vec::with_capacity(manifest.parts.len());
    let total = manifest.parts.len();
    for (index, part) in manifest.parts.into_iter().enumerate() {
        let url = if part.path.starts_with("http://") || part.path.starts_with("https://") {
            part.path
        } else {
            let manifest_parent = manifest_url
                .rsplit_once('/')
                .map(|(parent, _)| parent)
                .unwrap_or(manifest_url.as_str());
            join_url(manifest_parent, part.path.as_str())
        };
        set_wasm_model_progress(
            progress,
            format!("downloading {component} part {}/{}", index + 1, total),
        );
        parts.push(fetch_url_bytes(url.as_str()).await?);
    }
    set_wasm_model_progress(
        progress,
        format!("downloaded {component} parts ({total}/{total})"),
    );
    Ok(parts)
}

#[cfg(target_arch = "wasm32")]
async fn fetch_url_bytes(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or_else(|| "window is unavailable".to_string())?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|err| format!("fetch failed for {url}: {}", js_value_to_string(&err)))?;
    let response: web_sys::Response = response_value
        .dyn_into()
        .map_err(|_| format!("fetch returned non-response value for {url}"))?;
    if !response.ok() {
        return Err(format!("HTTP {} while fetching {url}", response.status()));
    }
    let array_buffer = JsFuture::from(response.array_buffer().map_err(|err| {
        format!(
            "failed to read response bytes for {url}: {}",
            js_value_to_string(&err)
        )
    })?)
    .await
    .map_err(|err| format!("arrayBuffer failed for {url}: {}", js_value_to_string(&err)))?;
    Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
}

#[cfg(target_arch = "wasm32")]
fn js_value_to_string(value: &JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

#[cfg(target_arch = "wasm32")]
fn set_wasm_progress(progress: &WasmProgress, message: impl Into<String>) {
    *progress.borrow_mut() = Some(message.into());
}

#[cfg(target_arch = "wasm32")]
fn set_wasm_model_progress(progress: &WasmProgress, message: impl Into<String>) {
    set_wasm_progress(
        progress,
        format!("loading model weights: {}", message.into()),
    );
}

fn decode_thumbnail_handle(
    thumbnails: &mut Assets<Image>,
    name: &str,
    bytes: &[u8],
) -> Option<Handle<Image>> {
    let extension = Path::new(name)
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "png".to_string());

    let decoded = Image::from_buffer(
        bytes,
        ImageType::Extension(extension.as_str()),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::linear(),
        RenderAssetUsages::RENDER_WORLD,
    )
    .ok()?;

    let mut thumbnail = if let Ok(dynamic) = decoded.clone().try_into_dynamic() {
        let thumbnail_dynamic = dynamic.thumbnail(128, 128);
        Image::from_dynamic(thumbnail_dynamic, true, RenderAssetUsages::RENDER_WORLD)
    } else {
        decoded
    };
    thumbnail.sampler = ImageSampler::linear();
    Some(thumbnails.add(thumbnail))
}

fn add_loaded_image(
    ui: &mut UiState,
    name: String,
    bytes: Vec<u8>,
    thumbnail: Option<Handle<Image>>,
) {
    let mut selected_index = ui.selected_image_index;
    if let Some(index) = ui
        .selected_images
        .iter()
        .position(|image| image.name == name)
    {
        if let Some(existing) = ui.selected_images.get_mut(index) {
            existing.bytes = bytes;
            existing.thumbnail = thumbnail;
        }
        if selected_index.is_none() {
            selected_index = Some(index);
        }
    } else {
        let index = ui.selected_images.len();
        ui.selected_images.push(LoadedImage {
            name,
            bytes,
            thumbnail,
        });
        if selected_index.is_none() {
            selected_index = Some(index);
        }
    }

    ui.selected_image_index = selected_index;
    ui.staging_revision = ui.staging_revision.wrapping_add(1);
}

fn set_loaded_images_status(ui: &mut UiState, verb: &str) {
    ui.status = format!(
        "{verb} {} images. press 'Run Reconstruction' to infer gaussians.",
        ui.selected_images.len()
    );
}

fn launch_options_from_env() -> LaunchOptions {
    #[cfg(target_arch = "wasm32")]
    {
        LaunchOptions::default()
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        parse_launch_options_from_iter(std::env::args_os().skip(1))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_launch_options_from_iter<I, T>(args: I) -> LaunchOptions
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let mut options = LaunchOptions::default();
    let mut iter = args.into_iter().map(Into::into).peekable();

    while let Some(arg) = iter.next() {
        let arg_text = arg.to_string_lossy();
        match arg_text.as_ref() {
            "-h" | "--help" => {
                options.print_help = true;
                break;
            }
            "--rev" => {
                options.print_rev = true;
                break;
            }
            "--image" => {
                if let Some(path) = iter.next() {
                    options.startup_images.push(PathBuf::from(path));
                }
            }
            "--images" => {
                while let Some(path) = iter.peek() {
                    if path.to_string_lossy().starts_with("--") {
                        break;
                    }
                    let path = iter.next().expect("peeked image arg should exist");
                    options.startup_images.push(PathBuf::from(path));
                }
            }
            "--auto-run" | "--run-on-start" => {
                options.auto_run = true;
            }
            "--output-glb" => {
                if let Some(path) = iter.next() {
                    options.output_glb = Some(PathBuf::from(path));
                }
            }
            "--exit-after-run" => {
                options.exit_after_run = true;
            }
            "--" => {
                for path in iter {
                    options.startup_images.push(PathBuf::from(path));
                }
                break;
            }
            _ if arg_text.starts_with("--") => {
                eprintln!("bevy_reconstruction: ignoring unknown argument '{arg_text}'");
            }
            _ => {
                options.startup_images.push(PathBuf::from(arg));
            }
        }
    }

    if options.output_glb.is_some() || options.exit_after_run {
        options.auto_run = true;
    }
    if !options.startup_images.is_empty() {
        options.auto_run = true;
    }
    options
}

#[cfg(not(target_arch = "wasm32"))]
fn print_native_help() {
    println!(
        "\
bevy_reconstruction
multi-view image -> gaussian reconstruction viewer

usage:
  bevy_reconstruction [OPTIONS] [IMAGE ...]

options:
  --image <PATH>            add one startup image (repeatable)
  --images <PATH...>        add multiple startup images
  --auto-run                run inference after startup images load
  --run-on-start            alias for --auto-run
  --output-glb <PATH>       write GLB after inference (implies --auto-run)
  --exit-after-run          exit after inference finishes (implies --auto-run)
  --rev                     print 7-char git revision and exit
  -h, --help                print this help and exit

build:
  {}
",
        burn_reconstruction::build_label()
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn ingest_startup_images(
    launch: Res<LaunchOptions>,
    mut ui: ResMut<UiState>,
    mut thumbnails: ResMut<Assets<Image>>,
    mut frustums: ResMut<FrustumDebug>,
) {
    if launch.startup_images.is_empty() {
        if launch.auto_run {
            ui.queued_run = true;
            ui.status = "no startup images provided".to_string();
        }
        return;
    }

    ui.selected_images.clear();
    ui.selected_image_index = None;
    ui.pending_camera_focus_index = None;
    ui.staging_revision = ui.staging_revision.wrapping_add(1);
    frustums.poses_world_from_camera.clear();

    let mut loaded_count = 0usize;
    let mut failed_count = 0usize;

    for path in &launch.startup_images {
        match fs::read(path) {
            Ok(bytes) => {
                let name = path_display_name(path);
                let thumbnail = decode_thumbnail_handle(thumbnails.as_mut(), &name, &bytes);
                add_loaded_image(&mut ui, name, bytes, thumbnail);
                loaded_count += 1;
            }
            Err(err) => {
                failed_count += 1;
                log::warn!(
                    "bevy_reconstruction: failed to load startup image {}: {err}",
                    path.display()
                );
            }
        }
    }

    if loaded_count == 0 {
        ui.status = "failed to load startup images from CLI args".to_string();
        log::warn!("bevy_reconstruction: no startup images could be loaded.");
        if launch.auto_run {
            ui.queued_run = true;
        }
        return;
    }

    set_loaded_images_status(&mut ui, "loaded");
    if failed_count > 0 {
        ui.status = format!(
            "loaded {} images ({} failed). press 'Run Reconstruction' to infer gaussians.",
            ui.selected_images.len(),
            failed_count
        );
    }
    log::info!(
        "bevy_reconstruction: loaded {} startup images ({} failed).",
        loaded_count,
        failed_count
    );

    if launch.auto_run {
        ui.queued_run = true;
        log::info!("bevy_reconstruction: auto-run queued.");
        if ui.selected_images.len() >= 2 {
            ui.status = format!(
                "loaded {} startup images. running reconstruction...",
                ui.selected_images.len()
            );
        } else {
            ui.status = format!(
                "loaded {} startup images; need at least 2 to run reconstruction.",
                ui.selected_images.len()
            );
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn ingest_startup_images() {}

#[cfg(not(target_arch = "wasm32"))]
fn path_display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

#[cfg(not(target_arch = "wasm32"))]
fn build_planar_cloud_from_pipeline(
    gaussians: &PipelineGaussians,
    options: &GlbExportOptions,
) -> Result<CloudBuildResult, String> {
    let packed = pack_gaussian_rows_full(gaussians)
        .map_err(|err| format!("failed to read packed gaussians from backend: {err}"))?;
    if packed.rows == 0 {
        return Err("no gaussians available".to_string());
    }

    build_planar_cloud_from_buffers(
        packed.rows,
        packed.d_sh,
        packed.row_width,
        packed.values.as_slice(),
        options,
    )
}

#[cfg(target_arch = "wasm32")]
async fn build_planar_cloud_from_pipeline_async(
    gaussians: &PipelineGaussians,
    options: &GlbExportOptions,
) -> Result<CloudBuildResult, String> {
    let packed = pack_gaussian_rows_full_async(gaussians)
        .await
        .map_err(|err| format!("failed to read packed gaussians from backend: {err}"))?;
    if packed.rows == 0 {
        return Err("no gaussians available".to_string());
    }

    build_planar_cloud_from_buffers(
        packed.rows,
        packed.d_sh,
        packed.row_width,
        packed.values.as_slice(),
        options,
    )
}

struct CloudBuildResult {
    cloud: PlanarGaussian3d,
    selected_gaussians: usize,
    scene_bounds: Option<SceneBounds>,
}

fn build_planar_cloud_from_buffers(
    total: usize,
    d_sh: usize,
    row_width: usize,
    packed_rows: &[f32],
    options: &GlbExportOptions,
) -> Result<CloudBuildResult, String> {
    let expected_row_width = 3 + 3 * d_sh + 9 + 3 + 4 + 1;
    if row_width != expected_row_width {
        return Err(format!(
            "unexpected packed row width: got {row_width}, expected {expected_row_width}"
        ));
    }
    if packed_rows.len() != total * row_width {
        return Err(format!(
            "packed gaussian row count mismatch: got {}, expected {}",
            packed_rows.len(),
            total * row_width
        ));
    }

    let harmonics_offset = 3usize;
    let covariance_offset = harmonics_offset + 3 * d_sh;
    let scales_offset = covariance_offset + 9;
    let rotations_offset = scales_offset + 3;
    let opacity_offset = rotations_offset + 4;

    let mut indices: Vec<usize> = (0..total)
        .filter(|&idx| packed_rows[idx * row_width + opacity_offset] >= options.opacity_threshold)
        .collect();
    if matches!(options.sort_mode, GlbSortMode::Opacity) {
        indices.sort_by(|lhs, rhs| {
            packed_rows[*rhs * row_width + opacity_offset]
                .partial_cmp(&packed_rows[*lhs * row_width + opacity_offset])
                .unwrap_or(Ordering::Equal)
        });
    }
    indices.truncate(options.max_gaussians.max(1));
    if indices.is_empty() {
        return Err("no gaussians survived viewer selection".to_string());
    }

    let mut packed = Vec::with_capacity(indices.len());
    let mut scene_positions = Vec::with_capacity(indices.len());
    let mut scene_weights = Vec::with_capacity(indices.len());
    let mut max_scale_component = 0.0f32;
    for idx in indices.iter().copied() {
        let off = idx * row_width;
        let pos_off = off;
        let harm_off = off + harmonics_offset;
        let cov_off = off + covariance_offset;
        let scale_off = off + scales_offset;
        let rot_off = off + rotations_offset;
        let opacity = packed_rows[off + opacity_offset];
        let transform = canonicalize_gaussian_transform_cv(
            [
                packed_rows[pos_off],
                packed_rows[pos_off + 1],
                packed_rows[pos_off + 2],
            ],
            [
                packed_rows[cov_off],
                packed_rows[cov_off + 1],
                packed_rows[cov_off + 2],
                packed_rows[cov_off + 3],
                packed_rows[cov_off + 4],
                packed_rows[cov_off + 5],
                packed_rows[cov_off + 6],
                packed_rows[cov_off + 7],
                packed_rows[cov_off + 8],
            ],
            [
                packed_rows[scale_off],
                packed_rows[scale_off + 1],
                packed_rows[scale_off + 2],
            ],
            [
                packed_rows[rot_off],
                packed_rows[rot_off + 1],
                packed_rows[rot_off + 2],
                packed_rows[rot_off + 3],
            ],
            opacity,
        );
        let position = transform.position;
        let rotation = transform.rotation_wxyz;
        let scale = transform.scale;
        let opacity = transform.opacity;
        max_scale_component = max_scale_component.max(scale[0].max(scale[1]).max(scale[2]));
        scene_positions.push(Vec3::new(position[0], position[1], position[2]));
        scene_weights.push(opacity.max(0.01));

        // Preserve raw SH coefficients from model output for correct color rendering.
        let mut spherical_harmonic = SphericalHarmonicCoefficients::default();
        let coeffs_per_channel = spherical_harmonic.coefficients.len() / 3;
        let copy_coeffs = d_sh.min(coeffs_per_channel);
        for coeff in 0..copy_coeffs {
            for channel in 0..3 {
                // Model storage is channel-major [R coeffs..., G coeffs..., B coeffs...].
                let src = harm_off + channel * d_sh + coeff;
                // Renderer storage is interleaved by SH degree.
                let dst = coeff * 3 + channel;
                spherical_harmonic.set(dst, packed_rows[src]);
            }
        }

        packed.push(Gaussian3d {
            position_visibility: [position[0], position[1], position[2], 1.0].into(),
            spherical_harmonic,
            rotation: rotation.into(),
            scale_opacity: [scale[0], scale[1], scale[2], opacity].into(),
        });
    }

    let scene_bounds = estimate_scene_bounds(
        scene_positions.as_slice(),
        scene_weights.as_slice(),
        max_scale_component,
    );

    Ok(CloudBuildResult {
        cloud: packed.into(),
        selected_gaussians: indices.len(),
        scene_bounds,
    })
}

fn estimate_scene_bounds(
    positions: &[Vec3],
    weights: &[f32],
    max_scale_component: f32,
) -> Option<SceneBounds> {
    if positions.is_empty() || positions.len() != weights.len() {
        return None;
    }

    let mut weighted_sum = Vec3::ZERO;
    let mut weight_sum = 0.0f32;
    for (position, weight) in positions.iter().zip(weights.iter()) {
        if !position.is_finite() {
            continue;
        }
        let clamped_weight = if weight.is_finite() {
            (*weight).max(0.01)
        } else {
            0.01
        };
        weighted_sum += *position * clamped_weight;
        weight_sum += clamped_weight;
    }
    if weight_sum <= f32::EPSILON {
        return None;
    }

    let center = weighted_sum / weight_sum;
    if !center.is_finite() {
        return None;
    }

    let mut distances = Vec::with_capacity(positions.len());
    for position in positions.iter().copied() {
        if !position.is_finite() {
            continue;
        }
        let distance = (position - center).length();
        if distance.is_finite() {
            distances.push(distance);
        }
    }
    if distances.is_empty() {
        return None;
    }

    distances.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    let quantile_index =
        ((distances.len().saturating_sub(1)) as f32 * SCENE_RADIUS_QUANTILE).round() as usize;
    let quantile_radius = distances[quantile_index.min(distances.len() - 1)];
    let scale_padding = if max_scale_component.is_finite() {
        max_scale_component.max(0.0) * SCENE_RADIUS_PADDING_SCALE
    } else {
        0.0
    };
    let radius = (quantile_radius + scale_padding).max(SCENE_RADIUS_FLOOR);
    if !radius.is_finite() {
        return None;
    }

    Some(SceneBounds { center, radius })
}

fn cv_to_bevy_basis_mat4() -> Mat4 {
    Mat4::from_diagonal(Vec4::new(1.0, -1.0, -1.0, 1.0))
}

fn reconstruction_cloud_components(
    handle: Handle<PlanarGaussian3d>,
) -> (
    Name,
    PlanarGaussian3dHandle,
    CloudSettings,
    Transform,
    Visibility,
) {
    (
        Name::new("reconstruction_cloud"),
        PlanarGaussian3dHandle(handle),
        CloudSettings {
            sort_mode: SortMode::Std,
            ..CloudSettings::default()
        },
        Transform::default(),
        Visibility::Visible,
    )
}

fn spawn_reconstruction_cloud(commands: &mut Commands, handle: Handle<PlanarGaussian3d>) -> Entity {
    commands.spawn(reconstruction_cloud_components(handle)).id()
}

fn row_major_pose_to_mat4(pose: [[f32; 4]; 4]) -> Mat4 {
    let world_from_camera_cv = Mat4::from_cols(
        Vec4::new(pose[0][0], pose[1][0], pose[2][0], pose[3][0]),
        Vec4::new(pose[0][1], pose[1][1], pose[2][1], pose[3][1]),
        Vec4::new(pose[0][2], pose[1][2], pose[2][2], pose[3][2]),
        Vec4::new(pose[0][3], pose[1][3], pose[2][3], pose[3][3]),
    );
    let basis = cv_to_bevy_basis_mat4();
    basis * world_from_camera_cv * basis
}

#[derive(Debug, Clone, Copy)]
struct FrustumShape {
    near: f32,
    far: f32,
    near_half_base: f32,
    far_half_base: f32,
}

fn draw_camera_frustums(
    mut gizmos: Gizmos,
    frustums: Res<FrustumDebug>,
    ui: Res<UiState>,
    images: Res<Assets<Image>>,
) {
    let shape = adaptive_frustum_shape(
        frustums.poses_world_from_camera.as_slice(),
        ui.active_scene_bounds,
    );
    for (index, pose) in frustums.poses_world_from_camera.iter().enumerate() {
        let aspect = frustum_view_aspect(index, ui.as_ref(), images.as_ref());
        draw_single_frustum(
            &mut gizmos,
            *pose,
            aspect,
            shape,
            Color::srgb(0.2, 0.9, 0.95),
        );
    }
}

fn adaptive_frustum_shape(
    poses_world_from_camera: &[Mat4],
    scene_bounds: Option<SceneBounds>,
) -> FrustumShape {
    let mut far = FRUSTUM_DEFAULT_FAR;

    if let Some(neighbor_distance) = median_nearest_camera_distance(poses_world_from_camera) {
        far = neighbor_distance * FRUSTUM_NEIGHBOR_FAR_SCALE;
    }

    if let Some(bounds) = scene_bounds {
        if bounds.radius.is_finite() && bounds.radius > 0.0 {
            let cap = (bounds.radius * FRUSTUM_SCENE_RADIUS_FAR_CAP_SCALE).max(FRUSTUM_MIN_FAR);
            far = far.min(cap);
        }
    }

    far = far.clamp(FRUSTUM_MIN_FAR, FRUSTUM_MAX_FAR);
    let near = (far * FRUSTUM_NEAR_TO_FAR_RATIO).max(FRUSTUM_MIN_FAR * FRUSTUM_NEAR_TO_FAR_RATIO);

    FrustumShape {
        near,
        far,
        near_half_base: near * FRUSTUM_NEAR_HALF_RATIO,
        far_half_base: far * FRUSTUM_FAR_HALF_RATIO,
    }
}

fn median_nearest_camera_distance(poses_world_from_camera: &[Mat4]) -> Option<f32> {
    if poses_world_from_camera.len() < 2 {
        return None;
    }

    let origins = poses_world_from_camera
        .iter()
        .map(|pose| pose.to_scale_rotation_translation().2)
        .collect::<Vec<_>>();

    let mut nearest = Vec::with_capacity(origins.len());
    for (index, origin) in origins.iter().enumerate() {
        if !origin.is_finite() {
            continue;
        }
        let mut best = f32::INFINITY;
        for (other_index, other) in origins.iter().enumerate() {
            if index == other_index || !other.is_finite() {
                continue;
            }
            let distance = origin.distance(*other);
            if distance.is_finite() && distance > f32::EPSILON {
                best = best.min(distance);
            }
        }
        if best.is_finite() {
            nearest.push(best);
        }
    }

    if nearest.is_empty() {
        return None;
    }
    nearest.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    Some(nearest[nearest.len() / 2])
}

fn frustum_view_aspect(index: usize, ui: &UiState, images: &Assets<Image>) -> f32 {
    let Some(staged) = ui.selected_images.get(index) else {
        return 1.0;
    };
    let Some(handle) = staged.thumbnail.as_ref() else {
        return 1.0;
    };
    let Some(image) = images.get(handle) else {
        return 1.0;
    };

    let width = image.texture_descriptor.size.width as f32;
    let height = image.texture_descriptor.size.height as f32;
    if width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0 {
        width / height
    } else {
        1.0
    }
}

fn draw_single_frustum(
    gizmos: &mut Gizmos,
    world_from_camera: Mat4,
    image_aspect: f32,
    shape: FrustumShape,
    color: Color,
) {
    // Camera poses are converted to Bevy basis in `row_major_pose_to_mat4`.
    // In Bevy camera convention, forward is along local -Z.
    let near = shape.near;
    let far = shape.far;
    let (half_near_x, half_near_y) = frustum_half_extents(shape.near_half_base, image_aspect);
    let (half_far_x, half_far_y) = frustum_half_extents(shape.far_half_base, image_aspect);

    let (_, mut rotation, mut origin) = world_from_camera.to_scale_rotation_translation();
    if !rotation.is_finite() || rotation.length_squared() <= f32::EPSILON {
        rotation = Quat::IDENTITY;
    } else {
        rotation = rotation.normalize();
    }
    if !origin.is_finite() {
        origin = Vec3::ZERO;
    }

    let near_corners = [
        Vec3::new(-half_near_x, half_near_y, -near),
        Vec3::new(half_near_x, half_near_y, -near),
        Vec3::new(half_near_x, -half_near_y, -near),
        Vec3::new(-half_near_x, -half_near_y, -near),
    ]
    .map(|point| origin + rotation * point);

    let far_corners = [
        Vec3::new(-half_far_x, half_far_y, -far),
        Vec3::new(half_far_x, half_far_y, -far),
        Vec3::new(half_far_x, -half_far_y, -far),
        Vec3::new(-half_far_x, -half_far_y, -far),
    ]
    .map(|point| origin + rotation * point);

    for corners in [near_corners, far_corners] {
        for index in 0..4 {
            let next = (index + 1) % 4;
            gizmos.line(corners[index], corners[next], color);
        }
    }

    for index in 0..4 {
        gizmos.line(near_corners[index], far_corners[index], color);
    }
}

fn frustum_half_extents(base_half: f32, image_aspect: f32) -> (f32, f32) {
    let aspect = if image_aspect.is_finite() && image_aspect > 0.0 {
        image_aspect
    } else {
        1.0
    };
    if aspect >= 1.0 {
        (base_half * aspect, base_half)
    } else {
        (base_half, base_half / aspect.max(1e-6))
    }
}

fn format_inference_complete_status(
    timings: &burn_reconstruction::ForwardTimings,
    selected_gaussians: usize,
    total_gaussians: usize,
    frustum_count: Option<usize>,
) -> String {
    let millis = timings.total.as_secs_f64() * 1000.0;
    match frustum_count {
        Some(count) if millis.is_finite() && millis >= 0.05 => format!(
            "inference complete: {} / {} gaussians, {:.2} ms total (frustums: {})",
            selected_gaussians, total_gaussians, millis, count
        ),
        Some(count) => format!(
            "inference complete: {} / {} gaussians (frustums: {})",
            selected_gaussians, total_gaussians, count
        ),
        None if millis.is_finite() && millis >= 0.05 => format!(
            "inference complete: {} / {} gaussians, {:.2} ms total",
            selected_gaussians, total_gaussians, millis
        ),
        None => format!(
            "inference complete: {} / {} gaussians",
            selected_gaussians, total_gaussians
        ),
    }
}

fn status_is_busy(status_lower: &str) -> bool {
    [
        "running inference",
        "queued reconstruction",
        "reconstruction queued",
        "loading model",
        "resolving model",
        "initializing yono",
        "initializing wasm",
        "downloading",
        "downloaded",
        "manifest",
        "parts",
        "cached",
        "fetching",
        "preparing inference",
    ]
    .iter()
    .any(|needle| status_lower.contains(needle))
}

fn status_badge_from_ui(ui: &UiState) -> (String, StatusBadgeTone) {
    let status_lower = ui.status.to_ascii_lowercase();
    if status_lower.contains("failed") || status_lower.contains("error") {
        return ("error".to_string(), StatusBadgeTone::Error);
    }
    if status_lower.contains("inference complete") {
        return ("complete".to_string(), StatusBadgeTone::Success);
    }
    if status_is_busy(&status_lower) {
        return ("running".to_string(), StatusBadgeTone::Busy);
    }
    if ui.selected_images.len() >= 2 {
        return (
            format!("ready ({})", ui.selected_images.len()),
            StatusBadgeTone::Pending,
        );
    }
    if ui.selected_images.is_empty() {
        return ("idle".to_string(), StatusBadgeTone::Idle);
    }
    (
        format!("need {}", 2 - ui.selected_images.len()),
        StatusBadgeTone::Pending,
    )
}

fn status_badge_palette(tone: StatusBadgeTone) -> (Color, Color, Color, Color) {
    match tone {
        StatusBadgeTone::Idle => (
            Color::srgb(0.52, 0.56, 0.64),
            Color::srgb(0.72, 0.76, 0.84),
            STATUS_BADGE_BG,
            STATUS_BADGE_BORDER,
        ),
        StatusBadgeTone::Pending => (
            Color::srgb(0.26, 0.62, 0.88),
            Color::srgb(0.76, 0.86, 0.98),
            Color::srgb(0.08, 0.15, 0.2),
            Color::srgb(0.2, 0.4, 0.55),
        ),
        StatusBadgeTone::Busy => (
            Color::srgb(0.93, 0.66, 0.2),
            Color::srgb(0.95, 0.89, 0.74),
            Color::srgb(0.21, 0.15, 0.08),
            Color::srgb(0.58, 0.41, 0.18),
        ),
        StatusBadgeTone::Success => (
            Color::srgb(0.26, 0.73, 0.46),
            Color::srgb(0.84, 0.96, 0.88),
            Color::srgb(0.09, 0.18, 0.11),
            Color::srgb(0.24, 0.46, 0.3),
        ),
        StatusBadgeTone::Error => (
            Color::srgb(0.86, 0.28, 0.28),
            Color::srgb(0.96, 0.72, 0.72),
            Color::srgb(0.22, 0.1, 0.1),
            Color::srgb(0.58, 0.23, 0.23),
        ),
    }
}

fn button_palette(
    kind: ControlButtonKind,
    interaction: Interaction,
    disabled: bool,
) -> (Color, Color, Color) {
    if disabled {
        return (
            BUTTON_BG_DISABLED,
            BUTTON_BORDER_DISABLED,
            BUTTON_TEXT_DISABLED,
        );
    }

    match (kind, interaction) {
        (ControlButtonKind::Accent, Interaction::Pressed) => (
            BUTTON_OPEN_BG_PRESSED,
            BUTTON_OPEN_BORDER_PRESSED,
            BUTTON_TEXT,
        ),
        (ControlButtonKind::Accent, Interaction::Hovered) => {
            (BUTTON_OPEN_BG_HOVER, BUTTON_OPEN_BORDER_HOVER, BUTTON_TEXT)
        }
        (ControlButtonKind::Accent, Interaction::None) => {
            (BUTTON_OPEN_BG, BUTTON_OPEN_BORDER, BUTTON_TEXT)
        }
        (ControlButtonKind::Danger, Interaction::Pressed) => (
            Color::srgb(0.35, 0.15, 0.16),
            Color::srgb(0.68, 0.31, 0.33),
            BUTTON_TEXT,
        ),
        (ControlButtonKind::Danger, Interaction::Hovered) => (
            Color::srgb(0.28, 0.13, 0.14),
            Color::srgb(0.55, 0.27, 0.29),
            BUTTON_TEXT,
        ),
        (ControlButtonKind::Danger, Interaction::None) => (
            Color::srgb(0.22, 0.11, 0.12),
            Color::srgb(0.44, 0.23, 0.24),
            BUTTON_TEXT,
        ),
        (_, Interaction::Pressed) => (BUTTON_BG_PRESSED, BUTTON_BORDER_PRESSED, BUTTON_TEXT),
        (_, Interaction::Hovered) => (BUTTON_BG_HOVER, BUTTON_BORDER_HOVER, BUTTON_TEXT),
        (_, Interaction::None) => (BUTTON_BG, BUTTON_BORDER, BUTTON_TEXT),
    }
}

fn update_status_label(ui: Res<UiState>, mut labels: Query<&mut Text, With<StatusLabel>>) {
    for mut text in &mut labels {
        text.0 = ui.status.clone();
    }
}

fn update_selection_label(ui: Res<UiState>, mut labels: Query<&mut Text, With<SelectionLabel>>) {
    for mut text in &mut labels {
        text.0 = format!("selected images: {}", ui.selected_images.len());
    }
}

fn update_status_badge(
    ui: Res<UiState>,
    mut texts: Query<&mut Text, With<QueueBadgeText>>,
    mut dots: Query<&mut BackgroundColor, (With<QueueBadgeDot>, Without<QueueBadge>)>,
    mut badges: Query<QueueBadgeVisualQuery<'_>, QueueBadgeVisualFilter>,
    mut text_colors: Query<&mut TextColor, With<QueueBadgeText>>,
) {
    let (label, tone) = status_badge_from_ui(&ui);
    let (dot, text, badge_bg, badge_border) = status_badge_palette(tone);

    for mut entry in &mut texts {
        entry.0 = label.clone();
    }
    for mut entry in &mut text_colors {
        entry.0 = text;
    }
    for mut entry in &mut dots {
        entry.0 = dot;
    }
    for (mut bg, mut border) in &mut badges {
        bg.0 = badge_bg;
        *border = BorderColor::all(badge_border);
    }
}

fn update_control_button_visuals(
    ui: Res<UiState>,
    frustums: Res<FrustumDebug>,
    mut buttons: Query<ControlButtonVisualQuery<'_>, With<Button>>,
    mut labels: Query<&mut TextColor, With<ButtonLabel>>,
) {
    let run_disabled = ui.selected_images.len() < 2;
    let clear_disabled = ui.selected_images.is_empty()
        && ui.active_cloud.is_none()
        && frustums.poses_world_from_camera.is_empty();

    for (interaction, button, run, clear, children, mut bg, mut border) in &mut buttons {
        let disabled = (run.is_some() && run_disabled) || (clear.is_some() && clear_disabled);
        let (button_bg, button_border, text_color) =
            button_palette(button.0, *interaction, disabled);

        bg.0 = button_bg;
        *border = BorderColor::all(button_border);
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child) {
                label.0 = text_color;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parse_launch_options_supports_images_and_flags() {
        let args = [
            "--images",
            "a.png",
            "b.png",
            "--auto-run",
            "--output-glb",
            "out.glb",
        ];
        let parsed = parse_launch_options_from_iter(args);
        assert_eq!(parsed.startup_images.len(), 2);
        assert_eq!(parsed.startup_images[0], PathBuf::from("a.png"));
        assert_eq!(parsed.startup_images[1], PathBuf::from("b.png"));
        assert!(parsed.auto_run);
        assert_eq!(parsed.output_glb, Some(PathBuf::from("out.glb")));
        assert!(!parsed.exit_after_run);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parse_launch_options_supports_repeated_image_and_exit() {
        let args = [
            "--image",
            "left.png",
            "--image",
            "right.png",
            "--exit-after-run",
        ];
        let parsed = parse_launch_options_from_iter(args);
        assert_eq!(parsed.startup_images.len(), 2);
        assert_eq!(parsed.startup_images[0], PathBuf::from("left.png"));
        assert_eq!(parsed.startup_images[1], PathBuf::from("right.png"));
        assert!(parsed.auto_run);
        assert!(parsed.exit_after_run);
        assert!(parsed.output_glb.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parse_launch_options_auto_runs_when_images_present() {
        let args = ["--image", "left.png", "--image", "right.png"];
        let parsed = parse_launch_options_from_iter(args);
        assert_eq!(parsed.startup_images.len(), 2);
        assert!(parsed.auto_run);
        assert!(!parsed.exit_after_run);
        assert!(parsed.output_glb.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parse_launch_options_supports_help_and_rev() {
        let help = parse_launch_options_from_iter(["--help"]);
        assert!(help.print_help);
        assert!(!help.print_rev);

        let rev = parse_launch_options_from_iter(["--rev"]);
        assert!(rev.print_rev);
        assert!(!rev.print_help);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn reconstruction_cloud_uses_cpu_sort_mode() {
        let cloud = reconstruction_cloud_components(Handle::<PlanarGaussian3d>::default());
        assert!(matches!(cloud.2.sort_mode, SortMode::Std));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cv_xyzw_identity_maps_to_bevy_wxyz_identity() {
        let q = cv_xyzw_to_canonical_wxyz([0.0, 0.0, 0.0, 1.0]);
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!(q[1].abs() < 1e-6);
        assert!(q[2].abs() < 1e-6);
        assert!(q[3].abs() < 1e-6);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn row_major_pose_identity_is_stable_after_basis_conversion() {
        let pose = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mat = row_major_pose_to_mat4(pose);
        let cols = mat.to_cols_array();
        let id = Mat4::IDENTITY.to_cols_array();
        for (lhs, rhs) in cols.iter().zip(id.iter()) {
            assert!((*lhs - *rhs).abs() < 1e-6);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn frustum_forward_points_along_negative_z_in_bevy_space() {
        let world_from_camera = Mat4::IDENTITY;
        let shape = adaptive_frustum_shape(&[world_from_camera], None);
        let near_point = world_from_camera.transform_point3(Vec3::new(0.0, 0.0, -shape.near));
        let far_point = world_from_camera.transform_point3(Vec3::new(0.0, 0.0, -shape.far));
        assert!(near_point.z < 0.0);
        assert!(far_point.z < near_point.z);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn adaptive_frustum_shrinks_for_dense_camera_clusters() {
        let poses = [
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(0.08, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(0.09, 0.0, 0.0)),
        ];
        let shape = adaptive_frustum_shape(&poses, None);
        assert!(shape.far < FRUSTUM_DEFAULT_FAR);
        assert!(shape.far >= FRUSTUM_MIN_FAR);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn adaptive_frustum_respects_scene_radius_cap() {
        let poses = [
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)),
        ];
        let scene_bounds = SceneBounds {
            center: Vec3::ZERO,
            radius: 0.5,
        };
        let shape = adaptive_frustum_shape(&poses, Some(scene_bounds));
        let expected_cap = scene_bounds.radius * FRUSTUM_SCENE_RADIUS_FAR_CAP_SCALE;
        assert!(shape.far <= expected_cap + 1e-6);
        assert!(shape.far >= FRUSTUM_MIN_FAR);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn frustum_half_extents_expand_horizontally_for_landscape_aspect() {
        let (half_x, half_y) = frustum_half_extents(0.03, 16.0 / 9.0);
        assert!(half_x > half_y);
        assert!((half_y - 0.03).abs() < 1e-6);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn frustum_half_extents_expand_vertically_for_portrait_aspect() {
        let (half_x, half_y) = frustum_half_extents(0.03, 3.0 / 4.0);
        assert!(half_y > half_x);
        assert!((half_x - 0.03).abs() < 1e-6);
    }

    #[test]
    fn viewer_scale_sanitization_enforces_floor_and_ratio() {
        let scale = sanitize_scale_for_viewer([0.0, 0.3, 1e-6]);
        assert!(scale.iter().all(|value| value.is_finite()));
        assert!(scale.iter().all(|value| *value >= 1e-3));
        assert!(scale.iter().all(|value| *value <= 0.3));

        let min_scale = scale.iter().copied().fold(f32::INFINITY, f32::min);
        let max_scale = scale.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_scale <= min_scale * 256.0 + 1e-6);
    }

    #[test]
    fn canonical_transform_applies_cv_basis_flip() {
        let transformed = canonicalize_gaussian_transform_cv(
            [0.1, 0.2, 0.3],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.01, 0.02, 0.03],
            [0.0, 0.0, 0.0, 1.0],
            1.0,
        );
        assert_eq!(transformed.position, [0.1, -0.2, -0.3]);
    }

    #[test]
    fn planar_cloud_conversion_preserves_model_opacity_values() {
        let d_sh = 1usize;
        let row_width = 3 + 3 * d_sh + 9 + 3 + 4 + 1;
        let mut packed = Vec::with_capacity(row_width * 2);

        let mut push_row = |opacity: f32| {
            packed.extend_from_slice(&[
                0.0, 0.0, 0.0, // mean
                0.2, 0.3, 0.4, // SH dc
                1.0, 0.0, 0.0, // covariance row 0
                0.0, 1.0, 0.0, // covariance row 1
                0.0, 0.0, 1.0, // covariance row 2
                0.1, 0.1, 0.1, // fallback scale
                0.0, 0.0, 0.0, 1.0,     // fallback rotation xyzw
                opacity, // opacity
            ]);
        };

        push_row(0.25);
        push_row(0.75);

        let result = build_planar_cloud_from_buffers(
            2,
            d_sh,
            row_width,
            packed.as_slice(),
            &GlbExportOptions {
                max_gaussians: 2,
                opacity_threshold: 0.0,
                sort_mode: GlbSortMode::Index,
            },
        )
        .expect("planar cloud conversion should succeed");

        assert_eq!(result.selected_gaussians, 2);
        assert!((result.cloud.scale_opacity[0].opacity - 0.25).abs() < 1e-6);
        assert!((result.cloud.scale_opacity[1].opacity - 0.75).abs() < 1e-6);
    }

    #[test]
    fn planar_cloud_conversion_clamps_only_out_of_range_opacity() {
        let d_sh = 1usize;
        let row_width = 3 + 3 * d_sh + 9 + 3 + 4 + 1;
        let mut packed = Vec::with_capacity(row_width * 3);

        let mut push_row = |opacity: f32| {
            packed.extend_from_slice(&[
                0.0, 0.0, 0.0, // mean
                0.1, 0.1, 0.1, // SH dc
                1.0, 0.0, 0.0, // covariance row 0
                0.0, 1.0, 0.0, // covariance row 1
                0.0, 0.0, 1.0, // covariance row 2
                0.1, 0.1, 0.1, // fallback scale
                0.0, 0.0, 0.0, 1.0,     // fallback rotation xyzw
                opacity, // opacity
            ]);
        };

        push_row(-0.5);
        push_row(0.42);
        push_row(1.8);

        let result = build_planar_cloud_from_buffers(
            3,
            d_sh,
            row_width,
            packed.as_slice(),
            &GlbExportOptions {
                max_gaussians: 3,
                opacity_threshold: -1.0,
                sort_mode: GlbSortMode::Index,
            },
        )
        .expect("planar cloud conversion should succeed");

        assert_eq!(result.selected_gaussians, 3);
        assert!((result.cloud.scale_opacity[0].opacity - 0.0).abs() < 1e-6);
        assert!((result.cloud.scale_opacity[1].opacity - 0.42).abs() < 1e-6);
        assert!((result.cloud.scale_opacity[2].opacity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn add_loaded_image_replaces_existing_name() {
        let mut ui = UiState::default();
        add_loaded_image(&mut ui, "same.png".to_string(), vec![1, 2, 3], None);
        add_loaded_image(&mut ui, "same.png".to_string(), vec![9, 8], None);
        assert_eq!(ui.selected_images.len(), 1);
        assert_eq!(ui.selected_images[0].bytes, vec![9, 8]);
    }

    #[test]
    fn status_badge_detects_error_and_success() {
        let mut ui = UiState {
            status: "inference complete: 100 / 100".to_string(),
            ..UiState::default()
        };
        let (_, tone_success) = status_badge_from_ui(&ui);
        assert_eq!(tone_success, StatusBadgeTone::Success);

        ui.status = "failed to load startup images".to_string();
        let (_, tone_error) = status_badge_from_ui(&ui);
        assert_eq!(tone_error, StatusBadgeTone::Error);
    }

    #[test]
    fn status_badge_treats_model_init_and_resolve_as_busy() {
        let mut ui = UiState {
            status: "resolving model weights...".to_string(),
            ..UiState::default()
        };
        let (_, tone_resolving) = status_badge_from_ui(&ui);
        assert_eq!(tone_resolving, StatusBadgeTone::Busy);

        ui.status = "initializing yono pipeline modules...".to_string();
        let (_, tone_initializing) = status_badge_from_ui(&ui);
        assert_eq!(tone_initializing, StatusBadgeTone::Busy);

        ui.status = "yono modules initialized in 12.0s; preparing inference...".to_string();
        let (_, tone_preparing) = status_badge_from_ui(&ui);
        assert_eq!(tone_preparing, StatusBadgeTone::Busy);

        ui.status = "loading model weights: cached backbone part 1/16".to_string();
        let (_, tone_cached_parts) = status_badge_from_ui(&ui);
        assert_eq!(tone_cached_parts, StatusBadgeTone::Busy);
    }

    #[test]
    fn button_palette_uses_disabled_colors() {
        let (bg, border, text) =
            button_palette(ControlButtonKind::Primary, Interaction::Hovered, true);
        assert_eq!(bg, BUTTON_BG_DISABLED);
        assert_eq!(border, BUTTON_BORDER_DISABLED);
        assert_eq!(text, BUTTON_TEXT_DISABLED);
    }

    #[test]
    fn orbit_targets_identity_view_are_stable() {
        let Some((focus, yaw, pitch, radius)) = orbit_targets_for_view(Mat4::IDENTITY, 2.0, None)
        else {
            panic!("expected valid orbit target");
        };
        assert!((focus.x - 0.0).abs() < 1e-6);
        assert!((focus.y - 0.0).abs() < 1e-6);
        assert!((focus.z + 2.0).abs() < 1e-6);
        assert!(yaw.abs() < 1e-6);
        assert!(pitch.abs() < 1e-6);
        assert!((radius - 2.0).abs() < 1e-6);
    }

    #[test]
    fn orbit_targets_use_scene_bounds_over_large_fallback_radius() {
        let world_from_camera = Mat4::IDENTITY;
        let scene_bounds = SceneBounds {
            center: Vec3::new(0.0, 0.0, -5.0),
            radius: 1.0,
        };
        let Some((focus, _, _, radius)) =
            orbit_targets_for_view(world_from_camera, 30.0, Some(scene_bounds))
        else {
            panic!("expected valid scene-aware orbit target");
        };

        // Focus should land near the visible scene rather than using a huge stale orbit radius.
        assert!(focus.z < -3.5 && focus.z > -5.5);
        assert!(radius < 10.0);
    }

    #[test]
    fn estimate_scene_bounds_is_robust_to_single_outlier() {
        let mut positions = vec![
            Vec3::new(-0.2, 0.0, -1.0),
            Vec3::new(0.2, 0.0, -1.0),
            Vec3::new(0.0, 0.2, -1.0),
            Vec3::new(0.0, -0.2, -1.0),
            Vec3::new(0.1, 0.1, -1.1),
            Vec3::new(-0.1, -0.1, -0.9),
        ];
        positions.push(Vec3::new(1000.0, 0.0, 0.0));

        let weights = vec![1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.01];
        let Some(bounds) = estimate_scene_bounds(positions.as_slice(), weights.as_slice(), 0.02)
        else {
            panic!("expected scene bounds");
        };

        assert!(bounds.center.z < -0.7 && bounds.center.z > -1.3);
        assert!(bounds.radius < 3.0);
    }

    #[test]
    fn ui_button_activation_triggers_once_on_release_over_button() {
        assert!(!ui_button_activated(Interaction::Pressed, true, false));
        assert!(!ui_button_activated(Interaction::Pressed, false, false));
        assert!(!ui_button_activated(Interaction::None, false, true));
        assert!(ui_button_activated(Interaction::Hovered, false, true));
    }
}
