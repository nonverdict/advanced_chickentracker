# --- START OF FILE server_websocket.py ---
import asyncio
import cv2
import json
import time
import logging
from pathlib import Path
from aiohttp import web
import concurrent.futures
import numpy as np
from ultralytics import YOLO
import torch 

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 5000
WEBCAM_INDEX = 0 
WEBCAM_IS_VIDEO_FILE = isinstance(WEBCAM_INDEX, str)
STATIC_VIDEO_PATH_STR = 'static/chicken_demo.mp4' 
FRAME_WIDTH = 640 
FRAME_HEIGHT = 480
JPEG_QUALITY = 70
FRAME_DELAY = 1 / 15 # Aim for 15 FPS
CAMERA_INIT_TIMEOUT = 25
STATIC_VIDEO_INIT_TIMEOUT = 15

YOLO_MODEL_PATH = 'yolov8x.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.25 # Default for yolov8, can adjust
YOLO_IOU_THRESHOLD_NMS = 0.45
# ===>>> CRITICAL FIX: Class ID for "bird" in COCO is 14 <<<===
YOLO_TARGET_CLASSES = [14] # Was [15] (cat)

# --- Globals ---
# ... (rest of globals as before) ...
STATIC_FILES_DIR = Path(__file__).parent.resolve()
yolo_model = None
tracked_objects = {}; next_track_id = 0; current_frame_index = 0 
MAX_UNSEEN_FRAMES_TRACKER = 25; TRACKER_MAX_DIST = 100 
CLIENTS = set(); SUBSCRIPTIONS = { "normal_video": set(), "static_video": set() }
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s')
logger = logging.getLogger("PoultryScopeServer")
normal_video_cap, normal_video_lock, normal_video_task = None, asyncio.Lock(), None
static_video_cap, static_video_lock, static_video_task = None, asyncio.Lock(), None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix='ProcWorker')

# ... (Rest of the file from the PREVIOUS CORRECTED version, no other changes needed for this specific bug)
# Make sure to use the version that fixed the IndentationErrors you found earlier.
# The only change needed is YOLO_TARGET_CLASSES.

async def initialize_yolo_model_async():
    global yolo_model
    if yolo_model is not None: return True
    logger.info(f"Initializing YOLO model using '{YOLO_MODEL_PATH}'...")
    try:
        loop = asyncio.get_running_loop()
        model_name_or_path = YOLO_MODEL_PATH
        model_path_obj = Path(model_name_or_path)
        true_model_source = model_name_or_path 
        if model_path_obj.is_file(): true_model_source = str(model_path_obj.resolve())
        elif model_path_obj.parts and not model_path_obj.name == model_name_or_path : 
            resolved_local = (STATIC_FILES_DIR / model_name_or_path).resolve()
            if resolved_local.is_file(): true_model_source = str(resolved_local)
            else: logger.error(f"YOLO model file not found: {model_name_or_path} or {resolved_local}"); return False
        
        logger.info(f"Attempting to load YOLO model from/using: {true_model_source}")
        temp_model = await loop.run_in_executor(executor, lambda: YOLO(true_model_source))
        model_device = getattr(temp_model, 'device', 'unknown_device')
        logger.info(f"YOLO model loaded. Ultralytics reports model device: {model_device}")
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA is available. Device count: {torch.cuda.device_count()}. Current torch device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
            if 'cpu' in str(model_device).lower(): logger.warning("Model loaded on CPU despite CUDA. Inference calls will attempt 'cuda'.")
        else: logger.warning("PyTorch CUDA NOT available. YOLO will run on CPU.")
        yolo_model = temp_model
        dummy_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        effective_warmup_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Performing YOLO model warmup on {FRAME_WIDTH}x{FRAME_HEIGHT} frame (attempting device: {effective_warmup_device})...")
        await loop.run_in_executor(executor, lambda: yolo_model(dummy_frame, verbose=False, conf=0.05, device=effective_warmup_device))
        logger.info(f"YOLO model warmup complete. Model effective device after warmup: {getattr(yolo_model, 'device', 'N/A')}")
        return True
    except Exception as e: logger.error(f"Failed to initialize YOLO model: {e}", exc_info=True); yolo_model = None; return False

def _open_cv_capture_blocking(source_identifier, source_is_file: bool):
    log_name = str(source_identifier); actual_source_for_cv = source_identifier
    if source_is_file:
        path_obj = Path(source_identifier)
        resolved_path = path_obj if path_obj.is_absolute() else (STATIC_FILES_DIR / path_obj).resolve()
        if not resolved_path.is_file(): logger.error(f"[Executor] Video file not found: {resolved_path}"); return None
        actual_source_for_cv = str(resolved_path); log_name = resolved_path.name
        logger.info(f"[Executor] Attempting cv2.VideoCapture for video file: {actual_source_for_cv}...")
    else: logger.info(f"[Executor] Attempting cv2.VideoCapture for webcam index: {actual_source_for_cv}...")
    try:
        backends_to_try = [cv2.CAP_ANY]; cap = None
        if not source_is_file: backends_to_try.extend([cv2.CAP_V4L2])
        for backend in backends_to_try:
            logger.debug(f"[Executor] Trying OpenCV backend: {backend} for source {actual_source_for_cv}")
            cap = cv2.VideoCapture(actual_source_for_cv, backend)
            if cap and cap.isOpened(): logger.info(f"[Executor] Successfully opened '{log_name}' with backend: {backend}"); break
            if cap: cap.release(); cap = None 
        if not cap or not cap.isOpened(): logger.error(f"[Executor] cv2.VideoCapture({actual_source_for_cv}) failed with all tried backends."); return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[Executor] Source '{log_name}' details - Actual Res: {w}x{h}, FPS: {fps:.2f}.")
        return cap
    except Exception as e: logger.error(f"[Executor] Exception during cv2.VideoCapture({actual_source_for_cv}): {e}", exc_info=True); return None

def _release_cv_capture_blocking(cap_obj):
    if cap_obj: logger.info("[Executor] Releasing cv.VideoCapture object..."); cap_obj.release(); logger.info("[Executor] Capture released.")

async def initialize_stream_source(stream_type: str):
    global normal_video_cap, static_video_cap
    if not await initialize_yolo_model_async(): logger.error(f"Cannot initialize {stream_type} source: YOLO model failed."); return False
    is_static = stream_type == "static_video"; lock = static_video_lock if is_static else normal_video_lock
    async with lock:
        current_cap_obj_val = static_video_cap if is_static else normal_video_cap
        if current_cap_obj_val and current_cap_obj_val.isOpened(): logger.debug(f"{stream_type.capitalize()} source already initialized."); return True
        source_id_or_path = STATIC_VIDEO_PATH_STR if is_static else WEBCAM_INDEX
        is_file = is_static or (not is_static and WEBCAM_IS_VIDEO_FILE)
        timeout = STATIC_VIDEO_INIT_TIMEOUT if is_static else CAMERA_INIT_TIMEOUT
        desc = f"{stream_type.capitalize()} Source ({'File' if is_file else 'Webcam'})"
        logger.info(f"Requesting {desc} initialization (timeout: {timeout}s)...")
        loop = asyncio.get_running_loop()
        try:
            cap = await asyncio.wait_for( loop.run_in_executor(executor, _open_cv_capture_blocking, source_id_or_path, is_file), timeout=timeout)
            if cap:
                if is_static: static_video_cap = cap
                else: normal_video_cap = cap
                logger.info(f"{desc} initialization successful."); return True
            else: 
                if is_static: static_video_cap = None
                else: normal_video_cap = None
                logger.error(f"{desc} initialization failed (returned None)."); return False
        except asyncio.TimeoutError: 
            if is_static: static_video_cap = None
            else: normal_video_cap = None
            logger.error(f"{desc} initialization timed out."); return False
        except Exception as e: 
            if is_static: static_video_cap = None
            else: normal_video_cap = None
            logger.error(f"Unexpected error during {desc} init: {e}", exc_info=True); return False

async def release_stream_source(stream_type: str):
    global normal_video_cap, normal_video_task, static_video_cap, static_video_task
    is_static = stream_type == "static_video"; lock = static_video_lock if is_static else normal_video_lock
    desc = f"{stream_type.capitalize()} Source ({'File' if (is_static or (not is_static and WEBCAM_IS_VIDEO_FILE)) else 'Webcam'})"
    async with lock:
        cap_to_release = static_video_cap if is_static else normal_video_cap
        task_to_cancel = static_video_task if is_static else normal_video_task
        if is_static: static_video_cap, static_video_task = None, None
        else: normal_video_cap, normal_video_task = None, None
        if task_to_cancel and not task_to_cancel.done():
            logger.info(f"Cancelling {desc} broadcast task..."); task_to_cancel.cancel()
            try: await task_to_cancel
            except asyncio.CancelledError: logger.info(f"{desc} task cancelled.")
            except Exception as e: logger.error(f"Error awaiting cancelled {desc} task: {e}")
        if cap_to_release:
            logger.info(f"Requesting {desc} release via executor...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, _release_cv_capture_blocking, cap_to_release)
        logger.info(f"{desc} release process finished.")

def _read_frame_blocking(cap_obj):
    if cap_obj is None or not cap_obj.isOpened(): logger.debug("[Executor] Read frame: Capture object not open/None."); return None
    try: success, frame = cap_obj.read()
    except Exception as e: logger.error(f"[Executor] Exception during frame read: {e}", exc_info=True); return None
    if not success: logger.debug("[Executor] Frame read failed (cv2.read() returned False)."); return None
    return frame

def _process_frame_yolo_tracking_blocking(frame_bgr_input, frame_idx_for_tracking_state):
    global yolo_model, tracked_objects, next_track_id 
    if yolo_model is None: logger.warning("YOLO model not available for processing."); return frame_bgr_input, []
    processed_frame = frame_bgr_input.copy(); detections_for_tracker = []; raw_yolo_detections_count = 0
    try:
        effective_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = yolo_model(frame_bgr_input, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD, iou=YOLO_IOU_THRESHOLD_NMS, classes=YOLO_TARGET_CLASSES, device=effective_device)
    except Exception as e: logger.error(f"Error during YOLO inference: {e}", exc_info=True); return frame_bgr_input, []
    if results and len(results) > 0:
        res = results[0]; boxes_data = res.boxes
        for i in range(len(boxes_data.xyxy)):
            x1, y1, x2, y2 = map(int, boxes_data.xyxy[i]); confidence = float(boxes_data.conf[i]); class_id = int(boxes_data.cls[i])      
            label = f"R:{yolo_model.names[class_id]} {confidence:.1f}"
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 1) 
            cv2.putText(processed_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
            raw_yolo_detections_count += 1
            w, h = x2 - x1, y2 - y1; cx, cy = x1 + w // 2, y1 + h // 2
            detections_for_tracker.append({'bbox_center': (cx, cy), 'bbox_xywh': (x1, y1, w, h)})
    if raw_yolo_detections_count > 0: logger.info(f"Frame {frame_idx_for_tracking_state}: YOLO identified {raw_yolo_detections_count} target objects.")
    elif frame_idx_for_tracking_state % 60 == 0 : logger.info(f"Frame {frame_idx_for_tracking_state}: No target objects met criteria for drawing.") # Log less frequently if no detections

    current_tracks_output = []; temp_tracked_objects_next_state = {}; unmatched_detection_indices = list(range(len(detections_for_tracker)))
    for track_id, data in tracked_objects.items():
        data["frames_unseen"] += 1; best_match_det_idx_for_this_track = -1; min_dist_for_this_track = TRACKER_MAX_DIST
        for det_idx_ptr in unmatched_detection_indices:
            det_center = detections_for_tracker[det_idx_ptr]['bbox_center']
            dist = np.linalg.norm(np.array(data["bbox_center"]) - np.array(det_center))
            if dist < min_dist_for_this_track: min_dist_for_this_track = dist; best_match_det_idx_for_this_track = det_idx_ptr
        if best_match_det_idx_for_this_track != -1:
            matched_det_data = detections_for_tracker[best_match_det_idx_for_this_track]
            data["bbox_center"] = matched_det_data['bbox_center']; data["last_seen_frame"] = frame_idx_for_tracking_state; data["frames_unseen"] = 0
            data["bbox_history"].append(matched_det_data['bbox_xywh']); 
            if len(data["bbox_history"]) > 10: data["bbox_history"].pop(0)
            current_tracks_output.append({"id": track_id, "bbox_xywh": matched_det_data['bbox_xywh'], "center": matched_det_data['bbox_center']})
            unmatched_detection_indices.remove(best_match_det_idx_for_this_track); temp_tracked_objects_next_state[track_id] = data
        elif data["frames_unseen"] <= MAX_UNSEEN_FRAMES_TRACKER: temp_tracked_objects_next_state[track_id] = data
    for det_idx_ptr in unmatched_detection_indices:
        new_det_data = detections_for_tracker[det_idx_ptr]
        temp_tracked_objects_next_state[next_track_id] = { "bbox_center": new_det_data['bbox_center'], "last_seen_frame": frame_idx_for_tracking_state, "frames_unseen": 0, "bbox_history": [new_det_data['bbox_xywh']]}
        current_tracks_output.append({"id": next_track_id, "bbox_xywh": new_det_data['bbox_xywh'], "center": new_det_data['bbox_center']})
        next_track_id += 1
        if next_track_id > 20000: next_track_id = 0 
    tracked_objects.clear(); tracked_objects.update(temp_tracked_objects_next_state)
    if current_tracks_output: logger.debug(f"Frame {frame_idx_for_tracking_state}: Drawing {len(current_tracks_output)} final tracks.")
    for track_info in current_tracks_output:
        x, y, w, h = track_info["bbox_xywh"]; track_id_to_draw = track_info["id"]
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        cv2.putText(processed_frame, f"T:{track_id_to_draw}", (x, y - 25 if y > 25 else y + h + 15), cv2.FONT_HERSHEY_PLAIN, 0.9, (255,0,0), 1)
    return processed_frame, current_tracks_output

async def analyze_tracks_for_anomalies(tracks, frame_idx): alerts = []; return alerts
async def send_anomaly_alerts_to_clients(alerts):
    if not alerts: return
    current_clients_snapshot = list(CLIENTS)
    for alert_data in alerts:
        payload = json.dumps(alert_data)
        for client_ws in current_clients_snapshot: # Correctly indented loop
            if not client_ws.closed:
                try: await client_ws.send_str(payload)
                except Exception as e: logger.warning(f"Failed to send anomaly alert to {getattr(client_ws, 'addr_str', 'unknown')}: {e}")

async def check_and_manage_normal_video_task():
    global normal_video_task, normal_video_cap 
    if SUBSCRIPTIONS["normal_video"]:
        if normal_video_cap is None and not await initialize_stream_source(stream_type="normal_video"): return
        if normal_video_cap is not None and (normal_video_task is None or normal_video_task.done()):
            desc = "Normal Video File" if WEBCAM_IS_VIDEO_FILE else "Live Camera"
            logger.info(f"{desc} source ready. Starting broadcast (YOLO)...")
            normal_video_task = asyncio.create_task(broadcast_frames_loop(stream_type="normal_video"))
    elif normal_video_cap is not None or (normal_video_task and not normal_video_task.done()):
        await release_stream_source(stream_type="normal_video")
async def check_and_manage_static_video_task():
    global static_video_task, static_video_cap
    if SUBSCRIPTIONS["static_video"]:
        static_video_full_path = STATIC_FILES_DIR / STATIC_VIDEO_PATH_STR
        if not static_video_full_path.is_file(): logger.warning(f"Static video {static_video_full_path} not found."); return
        if static_video_cap is None and not await initialize_stream_source(stream_type="static_video"): return
        if static_video_cap is not None and (static_video_task is None or static_video_task.done()):
            logger.info("Static Vid source ready. Starting broadcast (YOLO)...")
            static_video_task = asyncio.create_task(broadcast_frames_loop(stream_type="static_video"))
    elif static_video_cap is not None or (static_video_task and not static_video_task.done()):
        await release_stream_source(stream_type="static_video")

async def broadcast_frames_loop(stream_type: str):
    global current_frame_index, tracked_objects, next_track_id, normal_video_cap, static_video_cap 
    is_static = stream_type == "static_video"; is_file_feed = is_static or (not is_static and WEBCAM_IS_VIDEO_FILE)
    source_name = "Static Video" if is_static else ("Normal Video File" if WEBCAM_IS_VIDEO_FILE else "Live Camera")
    logger.info(f"{source_name} broadcast task started (YOLO + Simple Tracking).")
    loop = asyncio.get_running_loop(); frame_count_log = 0; last_log_time = time.monotonic()
    if is_file_feed : tracked_objects = {}; next_track_id = 0
    while True:
        start_time_cycle = time.monotonic()
        cap_obj = static_video_cap if is_static else normal_video_cap
        subscription_key = "static_video" if is_static else "normal_video"
        if not SUBSCRIPTIONS[subscription_key]: logger.info(f"No subscribers for {source_name}, stopping task."); break
        if cap_obj is None or yolo_model is None: await asyncio.sleep(0.2); continue
        try:
            current_frame_color = await loop.run_in_executor(executor, _read_frame_blocking, cap_obj)
            if current_frame_color is None:
                if is_file_feed:
                    logger.info(f"End of {source_name} or read error. Attempting to loop/reopen...")
                    await release_stream_source(stream_type=stream_type) 
                    reinit_success = await initialize_stream_source(stream_type=stream_type)
                    if reinit_success:
                        logger.info(f"{source_name} re-initialized for looping.")
                        if is_file_feed: # Correctly indented block
                            tracked_objects = {} 
                            next_track_id = 0
                        continue 
                    else: logger.error(f"Failed to re-initialize {source_name}. Stopping task."); break
                else: logger.warning(f"Failed to capture frame from {source_name} (live webcam). Trying again..."); await asyncio.sleep(0.5); continue
            current_frame_index +=1 
            if len(current_frame_color.shape) == 2: current_frame_color = cv2.cvtColor(current_frame_color, cv2.COLOR_GRAY2BGR)
            elif current_frame_color.shape[2] == 4: current_frame_color = cv2.cvtColor(current_frame_color, cv2.COLOR_BGRA2BGR)
            resized_for_yolo = cv2.resize(current_frame_color, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            processed_frame_resized, tracks = await loop.run_in_executor( executor, _process_frame_yolo_tracking_blocking, resized_for_yolo, current_frame_index )
            if processed_frame_resized is None: logger.warning(f"Processing failed for {source_name} frame."); continue
            anomaly_alerts = await analyze_tracks_for_anomalies(tracks, current_frame_index)
            if anomaly_alerts: asyncio.create_task(send_anomaly_alerts_to_clients(anomaly_alerts))
            ret, buffer = cv2.imencode('.jpg', processed_frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ret: logger.warning(f"Failed to encode {source_name} frame."); continue
            frame_bytes = buffer.tobytes()
            active_subscribers = list(SUBSCRIPTIONS[subscription_key])
            if active_subscribers:
                send_tasks = [asyncio.create_task(safe_send_binary(ws, frame_bytes)) for ws in active_subscribers if not ws.closed]
                if send_tasks: await asyncio.gather(*send_tasks, return_exceptions=True)
            frame_count_log += 1
            if time.monotonic() - last_log_time >= 5.0:
                fps = frame_count_log / (time.monotonic() - last_log_time); cycle_time_ms = (time.monotonic() - start_time_cycle) * 1000
                logger.info(f"{source_name} FPS: {fps:.1f} (YOLO) (Cycle: {cycle_time_ms:.1f}ms, Frame Idx: {current_frame_index})")
                frame_count_log = 0; last_log_time = time.monotonic()
            elapsed_cycle = time.monotonic() - start_time_cycle
            await asyncio.sleep(max(0, FRAME_DELAY - elapsed_cycle))
        except asyncio.CancelledError: logger.info(f"{source_name} broadcast task cancelled."); break
        except Exception as e: logger.error(f"Error in {source_name} broadcast loop: {e}", exc_info=True); await asyncio.sleep(1)
    logger.info(f"{source_name} broadcast task finished.")

async def handle_websocket_message(ws, addr, message_str):
    try:
        data = json.loads(message_str)
        action = data.get("action"); stream = data.get("stream"); stream_updated = False
        target_stream_set = SUBSCRIPTIONS.get(stream)

        if target_stream_set is not None:
            if action == "subscribe":
                if ws not in target_stream_set:
                    other_stream_key = "static_video" if stream == "normal_video" else "normal_video"
                    # CORRECTED INDENTATION FOR THIS BLOCK
                    if ws in SUBSCRIPTIONS[other_stream_key]: 
                        SUBSCRIPTIONS[other_stream_key].discard(ws)
                        logger.info(f"Client {addr} auto-unsubscribed from '{other_stream_key}'.")
                        if other_stream_key == "normal_video": 
                            await check_and_manage_normal_video_task()
                        else: 
                            await check_and_manage_static_video_task()
                    target_stream_set.add(ws) # This was correctly indented
                    logger.info(f"Client {addr} subscribed to '{stream}'. Total subs: {len(target_stream_set)}")
                    stream_updated = True
            elif action == "unsubscribe":
                 if ws in target_stream_set:
                    target_stream_set.discard(ws)
                    logger.info(f"Client {addr} unsubscribed from '{stream}'. Total subs: {len(target_stream_set)}")
                    stream_updated = True
            else: 
                logger.warning(f"Unknown action '{action}' for stream '{stream}' from {addr}")
        else: 
            logger.warning(f"Unknown stream '{stream}' from {addr}: {data if isinstance(data, dict) else str(data)[:50]}")

        if stream_updated:
            if stream == "normal_video": 
                await check_and_manage_normal_video_task()
            elif stream == "static_video": 
                await check_and_manage_static_video_task()
    except json.JSONDecodeError: 
        logger.error(f"Non-JSON message from {addr}: {message_str}")
    except Exception as e: 
        logger.error(f"Error processing message '{message_str}' from {addr}: {e}", exc_info=True)

async def websocket_handler(request):
    addr_str = str(request.remote) if request.remote else "unknown_ws_client"; ws = web.WebSocketResponse();
    if not ws.can_prepare(request).ok: logger.error(f"WebSocket preparation failed for {addr_str}"); return web.Response(status=400, text="WebSocket upgrade failed")
    await ws.prepare(request); logger.info(f"WebSocket connection established from {addr_str}"); CLIENTS.add(ws); ws.addr_str = addr_str;
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT: await handle_websocket_message(ws, addr_str, msg.data)
            elif msg.type == web.WSMsgType.BINARY: logger.warning(f"Unexpected binary WebSocket message from {addr_str}.")
            elif msg.type == web.WSMsgType.ERROR: logger.error(f'WebSocket connection for {addr_str} closed with exception {ws.exception()}')
    except asyncio.CancelledError: logger.info(f"WebSocket task for {addr_str} cancelled."); raise 
    except Exception as e: logger.error(f"Error in WebSocket handler for {addr_str}: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket connection closing for {addr_str}"); CLIENTS.discard(ws);
        if ws in SUBSCRIPTIONS["normal_video"]: 
            SUBSCRIPTIONS["normal_video"].discard(ws); logger.info(f"Client {addr_str} removed from 'normal_video'."); 
            asyncio.create_task(check_and_manage_normal_video_task());
        if ws in SUBSCRIPTIONS["static_video"]: 
            SUBSCRIPTIONS["static_video"].discard(ws); logger.info(f"Client {addr_str} removed from 'static_video'."); 
            asyncio.create_task(check_and_manage_static_video_task());
    return ws

async def safe_send_binary(ws, data_bytes):
    addr = getattr(ws, 'addr_str', 'unknown_client_for_send');
    try: await ws.send_bytes(data_bytes);
    except (ConnectionResetError, asyncio.CancelledError, RuntimeError) as e:
        logger.warning(f"Client {addr} disconnected or error during send: {type(e).__name__}"); CLIENTS.discard(ws); 
        if ws in SUBSCRIPTIONS["normal_video"]: 
            SUBSCRIPTIONS["normal_video"].discard(ws); asyncio.create_task(check_and_manage_normal_video_task());
        if ws in SUBSCRIPTIONS["static_video"]: 
            SUBSCRIPTIONS["static_video"].discard(ws); asyncio.create_task(check_and_manage_static_video_task());
    except Exception as e:
        logger.error(f"Unexpected error sending frame to {addr}: {e}", exc_info=True); CLIENTS.discard(ws);
        if ws in SUBSCRIPTIONS["normal_video"]: 
            SUBSCRIPTIONS["normal_video"].discard(ws); asyncio.create_task(check_and_manage_normal_video_task());
        if ws in SUBSCRIPTIONS["static_video"]: 
            SUBSCRIPTIONS["static_video"].discard(ws); asyncio.create_task(check_and_manage_static_video_task());

async def handle_index(request): return web.FileResponse(STATIC_FILES_DIR / 'plan.html')

async def handle_static(request):
    req_path_str = request.match_info.get('filename', '')
    if not req_path_str: return web.Response(status=404, text="File not specified")
    try:
        if ".." in req_path_str: return web.Response(status=403, text="Forbidden Path Traversal")
        file_to_serve = None
        path_in_root = (STATIC_FILES_DIR / req_path_str).resolve()
        if path_in_root.is_file() and str(path_in_root).startswith(str(STATIC_FILES_DIR.resolve())) and path_in_root.parent == STATIC_FILES_DIR.resolve():
            file_to_serve = path_in_root
        if not file_to_serve: # Check static subdir only if not found in root or not allowed from other root subdirs
            path_in_static_subdir = (STATIC_FILES_DIR / "static" / req_path_str).resolve()
            if path_in_static_subdir.is_file() and str(path_in_static_subdir).startswith(str((STATIC_FILES_DIR / "static").resolve())):
                 file_to_serve = path_in_static_subdir
        
        if not file_to_serve:
            logger.debug(f"File not found or access denied for: {req_path_str}")
            return web.Response(status=404, text=f"'{req_path_str}' not found or access denied")

        content_type = None
        if file_to_serve.suffix == '.css': content_type = 'text/css'
        elif file_to_serve.suffix == '.js': content_type = 'application/javascript'
        elif file_to_serve.suffix == '.mp4': content_type = 'video/mp4'
        elif file_to_serve.suffix == '.svg': content_type = 'image/svg+xml'
        elif file_to_serve.suffix == '.ico': content_type = 'image/x-icon'
        return web.FileResponse(file_to_serve, headers={'Content-Type': content_type} if content_type else None)

    except FileNotFoundError: 
        logger.debug(f"File not found by handle_static (FileNotFoundError): {req_path_str}")
        return web.Response(status=404, text=f"'{req_path_str}' not found")
    except ValueError as e: 
        logger.warning(f"Bad static file request '{req_path_str}': {e}")
        return web.Response(status=400, text="Bad Request - Invalid Path")
    except Exception as e: 
        logger.error(f"Static file error for '{req_path_str}': {e}", exc_info=True)
        return web.Response(status=500, text="Internal Server Error")

async def on_app_startup(app_instance):
    logger.info("Application starting up..."); 
    asyncio.create_task(initialize_yolo_model_async())
    logger.info("YOLO model initialization process initiated.")
    static_video_full_path = STATIC_FILES_DIR / STATIC_VIDEO_PATH_STR
    if not static_video_full_path.is_file(): logger.warning(f"Static video file not found: {static_video_full_path}. Static streaming may fail.")
    else: logger.info(f"Static video file found: {static_video_full_path}")
async def on_app_cleanup(app_instance):
    logger.info("Application shutting down, cleaning up resources..."); tasks_to_cancel = []
    global normal_video_task, static_video_task 
    if normal_video_task and not normal_video_task.done(): tasks_to_cancel.append(normal_video_task)
    if static_video_task and not static_video_task.done(): tasks_to_cancel.append(static_video_task)
    for task in tasks_to_cancel:
        task_name = getattr(task, 'get_name', lambda: 'Unnamed Task')(); logger.info(f"Cancelling task {task_name} during cleanup...")
        task.cancel();
        try: await task
        except asyncio.CancelledError: logger.info(f"Task {task_name} successfully cancelled.")
        except Exception as e: logger.error(f"Error during cancellation of task {task_name}: {e}")
    await release_stream_source(stream_type="normal_video"); await release_stream_source(stream_type="static_video");
    logger.info("Cleanup: Shutting down thread pool executor..."); executor.shutdown(wait=True, cancel_futures=True); logger.info("Cleanup: Executor shut down.")

async def main():
     app = web.Application(logger=logger)
     app.on_startup.append(on_app_startup)
     app.on_cleanup.append(on_app_cleanup)
     app.router.add_get('/ws', websocket_handler); app.router.add_get('/', handle_index);
     favicon_path = STATIC_FILES_DIR / 'static' / 'favicon.ico' 
     if favicon_path.is_file(): app.router.add_get('/favicon.ico', lambda r: web.FileResponse(favicon_path))
     else: logger.warning(f"favicon.ico not found at {favicon_path}, will 404."); app.router.add_get('/favicon.ico', lambda r: web.Response(status=404, text="Favicon not found"))
     app.router.add_get('/{filename:.+}', handle_static); 
     runner = web.AppRunner(app); await runner.setup(); site = web.TCPSite(runner, HOST, PORT);
     logger.info(f"-----------------------------------------------------"); logger.info(f"Starting PoultryScope Server with YOLO Object Detection"); logger.info(f"Using YOLO model source: {YOLO_MODEL_PATH}"); logger.info(f"Normal Stream source: {WEBCAM_INDEX}");
     static_video_full_path = STATIC_FILES_DIR / STATIC_VIDEO_PATH_STR
     if static_video_full_path.is_file(): logger.info(f"Static Stream source: {static_video_full_path.name}")
     else: logger.warning(f"Static Video ({static_video_full_path.name}) NOT FOUND - streaming disabled.")
     logger.info(f"Serving on http://{HOST}:{PORT} (and ws://{HOST}:{PORT}/ws)"); logger.info(f"Find your local IP if needed to access from other devices on your network."); logger.info(f"Press Ctrl+C to stop."); logger.info(f"-----------------------------------------------------");
     await site.start();
     try:
         while True: await asyncio.sleep(3600) 
     except (KeyboardInterrupt, asyncio.CancelledError): logger.info("Shutdown signal received by main loop.")
     finally: logger.info("Main loop ending. Site and runner cleanup will proceed via on_app_cleanup.")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("Application shutting down (KeyboardInterrupt in __main__).")
    except Exception as e: logger.critical(f"Unhandled exception in __main__: {e}", exc_info=True)
    finally: logger.info("Application process definitively ending.")
# --- END OF FILE server_websocket.py ---
