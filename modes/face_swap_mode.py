import cv2 as cv
from utils.face_analyzer import FaceAnalyzer
from utils.effects import (
    apply_filter,
    add_overlay,
    create_face_overlay,
    EffectType,
    color_transfer,
    blend_with_original,
)


def face_swapper(cap, cam):
    print('Face Swap Mode')
    print('Controls:')
    print('  0         - Face swap OFF (apply effects only)')
    print('  [ / ]     - Select source face (enables face swap)')
    print('  e         - Toggle effect')
    print('  +/-       - Effect intensity')
    print('  7 / 8     - Preview window size')
    print('  < / >     - Performance (skip frames)')
    print('  c         - Toggle color correction')
    print('  b / B     - Adjust blend ratio (0 = off, 1.0 = full blend)')
    print('  s         - Show/hide source face overlay')
    print('  q         - Quit')

    analyzer = FaceAnalyzer()

    import glob

    face_images = {}
    png_files = sorted(glob.glob('./resources/*.png'))
    for i, png_file in enumerate(png_files):
        face_name = f'face_{i}'
        face_images[face_name] = png_file

    analyzer.preload_source_faces(face_images)

    available_faces = analyzer.get_source_faces_list()
    if not available_faces:
        print('Error: No source faces loaded!')
        return

    print(f'Loaded {len(available_faces)} source faces')
    print('Effects: none, sharpen, cartoon, vignette, pixelate, edge, hue')
    print()

    current_source = available_faces[0]
    effects_order = [
        EffectType.NONE,
        EffectType.SHARPEN,
        EffectType.CARTOON,
        EffectType.VIGNETTE,
        EffectType.PIXELATE,
        EffectType.EDGE_DETECTION,
        EffectType.HUE_SHIFT,
    ]
    current_effect_index = 0
    effect_intensity = 1.0
    show_overlay = False
    use_color_correction = False
    blend_ratio = 0.0
    swap_mode = False

    current_effect_type = effects_order[current_effect_index]
    frame_count = 0
    skip_frames = 2

    preview_scale = 0.25

    last_processed_frame = None

    source_face_img = None

    def update_source_overlay():
        nonlocal source_face_img
        source_face = analyzer.get_source_face(current_source)
        if source_face:
            source_img = cv.imread(face_images[current_source])
            if source_img is not None:
                bbox = source_face.bbox.astype(int)
                face_crop = source_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                if face_crop.size > 0:
                    source_face_img = create_face_overlay(face_crop, current_source)

    update_source_overlay()

    print(f'Current source: {current_source}')
    print(f'Current effect: {current_effect_type}')
    print(f'Skip frames: {skip_frames}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if swap_mode:
            processed_frame = analyzer.swap_faces_in_frame(
                frame, current_source, skip_frames=skip_frames, frame_count=frame_count
            )

            if processed_frame is not None and processed_frame is not frame:
                if use_color_correction:
                    try:
                        processed_frame = color_transfer(processed_frame, frame)
                    except:
                        pass

                if blend_ratio > 0:
                    processed_frame = blend_with_original(
                        frame, processed_frame, blend_ratio
                    )

                last_processed_frame = processed_frame.copy()

            if processed_frame is None and last_processed_frame is not None:
                processed_frame = last_processed_frame
            elif processed_frame is None:
                processed_frame = frame
        else:
            processed_frame = frame
            last_processed_frame = None

        if current_effect_type != EffectType.NONE:
            processed_frame = apply_filter(
                processed_frame, current_effect_type, effect_intensity
            )

        if show_overlay and source_face_img is not None and swap_mode:
            h, w = processed_frame.shape[:2]
            overlay_size = (min(150, w // 6), min(150, h // 6))
            processed_frame = add_overlay(
                processed_frame,
                source_face_img,
                position=(w - overlay_size[0] - 10, 10),
                size=overlay_size,
                alpha=0.8,
            )

        swap_status = f'Swap: {"ON" if swap_mode else "OFF"}'
        cv.putText(
            processed_frame,
            swap_status,
            (10, 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if swap_mode else (0, 255, 255),
            2,
        )

        if swap_mode:
            cv.putText(
                processed_frame,
                f'Source: {current_source}',
                (10, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv.putText(
                processed_frame,
                f'Effect: {current_effect_type}',
                (10, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            cv.putText(
                processed_frame,
                f'Effect: {current_effect_type}',
                (10, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        color_status = f'Color: {"ON" if use_color_correction else "OFF"}'
        cv.putText(
            processed_frame,
            color_status,
            (10, 125),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv.putText(
            processed_frame,
            f'Intensity: {effect_intensity:.1f}',
            (10, 150),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        preview_frame = cv.resize(
            processed_frame,
            None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv.INTER_AREA,
        )
        cv.imshow('Face Swap', preview_frame)

        key = cv.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('0'):
            swap_mode = False
            last_processed_frame = None
            print(f'Face swap: OFF')
        elif key == ord(']'):
            current_index = available_faces.index(current_source)
            next_index = (current_index + 1) % len(available_faces)
            current_source = available_faces[next_index]
            update_source_overlay()
            swap_mode = True
            last_processed_frame = None
            print(f'Source: {current_source} (Face swap ON)')
        elif key == ord('['):
            current_index = available_faces.index(current_source)
            prev_index = (current_index - 1) % len(available_faces)
            current_source = available_faces[prev_index]
            update_source_overlay()
            swap_mode = True
            last_processed_frame = None
            print(f'Source: {current_source} (Face swap ON)')
        elif key == ord('e'):
            current_effect_index = (current_effect_index + 1) % len(effects_order)
            current_effect_type = effects_order[current_effect_index]
            last_processed_frame = None
            print(f'Effect: {current_effect_type}')
        elif key == ord('+') or key == ord('='):
            effect_intensity = min(3.0, effect_intensity + 0.1)
            print(f'Intensity: {effect_intensity:.1f}')
        elif key == ord('-') or key == ord('_'):
            effect_intensity = max(0.1, effect_intensity - 0.1)
            print(f'Intensity: {effect_intensity:.1f}')
        elif key == ord('s'):
            show_overlay = not show_overlay
            print(f'Overlay: {"visible" if show_overlay else "hidden"}')
        elif key == ord('8'):
            preview_scale = min(1.0, preview_scale + 0.1)
            print(f'Preview scale: {preview_scale:.1f}')
        elif key == ord('7'):
            preview_scale = max(0.2, preview_scale - 0.1)
            print(f'Preview scale: {preview_scale:.1f}')
        elif key == ord('>') or key == ord('.'):
            skip_frames = min(10, skip_frames + 1)
            print(f'Skip frames: {skip_frames}')
        elif key == ord('<') or key == ord(','):
            skip_frames = max(0, skip_frames - 1)
            print(f'Skip frames: {skip_frames}')
        elif key == ord('c'):
            use_color_correction = not use_color_correction
            print(f'Color correction: {"ON" if use_color_correction else "OFF"}')
            last_processed_frame = None
        elif key == ord('b'):
            blend_ratio = max(0.0, min(1.0, blend_ratio - 0.05))
            print(f'Blend ratio: {blend_ratio:.2f}')
            last_processed_frame = None
        elif key == ord('B'):
            blend_ratio = max(0.0, min(1.0, blend_ratio + 0.05))
            print(f'Blend ratio: {blend_ratio:.2f}')
            last_processed_frame = None

        if cam:
            cam.send(cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()
