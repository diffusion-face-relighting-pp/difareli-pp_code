import argparse
import glob
import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image
import tqdm

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Extract pupil masks using MediaPipe and combine with face segmentation.')
parser.add_argument('--image_dir', required=True, help='Input aligned images directory')
parser.add_argument('--faceseg_dir', required=True, help='Input face_segment anno directory (contains anno_*.png)')
parser.add_argument('--output_path', required=True, help='Output face_segment_with_pupil anno directory')
parser.add_argument(
    '--model_path',
    default=os.path.join(SCRIPT_DIR, 'face_landmarker_v2_with_blendshapes.task'),
    help='Path to MediaPipe face_landmarker_v2_with_blendshapes.task',
)
parser.add_argument('--vis', action='store_true', default=False, help='Visualize results')


def create_image_grid(images, n_rows=1, text_labels=[]):
    n_images = len(images)
    n_cols = (n_images + n_rows - 1) // n_rows
    images += [np.zeros_like(images[0]) for _ in range(n_rows * n_cols - n_images)]

    rows = []
    for i in range(n_rows):
        row_images = images[i * n_cols:(i + 1) * n_cols]
        rows.append(np.concatenate(row_images, axis=1))

    grid_image = np.concatenate(rows, axis=0)

    img_height, img_width = images[0].shape[:2]
    for i, text_label in enumerate(text_labels):
        if isinstance(text_label, str):
            plt.text((i % n_cols) * img_width, -2, text_label, color='red', fontsize=20)
        else:
            plt.text((i % n_cols) * img_width, -2, f'{float(text_label):.5f}', color='red', fontsize=20)

    return grid_image


def face_segment(segment_part, img):
    if isinstance(img, Image.Image):
        face_segment_anno = np.array(img)
    else:
        face_segment_anno = img

    bg = (face_segment_anno == 0)
    skin = (face_segment_anno == 1)
    l_brow = (face_segment_anno == 2)
    r_brow = (face_segment_anno == 3)
    l_eye = (face_segment_anno == 4)
    r_eye = (face_segment_anno == 5)
    eye_g = (face_segment_anno == 6)
    l_ear = (face_segment_anno == 7)
    r_ear = (face_segment_anno == 8)
    ear_r = (face_segment_anno == 9)
    nose = (face_segment_anno == 10)
    mouth = (face_segment_anno == 11)
    u_lip = (face_segment_anno == 12)
    l_lip = (face_segment_anno == 13)
    neck = (face_segment_anno == 14)
    neck_l = (face_segment_anno == 15)
    cloth = (face_segment_anno == 16)
    hair = (face_segment_anno == 17)
    hat = (face_segment_anno == 18)
    l_pupil = (face_segment_anno == 19)
    r_pupil = (face_segment_anno == 20)
    face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip))

    if segment_part == 'faceseg_face':
        seg_m = face
    elif segment_part == 'faceseg_head':
        seg_m = (face | neck | hair)
    elif segment_part == 'faceseg_nohead':
        seg_m = ~(face | neck | hair)
    elif segment_part == 'faceseg_face&hair':
        seg_m = ~bg
    elif segment_part == 'faceseg_bg_noface&nohair':
        seg_m = (bg | hat | neck | neck_l | cloth)
    elif segment_part == 'faceseg_bg&ears_noface&nohair':
        seg_m = (bg | hat | neck | neck_l | cloth) | (l_ear | r_ear | ear_r)
    elif segment_part == 'faceseg_bg':
        seg_m = bg
    elif segment_part == 'faceseg_bg&noface':
        seg_m = (bg | hair | hat | neck | neck_l | cloth)
    elif segment_part == 'faceseg_faceskin':
        seg_m = skin
    elif segment_part == 'faceseg_faceskin&nose':
        seg_m = (skin | nose)
    elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows':
        seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye)
    elif segment_part == 'faceseg_faceskin&nose&mouth&eyebrows&eyes&glasses':
        seg_m = (skin | nose | mouth | u_lip | l_lip | l_brow | r_brow | l_eye | r_eye | eye_g)
    elif segment_part == 'faceseg_face_noglasses':
        seg_m = (~eye_g & face)
    elif segment_part == 'faceseg_face_noglasses_noeyes':
        seg_m = (~(l_eye | r_eye) & ~eye_g & face)
    elif segment_part == 'faceseg_eyes&glasses':
        seg_m = (l_eye | r_eye | eye_g)
    elif segment_part == 'faceseg_glasses':
        seg_m = eye_g
    elif segment_part == 'faceseg_hair':
        seg_m = hair
    elif segment_part == 'faceseg_eyes':
        seg_m = (l_eye | r_eye | l_pupil | r_pupil)
    elif segment_part == 'faceseg_Leye':
        seg_m = l_eye
    elif segment_part == 'faceseg_Reye':
        seg_m = r_eye
    elif segment_part == 'faceseg_ears':
        seg_m = (l_ear | r_ear | ear_r)
    elif segment_part == 'faceseg_nose':
        seg_m = nose
    elif segment_part == 'faceseg_mouth':
        seg_m = (mouth | u_lip | l_lip)
    elif segment_part == 'faceseg_neck':
        seg_m = neck
    elif segment_part == 'faceseg_pupils':
        seg_m = (l_pupil | r_pupil)
    elif segment_part == 'faceseg_eyes&ears&nose&mouth':
        seg_m = (l_eye | r_eye | l_ear | r_ear | ear_r | nose | mouth | u_lip | l_lip)
    elif segment_part in ['sobel_bg_mask', 'laplacian_bg_mask', 'sobel_bin_bg_mask']:
        seg_m = ~(face | neck | hair)
    elif segment_part in ['canny_edge_bg_mask']:
        seg_m = ~(face | neck | hair) | (l_ear | r_ear)
    else:
        raise NotImplementedError(f"Segment part: {segment_part} is not found!")

    return seg_m


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def find_eyes(img_path, faceseg_dir, detector, vis=False):
    name = os.path.splitext(os.path.basename(img_path))[0]
    image_mp = mp.Image.create_from_file(img_path)
    image = np.array(Image.open(img_path))
    mask_path = os.path.join(faceseg_dir, f'anno_{name}.png')
    mask = np.array(Image.open(mask_path))
    if len(mask.shape) == 3:
        mask = mask[..., 0]

    H, W, _ = image.shape

    detection_result = detector.detect(image_mp)
    if len(detection_result.face_landmarks) < 1:
        print(f"[#] No face detected in {img_path}")
        l_eye_mask = face_segment('faceseg_Leye', mask)
        r_eye_mask = face_segment('faceseg_Reye', mask)
        new_mask = mask.copy()
        new_mask[l_eye_mask] = 19  # L_pupil set to l_eye if no face detected
        new_mask[r_eye_mask] = 20  # R_pupil set to r_eye if no face detected
        if vis:
            plt.imshow(create_image_grid([image]))
            plt.show()
            plt.imshow(new_mask)
            plt.show()
        return new_mask

    face_landmarks_proto_denom = list(
        solutions.drawing_utils._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_height=H, image_width=W)
        for landmark in detection_result.face_landmarks[0]
    )
    face_landmarks_proto_denom = np.array(face_landmarks_proto_denom)
    if isinstance(face_landmarks_proto_denom[0], tuple):
        print(f"[#] None detected in {img_path}")
        x = np.array([
            i[0] if i is not None and idx not in [468, 469, 470, 471, 472, 473, 474, 475, 476, 477] else 0.0
            for idx, i in enumerate(face_landmarks_proto_denom)
        ])
        y = np.array([
            i[1] if i is not None and idx not in [468, 469, 470, 471, 472, 473, 474, 475, 476, 477] else 0.0
            for idx, i in enumerate(face_landmarks_proto_denom)
        ])
        face_landmarks_proto_denom = np.concatenate([x[:, None], y[:, None]], axis=1)

    parts = {
        'l_eye': [468, 469, 470, 471, 472],
        'r_eye': [473, 474, 475, 476, 477],
        'l_pupil': [468],
        'r_pupil': [473],
        'l_rad': np.mean(np.sqrt(np.sum((face_landmarks_proto_denom[468:469] - face_landmarks_proto_denom[469:472]) ** 2, 1)), axis=0),
        'r_rad': np.mean(np.sqrt(np.sum((face_landmarks_proto_denom[473:474] - face_landmarks_proto_denom[474:477]) ** 2, 1)), axis=0),
    }

    L_pupil_mask = np.zeros_like(image)
    R_pupil_mask = np.zeros_like(image)
    cv2.circle(L_pupil_mask, (int(face_landmarks_proto_denom[468][0]), int(face_landmarks_proto_denom[468][1])), int(parts['l_rad']), (255, 255, 255), -1)
    cv2.circle(R_pupil_mask, (int(face_landmarks_proto_denom[473][0]), int(face_landmarks_proto_denom[473][1])), int(parts['r_rad']), (255, 255, 255), -1)
    L_pupil_mask = L_pupil_mask > 128
    R_pupil_mask = R_pupil_mask > 128
    pupil_mask = L_pupil_mask | R_pupil_mask

    # Add mask=19 for L_pupil_mask and mask=20 for R_pupil_mask
    if vis:
        image_vis = image.copy()
        for part in ['l_eye', 'r_eye']:
            for lm in face_landmarks_proto_denom[parts[part]]:
                cv2.circle(image_vis, (int(lm[0]), int(lm[1])), 2, (0, 0, 255), -1)
        cv2.circle(image_vis, (int(face_landmarks_proto_denom[468][0]), int(face_landmarks_proto_denom[468][1])), int(parts['l_rad']), (0, 255, 0), 1)
        cv2.circle(image_vis, (int(face_landmarks_proto_denom[473][0]), int(face_landmarks_proto_denom[473][1])), int(parts['r_rad']), (0, 255, 0), 1)

        annotated_image = draw_landmarks_on_image(image_mp.numpy_view(), detection_result)

        plt.figure(figsize=(50, 60))
        plt.imshow(create_image_grid([
            image,
            image * face_segment('faceseg_eyes', mask)[..., None],
            image * pupil_mask,
            image * (pupil_mask * face_segment('faceseg_eyes', mask)[..., None]),
            L_pupil_mask * face_segment('faceseg_eyes', mask)[..., None] * 255,
            R_pupil_mask * face_segment('faceseg_eyes', mask)[..., None] * 255,
            pupil_mask * 255,
            np.repeat(face_segment('faceseg_eyes', mask)[..., None], 3, axis=-1) * 255,
            pupil_mask * face_segment('faceseg_eyes', mask)[..., None] * 255,
            image_vis,
            annotated_image,
        ]))
        plt.show()

    new_mask = mask.copy()
    L_pupil_mask = L_pupil_mask[..., 0] * face_segment('faceseg_eyes', mask)
    R_pupil_mask = R_pupil_mask[..., 0] * face_segment('faceseg_eyes', mask)
    new_mask[L_pupil_mask] = 19
    new_mask[R_pupil_mask] = 20
    if vis:
        plt.imshow(new_mask)
        plt.show()

    return new_mask


if __name__ == '__main__':
    args = parser.parse_args()

    base_options = python.BaseOptions(model_asset_path=args.model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    os.makedirs(args.output_path, exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(args.image_dir, '*.png')) + glob.glob(os.path.join(args.image_dir, '*.jpg')))
    for img_path in tqdm.tqdm(img_files):
        name = os.path.splitext(os.path.basename(img_path))[0]
        new_mask = find_eyes(img_path, args.faceseg_dir, detector, vis=args.vis)
        new_mask_img = Image.fromarray(new_mask)
        new_mask_img.save(os.path.join(args.output_path, f'anno_{name}.png'))
