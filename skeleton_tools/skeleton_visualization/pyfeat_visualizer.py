from skeleton_tools.skeleton_visualization.draw_utils import draw_bbox


class PyfeatVisualizer:
    def draw_facebox(self, frame, faces):
        for _, face in faces.iterrows():
            bbox = face[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']].astype(int).values.reshape(2, 2)
            color = (255, 0, 0) if face['is_child'] else (0, 0, 255)
            draw_bbox(frame, bbox, color)
        return frame