"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Real-time homography estimation demo. Note that scene has to be planar or just rotate the camera for the estimation to work properly.
"""

import cv2
import numpy as np
import torch

from time import time, sleep
import argparse, sys, tqdm
import threading

from modules.xfeat import XFeat

def argparser():
    parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
    parser.add_argument('--width', type=int, default=640, help='Width of the video capture stream.')
    parser.add_argument('--height', type=int, default=480, help='Height of the video capture stream.')
    parser.add_argument('--max_kpts', type=int, default=3_000, help='Maximum number of keypoints.')
    parser.add_argument('--method', type=str, choices=['ORB', 'SIFT', 'XFeat'], default='XFeat', help='Local feature detection method to use.')
    parser.add_argument('--cam', type=int, default=0, help='Webcam device number.')
    return parser.parse_args()


class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        _, self.frame = self.cap.read()
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream ended?).")
            self.frame = frame
            sleep(0.01)

    def stop(self):
        self.running = False
        self.cap.release()

    def get_last_frame(self):
        return self.frame

class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(torch.tensor(x).permute(2,0,1).float()[None])[0]

class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher

def init_method(method, max_kpts):
    if method == "ORB":
        return Method(descriptor=cv2.ORB_create(max_kpts, fastThreshold=10), matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True))
    elif method == "SIFT":
        return Method(descriptor=cv2.SIFT_create(max_kpts, contrastThreshold=-1, edgeThreshold=1000), matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True))
    elif method == "XFeat":
        return Method(descriptor=CVWrapper(XFeat(top_k = max_kpts)), matcher=XFeat())
    else:
        raise RuntimeError("Invalid Method.")


class MatchingDemo:
    def __init__(self, args):
        self.args = args
        self.cap = cv2.VideoCapture(args.cam)
        self.width = args.width
        self.height = args.height
        self.ref_frame = None
        self.ref_precomp = [[],[]]
        self.corners = [[50, 50], [640-50, 50], [640-50, 480-50], [50, 480-50]]
        self.current_frame = None
        self.H = None
        self.setup_camera()

        #Init frame grabber thread
        self.frame_grabber = FrameGrabber(self.cap)
        self.frame_grabber.start()

        #Homography params
        self.min_inliers = 50
        self.ransac_thr = 4.0

        #FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 30 #avg FPS over this number of frames

        #Set local feature method here -- we expect cv2 or Kornia convention
        self.method = init_method(args.method, max_kpts=args.max_kpts)
        
        # Setting up font for captions
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0,255,0)
        self.line_thickness = 3

        self.window_name = "Real-time matching - Press 's' to set the reference frame."

        # Removes toolbar and status bar
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.width*2, self.height*2)
        #Set Mouse Callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def draw_quad(self, frame, point_list):
        if len(self.corners) > 1:
            for i in range(len(self.corners) - 1):
                cv2.line(frame, tuple(point_list[i]), tuple(point_list[i + 1]), self.line_color, self.line_thickness, lineType = self.line_type)
            if len(self.corners) == 4:  # Close the quadrilateral if 4 corners are defined
                cv2.line(frame, tuple(point_list[3]), tuple(point_list[0]), self.line_color, self.line_thickness, lineType = self.line_type)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) >= 4:
                self.corners = []  # Reset corners if already 4 points were clicked
            self.corners.append((x, y))

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # Draw the border
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=borderColor, thickness=thickness+2, lineType=lineType)
        # Draw the text
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=textColor, thickness=thickness, lineType=lineType)

    def warp_points(self, points, H, x_offset = 0):
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)

        warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
        warped_points_np[:, 0] += x_offset
        warped_points = warped_points_np.astype(int).tolist()
        
        return warped_points

    def create_top_frame(self):
        top_frame_canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
        top_frame = np.hstack((self.ref_frame, self.current_frame))
        color = (3, 186, 252)
        cv2.rectangle(top_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)  # Orange color line as a separator
        top_frame_canvas[0:self.height, 0:self.width*2] = top_frame
        
        # Adding captions on the top frame canvas
        self.putText(canvas=top_frame_canvas, text="Reference Frame:", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        self.putText(canvas=top_frame_canvas, text="Target Frame:", org=(650, 30), fontFace=self.font, 
                    fontScale=self.font_scale,  textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
        self.draw_quad(top_frame_canvas, self.corners)
        
        return top_frame_canvas

    def process(self):
        # Create a blank canvas for the top frame
        top_frame_canvas = self.create_top_frame()

        # Match features and draw matches on the bottom frame
        bottom_frame = self.match_and_draw(self.ref_frame, self.current_frame)

        # Draw warped corners
        if self.H is not None and len(self.corners) > 1:
            self.draw_quad(top_frame_canvas, self.warp_points(self.corners, self.H, self.width))

        # Stack top and bottom frames vertically on the final canvas
        canvas = np.vstack((top_frame_canvas, bottom_frame))

        cv2.imshow(self.window_name, canvas)

    def match_and_draw(self, ref_frame, current_frame):

        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        # Detect and compute features
        if self.args.method in ['SIFT', 'ORB']:
            kp1, des1 = self.ref_precomp
            kp2, des2 = self.method.descriptor.detectAndCompute(current_frame, None)
        else:
            current = self.method.descriptor.detectAndCompute(current_frame)
            kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
            kpts2, descs2 = current['keypoints'], current['descriptors']
            idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
            points1 = kpts1[idx0].cpu().numpy()
            points2 = kpts2[idx1].cpu().numpy()

        if len(kp1) > 10 and len(kp2) > 10 and self.args.method in ['SIFT', 'ORB']:
            # Match descriptors
            matches = self.method.matcher.match(des1, des2)

            if len(matches) > 10:
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                for i, match in enumerate(matches):
                    points1[i, :] = kp1[match.queryIdx].pt
                    points2[i, :] = kp2[match.trainIdx].pt

        if len(points1) > 10 and len(points2) > 10:
            # Find homography
            self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
            inliers = inliers.flatten() > 0

            if inliers.sum() < self.min_inliers:
                self.H = None

            if self.args.method in ["SIFT", "ORB"]:
                good_matches = [m for i,m in enumerate(matches) if inliers[i]]
            else:
                kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
                kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
                good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

            # Draw matches
            matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
            
        else:
            matched_frame = np.hstack([ref_frame, current_frame])

        color = (240, 89, 169)

        # Add a colored rectangle to separate from the top frame
        cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)

        # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="%s Matches: %d"%(self.args.method, len(good_matches)), org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
                # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="FPS (registration): {:.1f}".format(self.FPS), org=(650, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        return matched_frame

    def main_loop(self):
        self.current_frame = self.frame_grabber.get_last_frame()
        self.ref_frame = self.current_frame.copy()
        self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

        while True:
            if self.current_frame is None:
                break

            t0 = time()
            self.process()

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.ref_frame = self.current_frame.copy()  # Update reference frame
                self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

            self.current_frame = self.frame_grabber.get_last_frame()

            #Measure avg. FPS
            self.time_list.append(time()-t0)
            if len(self.time_list) > self.max_cnt:
                self.time_list.pop(0)
            self.FPS = 1.0 / np.array(self.time_list).mean()
        
        self.cleanup()

    def cleanup(self):
        self.frame_grabber.stop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = MatchingDemo(args = argparser())
    demo.main_loop()
