#!/usr/bin/env python3

import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


#################################################################
#################################################################
##
##  This software is Copyright 2018, Mark Fassler
##  This software is licensed to you under the GPL version 3
##
#################################################################
#################################################################


import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pyrealsense2 as rs
import open3d

from rigid_transform import rigid_transform_3D

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


colormap =  np.int32(plt.cm.jet(np.linspace(0,1,256)) * 255)
def depth_to_color(d):
    dmin = 1.0  # if d is here, then ii should be 255.0
    dmax = 9.0  # if d is here, then ii should be 0.0

    m = -255.0 / (dmax - dmin);
    b = 255 - (m * dmin);

    ii = m*d + b;

    i = int(round(ii))
    if i < 0:
        i = 0
    elif i > 255:
        i = 255;

    # OpenCV is in BGR order
    return int(colormap[i][2]), int(colormap[i][1]), int(colormap[i][0])



lk_params = {
    'winSize': (15, 15),
    'maxLevel': 4,
    'criteria': (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
}

feature_params = {
    'maxCorners': 500,
    'qualityLevel': 0.04,   # originally 0.3
    'minDistance': 7,
    'blockSize': 7
}


track_len = 30
#track_len = 130
detect_interval = 3
frame_idx = 0;
tracks = []
new_tracks = []

cloud_points = []
cloud_colors = []
new_cloud_points = []
new_cloud_colors = []


notAddedYet = True
prev_gray = None


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
#config.enable_device_from_file(sys.argv[1], repeat_playback=False)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

rgb_stream = profile.get_stream(rs.stream.color)
rgb_stream_profile = rs.video_stream_profile(rgb_stream)
rgb_intrinsics = rgb_stream_profile.get_intrinsics()

w_minus_1 = rgb_intrinsics.width - 1
h_minus_1 = rgb_intrinsics.height - 1

align = rs.align(rs.stream.color)



vis = open3d.Visualizer()
vis.create_window(width=800, height=600, left=1100, right=50)  # "right" is "top"

pcd = open3d.PointCloud()
prev_pcd = open3d.PointCloud()


points = np.empty(1)
colors = np.empty(1)
prev_points = np.empty(1)
cur_points = np.empty(1)
prev_colors = np.empty(1)

def update_point_cloud():
    global points
    global colors
    global prev_points
    global cur_points
    global prev_colors

    numPts = 0
    for p in cloud_points:
        if len(p) > 1:
            numPts += 1

    prev_points = np.empty((numPts, 3))
    cur_points = np.empty((numPts, 3))
    prev_colors = np.empty((numPts, 3))
    cur_colors = np.empty((numPts, 3))
    for i, p in enumerate(cloud_points):
        if len(p) > 1:
            prev_points[i] = p[-2]
            cur_points[i] = p[-1]
            prev_colors[i] = 0.5, 0, 0
            cur_colors[i] = 0, 0.5, 0

    prev_pcd.points = open3d.Vector3dVector(prev_points)
    prev_pcd.colors = open3d.Vector3dVector(prev_colors)
    pcd.points = open3d.Vector3dVector(cur_points)
    pcd.colors = open3d.Vector3dVector(cur_colors)


#position = np.array((0,0,0,1))
position = np.array((0,0,0), np.float64)
direction = np.array((0,0,1), np.float64)

while True:

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("missing frame(s)")
        continue

    imRGB = np.asanyarray(color_frame.get_data())
    imD = np.asanyarray(depth_frame.get_data())


    # We use grayscale for calculations:
    frame_gray = cv.cvtColor(imRGB, cv.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = frame_gray.copy()

    if len(tracks):
        # The p0 vector is just the last point in each of the tracks items
        p0 = np.empty((len(tracks), 1, 2), np.float32)
        for i, t in enumerate(tracks):
            p0[i] = [t[-1]]

        # Forward tracking
        p1, _st, _err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

        # Reverse tracking
        p0r, _st, _err = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **lk_params)

        new_tracks = []
        new_cloud_points = []
        new_cloud_colors = []
        #d = abs(p0-p0r).reshape(-1, 2).max(-1)

        for i, p in enumerate(p0):
            x = p1[i][0][0]
            y = p1[i][0][1]
            xx = max(0, min(int(round(x)), w_minus_1))
            yy = max(0, min(int(round(y)), h_minus_1))
            d = cv.norm(p - p0r[i])  # TODO: perhaps this could be a single op in numpy?...
            z_depth = depth_scale * imD[yy, xx]
            if (d < 1.5) and (z_depth < 8.0) and (z_depth > 0.1):
                tracks[i].append( (x, y ) )
                new_tracks.append(tracks[i])
                z_color = depth_to_color(z_depth);
                color = imRGB[yy, xx] / 255
                cloud_colors[i].append( (float(color[2]), float(color[1]), float(color[0])) )
                #cloud_colors[i].append( (0.5, 0.5, 0.5) )
                cv.circle(imRGB, (x, y), 3, z_color, -1)
                new_cloud_colors.append(cloud_colors[i])
                pt3d = rs.rs2_deproject_pixel_to_point(rgb_intrinsics, [x,y], z_depth)

                #cloud_points[i].append(pt3d)
                cloud_points[i].append( (pt3d[0], -pt3d[1], -pt3d[2]) )
                new_cloud_points.append(cloud_points[i])

        tracks = new_tracks
        cloud_colors = new_cloud_colors
        cloud_points = new_cloud_points

        cv.polylines(imRGB, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        #for i, track in enumerate(tracks):
        #    cv.polylines(imRGB, np.int32(track), False, (0, 255, 0))

    # Every once-in-while, we'll try to add new points to the list of
    # points that we're tracking:
    if frame_idx % detect_interval == 0:

        # we won't bother detecting near points that we're already tracking:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for track in tracks:
            xy = track[-1]
            cv.circle(mask, xy, 5, 0, -1)

        pNew = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if pNew is not None:
            for x, y in np.float32(pNew).reshape(-1, 2):
                xx = max(0, min(int(round(x)), w_minus_1))
                yy = max(0, min(int(round(y)), h_minus_1))
                z_depth = depth_scale * imD[yy, xx]
                if z_depth < 8.0 and z_depth > 0.1:
                    tracks.append(collections.deque(maxlen=track_len))
                    cloud_points.append(collections.deque(maxlen=track_len))
                    cloud_colors.append(collections.deque(maxlen=track_len))
                    tracks[-1].append( (x, y) )
                    color = imRGB[yy, xx]
                    cloud_colors[-1].append( (color[2], color[1], color[0]) )
                    pt3d = rs.rs2_deproject_pixel_to_point(rgb_intrinsics, [x,y], z_depth)
                    cloud_points[-1].append( (pt3d[0], -pt3d[1], -pt3d[2]) )

    frame_idx += 1
    prev_gray = frame_gray

    update_point_cloud()

    if notAddedYet and len(pcd.points) > 50 and len(prev_pcd.points) > 50:
        vis.add_geometry(pcd)
        vis.add_geometry(prev_pcd)
        notAddedYet = False

    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    #reg_p2p = open3d.registration_icp(prev_pcd, pcd, 0.02, np.identity(4), open3d.TransformationEstimationPointToPoint())
    #position = np.dot(reg_p2p.transformation, position)

    if len(prev_points) > 10:
        R, tt = rigid_transform_3D(prev_points, cur_points)
        position = np.dot(R, position) + tt
        distance = cv.norm(position[:3])
        direction = np.dot(R, direction)
        #print(direction, position, "%.02f" % (distance))
        #print(R, tt)
        print(position[:3], "%.02f" % (distance))

    cv.imshow('lk_track', imRGB)
    cv.moveWindow('lk_track', 20, 20)
    cv.waitKey(1)




