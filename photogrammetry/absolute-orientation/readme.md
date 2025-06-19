## Problem Statement
given a set of image points taken from 2 cameras(preferably a stereo setup), we can compute the absolute orientation of the camera position and orientation.

### Solution
* from the stereo image pair, we take some points(features using SIFT?)
* we find the relative orientation of the cameras with the help of those points
* using triangulation, we find the 3D coordinates of those points in our camera coordinate system or the photogrammetry model
* from those points we take lets say 3/4 points and their coordinatees in the absolute(target) coordinate system. These will be our control points.
* now using these 3/4 points in both camera c.s. and absolute c.s., we can compute the absolute orientation of the camera position and orientation.

