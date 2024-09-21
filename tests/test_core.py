import unittest
from spt_align.io import locate_frames, load_tracks, estimate_timeline
from spt_align import estimate_displacement
import os
import numpy as np
import pandas as pd

TEST_IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), os.sep.join(['assets','image_mini']))
TEST_TRACK_FILE = os.path.join(os.path.dirname(__file__), os.sep.join(['assets','tracks_mini.csv']))

class TestEstimateDisplacement(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.fake_df = pd.DataFrame({
						'TRACK_ID': [1]*5 + [2]*5,  # Two tracks
						'FRAME': list(range(5)) * 2,  # 5 frames for each track
						'POSITION_X': [i for i in range(5)] + [i + 1 for i in range(5)],  # Shift of 1 unit in X
						'POSITION_Y': [i for i in range(5)] + [i + 1 for i in range(5)]   # Shift of 1 unit in Y
						})
		self.fake_timeline = np.arange(5)
		self.expected_displacement = np.array([[0,0],[1,1],[2,2],[3,3],[4,4]]).astype(float)


	def test_estimate_displacement(self):
		displacement = estimate_displacement(self.fake_df, self.fake_timeline, reference_time=0, nbr_tracks_threshold=0)
		self.assertTrue(np.allclose(displacement, self.expected_displacement))

	def test_not_enough_tracks(self):
		displacement = estimate_displacement(self.fake_df, self.fake_timeline, reference_time=0, nbr_tracks_threshold=3)
		self.assertTrue(np.all([v!=v for v in displacement.flatten()]))


if __name__=="__main__":
	unittest.main()