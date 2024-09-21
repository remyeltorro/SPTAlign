import unittest
from spt_align.io import locate_frames, is_float, load_tracks, estimate_timeline
import os
import numpy as np
import pandas as pd

TEST_IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), os.sep.join(['assets','image_mini']))
TEST_TRACK_FILE = os.path.join(os.path.dirname(__file__), os.sep.join(['assets','tracks_mini.csv']))

class TestLocateFrames(unittest.TestCase):

	def test_nbr_frames(self):
		frames = locate_frames(TEST_IMAGE_FOLDER)
		self.assertEqual(len(frames),3)

	def test_endswith_sep(self):
		frames = locate_frames(TEST_IMAGE_FOLDER+os.sep)
		self.assertEqual(len(frames),3)

	def test_loading(self):
		frames = locate_frames(TEST_IMAGE_FOLDER, load=True)
		self.assertIsInstance(frames[0], np.ndarray)	


class TestIsFloat(unittest.TestCase):

	def test_float_str_is_float(self):
		test = is_float('3.0')
		self.assertTrue(test)

	def test_int_str_is_float(self):
		test = is_float('3')
		self.assertTrue(test)

	def test_float_is_float(self):
		test = is_float(2.0)
		self.assertTrue(test)

	def test_integer_is_float(self):
		test = is_float(3)
		self.assertTrue(test)

	def test_str_is_float(self):
		test = is_float('whatever')
		self.assertFalse(test)

class TestLoadTracks(unittest.TestCase):

	def test_load_tracks(self):
		df = load_tracks(TEST_TRACK_FILE)
		self.assertIsInstance(df, pd.DataFrame)

class TestTimeline(unittest.TestCase):

	def test_same_timeline(self):

		frames = locate_frames(TEST_IMAGE_FOLDER)
		df = load_tracks(TEST_TRACK_FILE)
		timeline = estimate_timeline(frames, df)
		self.assertIsInstance(timeline, np.ndarray)

	def test_incomplete_timeline(self):

		frames = locate_frames(TEST_IMAGE_FOLDER)
		frames = frames[:2]

		df = load_tracks(TEST_TRACK_FILE)
		self.assertRaises(AssertionError, lambda: estimate_timeline(frames, df))

if __name__=="__main__":
	unittest.main()