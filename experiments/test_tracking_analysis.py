import copy
import unittest

from analyze_tracking import audit_rows, independently_grade
from tracking_task import Streams, grade, make_evaluation


class TrackingAnalysisTests(unittest.TestCase):
    def test_raw_action_grading_matches_exhaustive_reference(self):
        episode = make_evaluation(719, 1)['recovery'][0]
        actions = [0,1,2,3,4,5,6,3]*2
        self.assertEqual(independently_grade(episode, actions), grade(episode, actions))
        with self.assertRaises(AssertionError): independently_grade(episode, actions[:-1])

    def test_stream_and_label_corruption_are_rejected(self):
        stream = Streams(18, 2)
        rows = [{**row,'action':row['target']} for row in stream.current()]
        audit_rows(rows,Streams(18,2),'expert')
        for key in ('e','u','age','target'):
            bad = copy.deepcopy(rows)
            bad[0][key] += 1
            with self.assertRaises(AssertionError): audit_rows(bad,Streams(18,2),'expert')


if __name__ == '__main__': unittest.main()
