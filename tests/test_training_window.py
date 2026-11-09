import unittest
from unionmimic.training_window import remaining_seconds

class WindowTests(unittest.TestCase):
    def test_expired_authorization_never_renews(self):
        a=dict(user_authorized=True,started_unix=100,deadline_unix=86500)
        self.assertEqual(remaining_seconds(a,100),86400)
        with self.assertRaises(TimeoutError):remaining_seconds(a,86500)
        with self.assertRaises(TimeoutError):remaining_seconds(a,500000)
        self.assertEqual(a['deadline_unix'],86500)
    def test_invalid_or_unapproved_windows(self):
        for a in [dict(user_authorized=False,started_unix=0,deadline_unix=10),
                  dict(user_authorized=True,started_unix=0,deadline_unix=90000),
                  dict(user_authorized=True,started_unix=0,deadline_unix=float('inf'))]:
            with self.assertRaises((ValueError,PermissionError)):remaining_seconds(a,1)
if __name__=='__main__':unittest.main()
