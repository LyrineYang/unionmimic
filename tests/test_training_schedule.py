import datetime as dt
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

path=Path(__file__).resolve().parents[1]/'scripts/training_check_schedule.py'
spec=importlib.util.spec_from_file_location('training_check_schedule',path)
schedule=importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


class TrainingSchedule(unittest.TestCase):
    def test_two_hour_slots_include_one_final_check(self):
        now=dt.datetime(2026,10,31,20,44,tzinfo=dt.timezone.utc).timestamp()
        end=dt.datetime(2026,11,1,6,44,40,tzinfo=dt.timezone.utc).timestamp()
        slots=schedule.schedule_slots(now,end)
        self.assertEqual(len(slots),5)
        self.assertTrue(all(b-a==7200 for a,b in zip(slots,slots[1:])))
        self.assertGreaterEqual(slots[-1],end)
        self.assertLess(slots[-2],end)

    def test_refuses_to_stop_an_unrelated_unit(self):
        with mock.patch.object(schedule.subprocess,'run') as run:
            with self.assertRaises(AssertionError):
                schedule.stop_timer({'unit':'unrelated'})
            run.assert_not_called()


if __name__=='__main__':unittest.main()
