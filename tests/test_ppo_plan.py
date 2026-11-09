import unittest
from unionmimic.ppo_plan import validate_plan


class PPOPlanTests(unittest.TestCase):
    def plan(self):
        return dict(approved_by_user=True, recipe={'envs':4096}, run='work/runs/new', seed=3,
                    hours=2, scope='quick_sm_ppo_comparison', authorization_deadline_unix=9000,
                    automatic_resume=False)

    def test_scoped_short_budget(self):
        self.assertEqual(validate_plan(self.plan(), {'envs':4096}, '/tmp/work/runs/new', 3, 100, '/tmp'),2)

    def test_rejects_wrong_run_seed_recipe_deadline_and_longer_budget(self):
        for field, value in [('run','old'),('seed',4),('recipe',{'envs':64}),('authorization_deadline_unix',7000),('hours',3),('approved_by_user',False),('automatic_resume',True)]:
            plan=self.plan();plan[field]=value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_plan(plan,{'envs':4096},'/tmp/work/runs/new',3,100,'/tmp')

    def test_original_long_budget_not_silently_relaxed(self):
        plan=self.plan();plan.pop('scope')
        with self.assertRaises(AssertionError):validate_plan(plan,{'envs':4096},'/tmp/work/runs/new',3,100,'/tmp')
