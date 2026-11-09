#!/usr/bin/env python3
"""Summarize completed screening without selecting on the locked test or launching."""
import argparse
import json
from pathlib import Path
from screening_summary import summarize, ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',default='config/sm-screen-20261103.plan.json')
    ap.add_argument('--output',default='work/analysis/sm-screen-20261103/results.json');a=ap.parse_args()
    p=json.loads(Path(a.plan).read_text());s=summarize(a.plan)
    output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True)
    if s['complete'] and s['engineering_passed']:
        fields=['mean_return','mean_rotation','failure_rate','mean_axis_deviation']
        effects={}
        for factor,high in [('failure_fraction',.25),('critic_norm_source','real'),('utd',.4),('horizon',16)]:
            hi=[name for name,arm in p['arms'].items() if arm[factor]==high]
            lo=[name for name in p['arms'] if name not in hi]
            effects[factor]={metric:sum(s['arms'][name][metric] for name in hi)/4-
                            sum(s['arms'][name][metric] for name in lo)/4 for metric in fields}
        s['main_effect_high_minus_low']=effects
        s['interaction_aliases']=p['alias_pairs']
        s['statistical_claim']='Single seed, local checkpoint screening; effects may include higher-order interactions; no significance claim.'
        s['baseline_differences']={name:{k:row[k]-s['arms']['S0'][k] for k in fields} for name,row in s['arms'].items()}
        s['recommendation_status']='Requires review of time courses and precision before formal confirmation; no automatic long training.'
    output.write_text(json.dumps(s,indent=2)+'\n')
    rows=['# 联合筛选结果','', '完整：%s；工程验收：%s'%(s['complete'],s['engineering_passed']),'',
          '|组|状态|新增轮数|平均开发回报|旋转rad|失败率|轴偏差rad|GPU小时|',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,r in sorted(s['arms'].items()):
        if r['status']!='completed':rows.append('|%s|%s|||||||'%(name,r['status']));continue
        rows.append('|%s|完成|%d|%.2f|%.2f|%.2f%%|%.4f|%.3f|'%(name,r['cycles'],r['mean_return'],r['mean_rotation'],r['failure_rate']*100,r['mean_axis_deviation'],r['actual_gpu_hours']))
    rows+=['','两个固定时点重复相同开发种子，不是独立64回合测试。共同父训练与工程试跑另计。未使用最终测试集，也未启动长训。']
    if s.get('main_effect_high_minus_low'):
        rows+=['','## 高设置减低设置的平均差','', '|因素|回报|旋转rad|失败率百分点|轴偏差rad|','|---|---:|---:|---:|---:|']
        for f,r in effects.items():rows.append('|%s|%+.2f|%+.2f|%+.2f|%+.4f|'%(f,r['mean_return'],r['mean_rotation'],r['failure_rate']*100,r['mean_axis_deviation']))
        rows+=['','这是单训练种子的局部筛选；两因素交互成对混淆，不声明统计显著或全局最优。']
    output.with_suffix('.md').write_text('\n'.join(rows)+'\n')
    print('\n'.join(rows))

if __name__=='__main__':main()
