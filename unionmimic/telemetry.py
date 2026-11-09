"""Local, credential-free metrics for unattended training and TensorBoard."""
import json
import math
from pathlib import Path
import subprocess
import time
from tensorboardX import SummaryWriter


def numeric_leaves(value, prefix=''):
    result={}
    for key,item in value.items():
        name=prefix+str(key)
        if isinstance(item,dict):
            result.update(numeric_leaves(item,name+'/'))
        elif isinstance(item,(int,float)) and math.isfinite(item):
            result[name]=float(item)
    return result


class TrainingMetrics:
    def __init__(self,run,config,resume_state=None):
        self.run=Path(run)
        self.writer=SummaryWriter(str(self.run/'tensorboard'),flush_secs=10)
        self.stream=open(self.run/'events.jsonl','a',buffering=1)
        self.started=time.time()
        self.count=0
        self.totals={'sm':0,'calibrate':0}
        if resume_state is not None:
            self.started=resume_state['started_unix']
            self.totals.update(resume_state.get('attempted_updates',{}))
            # New TensorBoard event files append to the same run/axes.
            with open(self.run/'events.jsonl') as previous:
                for line in previous:
                    try:
                        self.count=max(self.count,json.loads(line)['event'])
                    except (ValueError,KeyError):
                        continue
        self.phase_base=0
        self.first_update=None
        self.gpu_checked=0.
        self.writer.add_text('configuration',json.dumps(config,indent=2),0)

    def emit(self,phase,cycle,metrics,step=None):
        self.count+=1
        step=cycle if step is None else step
        metrics=numeric_leaves(metrics)
        now=time.time()
        row=dict(event=self.count,wall_time=now,elapsed_seconds=now-self.started,
                 phase=phase,cycle=cycle,step=step,metrics=metrics)
        self.stream.write(json.dumps(row,allow_nan=False)+'\n')
        for key,value in metrics.items():
            self.writer.add_scalar(phase+'/'+key,value,step,walltime=now)
        self.writer.flush()

    def begin_phase(self,phase,cycle):
        key='calibrate' if phase.startswith('calibrate') else phase
        self.phase_base=self.totals.get(key,0)
        self.first_update=None
        self.writer.add_text('runtime/phase','cycle %d: %s'%(cycle,phase),self.count)

    def ingest(self,row,phase,cycle):
        kind=row.get('phase','')
        if kind in ('sm','calibrate'):
            update=row.get('step',row.get('update'))
            if self.first_update is None:
                self.first_update=update-1
            total=self.phase_base+update-self.first_update
            self.totals[kind]=total
            self.emit(kind,cycle,{k:v for k,v in row.items() if k not in ('phase','step','update')},total)
        elif kind=='validation':
            self.emit('prediction',cycle,dict(passed=row['passed'],**row['horizons']),self.totals['calibrate'])
        elif kind in ('collect','evaluate','recover'):
            self.emit('progress/'+kind,cycle,{k:v for k,v in row.items() if k!='phase'},self.count)

    def sample_gpu(self,gpu,cycle):
        if time.monotonic()-self.gpu_checked<30:
            return
        self.gpu_checked=time.monotonic()
        try:
            result=subprocess.run(['nvidia-smi','--id='+str(gpu),
                '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
            values=[float(x.strip()) for x in result.stdout.strip().split(',')]
            self.emit('system',cycle,dict(zip(('gpu_utilization_percent','gpu_memory_mib',
                      'gpu_temperature_c','gpu_power_w'),values)),int(time.time()-self.started))
        except (OSError,ValueError,subprocess.SubprocessError):
            pass

    def close(self):
        self.writer.close()
        self.stream.close()
