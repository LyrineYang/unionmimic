"""Differentiable URDF forward kinematics with explicit joint-name ordering."""
import math,xml.etree.ElementTree as ET
import torch
from unionmimic.motions import qmul,qrot

def rpy_quat(r,p,y):
    cr,sr=math.cos(r/2),math.sin(r/2);cp,sp=math.cos(p/2),math.sin(p/2);cy,sy=math.cos(y/2),math.sin(y/2)
    return [cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]

class HandKinematics:
    def __init__(self,path,joint_names,device='cpu'):
        self.device=device;self.names=list(joint_names);xml=ET.parse(path).getroot();self.nodes=[]
        parents={j.find('child').get('link') for j in xml.findall('joint')}
        roots=[x.get('name') for x in xml.findall('link') if x.get('name') not in parents]
        assert len(roots)==1;self.root=roots[0];remaining=list(xml.findall('joint'));known={self.root}
        def vec(s):return [float(x) for x in s.split()]
        while remaining:
            added=[]
            for j in remaining:
                par=j.find('parent').get('link');child=j.find('child').get('link')
                if par not in known:continue
                o=j.find('origin');xyz=vec(o.get('xyz','0 0 0')) if o is not None else [0,0,0]
                rpy=vec(o.get('rpy','0 0 0')) if o is not None else [0,0,0]
                typ=j.get('type');idx=self.names.index(j.get('name')) if typ in ('revolute','continuous') else None
                axis=vec(j.find('axis').get('xyz','1 0 0')) if idx is not None else [0,0,1]
                self.nodes.append((par,child,torch.tensor(xyz,device=device),torch.tensor(rpy_quat(*rpy),device=device),
                    idx,torch.nn.functional.normalize(torch.tensor(axis,dtype=torch.float32,device=device),dim=0)))
                known.add(child);added.append(j)
            if not added:raise ValueError('Disconnected or cyclic URDF')
            remaining=[x for x in remaining if x not in added]
        assert sum(x[4] is not None for x in self.nodes)==len(self.names)

    def forward(self,q,names=None):
        n=len(q);identity=torch.zeros(n,4,device=q.device,dtype=q.dtype);identity[:,0]=1
        positions={self.root:torch.zeros(n,3,device=q.device,dtype=q.dtype)};rotations={self.root:identity}
        for parent,child,p,r,idx,axis in self.nodes:
            pos=positions[parent]+qrot(rotations[parent],p.expand(n,-1))
            local=r.expand(n,-1)
            if idx is not None:
                half=q[:,idx:idx+1]/2;local=qmul(local,torch.cat((half.cos(),half.sin()*axis),-1))
            positions[child]=pos;rotations[child]=qmul(rotations[parent],local)
        if names is None:return positions,rotations
        return torch.stack([positions[k] for k in names],1)
