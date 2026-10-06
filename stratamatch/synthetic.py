"""Seeded, explicit truth fixtures. Truth is never supplied to inference."""
import numpy as np
from PIL import Image,ImageDraw
from .geometry import extract,topology
from .io import write,local

CASES=['continuous','split','merge','pinch_out','same_lith_unrelated','near_different_shape','similar_wrong_position','neighbor_change','ambiguous','local_anomaly','lithology_mismatch','unverified_registration']
COLORS={'shale':(146,185,210),'sandstone':(236,193,96),'limestone':(159,203,151),'UNKNOWN':(177,177,177)}

def band(top,bottom,left=4,right=156):
    y,x=np.mgrid[:120,:160]
    return (x>=left)&(x<right)&(y>=np.asarray(top))&(y<np.asarray(bottom))

def section(name,station,spec,complete=True):
    layers=[]; masks=[]
    for suffix,lith,mask,q in spec:
        layers.append(extract(mask,name+'_'+suffix,lith,[0,0,160,120],q)); masks.append(mask)
    topology(layers,masks,3)
    return {'id':name,'frame':'synthetic_common_depth','units':'m','station':station,'bounds':[0,0,160,120],
        'registration_verified':True,'complete':complete,'layers':sorted(layers,key=lambda a:a['order'])},masks

def make_case(kind,seed=1001):
    rng=np.random.default_rng(seed); noise=int(rng.integers(-2,3)); x=np.arange(160)[None,:]
    top=38+noise+np.round(1.5*np.sin(x/25)); bot=top+18
    a=[('U','shale',band(10,top),1.),('M','sandstone',band(top,bot),1.),('D','limestone',band(bot,98),1.)]
    b=[(i,l,np.roll(m,1,axis=0),q) for i,l,m,q in a]
    truth=[{'sources':['A_'+i],'targets':['B_'+i],'relation':'continuous'} for i,_,_,_ in a]
    uncertain=[]
    if kind in ['split','merge']:
        parts=[('M1','sandstone',band(top,bot,4,79),1.),('M2','sandstone',band(top,bot,81,156),1.)]
        split=[a[0],*parts,a[2]]
        if kind=='split':
            b=split; truth[1]={'sources':['A_M'],'targets':['B_M1','B_M2'],'relation':'split'}
        else:
            a=split; b=[('U','shale',band(10,top),1.),('M','sandstone',band(top,bot),1.),('D','limestone',band(bot,98),1.)]
            truth[1]={'sources':['A_M1','A_M2'],'targets':['B_M'],'relation':'merge'}
    elif kind=='pinch_out':
        thick=np.maximum(1,np.round(12*np.sin(np.linspace(0,np.pi,160))**2))[None,:]
        a=[('U','shale',band(10,44),1.),('M','sandstone',band(44,44+thick),1.),('D','limestone',band(44+thick,98),1.)]
        b=[('U','shale',band(10,44),1.),('D','limestone',band(44,98),1.)]
        truth[1]={'sources':['A_M'],'targets':[],'relation':'pinch-out'}
    elif kind in ['same_lith_unrelated','similar_wrong_position']:
        a=[('M','sandstone',band(14,28),1.)]; b=[('X','sandstone',band(88,102),1.)]; truth=[]; uncertain=['A_M']
    elif kind=='near_different_shape':
        a=[('M','sandstone',band(48,60),1.)]
        wave=np.round(25*np.sin(x/18))
        b=[('X','sandstone',band(48+wave,60+wave),1.)]; truth=[]; uncertain=['A_M']
    elif kind=='neighbor_change':
        b=[('U','limestone',band(10,top),1.),('M','sandstone',band(top,bot),1.),('D','shale',band(bot,98),1.)]
        truth=[]; uncertain=['A_U','A_M','A_D']
    elif kind=='ambiguous':
        a=[('M','sandstone',band(48,60,45,115),1.)]
        b=[('X','sandstone',band(47,59,30,100),1.),('Y','sandstone',band(61,73,60,130),1.)]
        truth=[]; uncertain=['A_M']
    elif kind=='local_anomaly':
        a[1]=('M','sandstone',a[1][2],.45); b[1]=('M','sandstone',b[1][2],.5)
        uncertain=['A_M'] # correspondence truth exists, but evidence quality must force abstention
    elif kind=='lithology_mismatch':
        a=[('M','sandstone',band(48,60),1.)]; b=[('X','shale',band(48,60),1.)]; truth=[]; uncertain=['A_M']
    sa,ma=section('A',0,a); sb,mb=section('B',100,b)
    if kind=='unverified_registration': sb['registration_verified']=False; uncertain=['A_U','A_M','A_D']
    return {'name':kind,'seed':seed,'source':sa,'target':sb,'truth':truth,'must_abstain':uncertain},(a,b)

def render_case(case,spec,out):
    out=local(out); out.mkdir(parents=True,exist_ok=True)
    configs=[]
    for s,layers in zip([case['source'],case['target']],spec):
        im=Image.new('RGB',(225,125),'white'); arr=np.array(im)
        for _,l,mask,_ in layers: arr[:120,:160][mask]=COLORS[l]
        im=Image.fromarray(arr); draw=ImageDraw.Draw(im); legend=[]
        for j,(l,color) in enumerate(COLORS.items()):
            yy=8+j*27; draw.rectangle([169,yy,182,yy+13],fill=color); draw.text((186,yy),l[:5],fill='black')
            legend.append({'lithology':l,'swatch':[171,yy+2,9,9],'verified':l!='UNKNOWN'})
        path=out/(s['id']+'.png'); im.save(path)
        cfg={k:s[k] for k in ['id','frame','units','station','bounds','registration_verified','complete']}
        cfg.update(image=str(path.relative_to(local('.'))),roi=[0,0,160,120],legend=legend,minimum_complete_coverage=.7)
        configs.append(cfg)
    write(out/'manifest.json',{'sections':configs})

def generate(out='data/synthetic'):
    cases=[]
    for kind in CASES:
        for seed in [1001,1002,1003]:
            c,spec=make_case(kind,seed); cases.append(c)
            write(f'{out}/{kind}_{seed}.json',c)
            if seed==1001: render_case(c,spec,f'{out}/images/{kind}')
    write(f'{out}/index.json',{'cases':[f'{out}/{c["name"]}_{c["seed"]}.json' for c in cases],
        'partition':'held_out_test','seeds':[1001,1002,1003],'truth_origin':'procedural_known_masks_not_real_geology'})
    return cases
