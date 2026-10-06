"""Resolve all paths inside the three-folder project layout.

Legacy logical prefixes (data/outputs/config/models/tools/logs/.tmp) remain
accepted so saved projects and older result snapshots continue to open after the
directory cleanup. New files are physically routed to 输入数据、输出数据或软件.
"""
from pathlib import Path
import json
import math

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOFTWARE_ROOT.parent
INPUT_ROOT = ROOT / '输入数据'
OUTPUT_ROOT = ROOT / '输出数据'

_PREFIXES = {
    'data': INPUT_ROOT,
    'outputs': OUTPUT_ROOT,
    'logs': OUTPUT_ROOT / '日志',
    '.tmp': OUTPUT_ROOT / '临时',
    'config': SOFTWARE_ROOT / 'config',
    'models': SOFTWARE_ROOT / 'models',
    'tools': SOFTWARE_ROOT / 'tools',
    'docs': SOFTWARE_ROOT / 'docs',
    'tests': SOFTWARE_ROOT / 'tests',
    'stratamatch': SOFTWARE_ROOT / 'stratamatch',
    '.venv': SOFTWARE_ROOT / '.venv',
}

def local(path):
    p = Path(path)
    if p.is_absolute():
        # Saved projects may contain absolute paths from the former flat layout.
        # Translate only paths that still point inside this project.
        try:legacy=p.relative_to(ROOT)
        except ValueError:legacy=None
        if legacy and legacy.parts and legacy.parts[0] in _PREFIXES:
            p=_PREFIXES[legacy.parts[0]].joinpath(*legacy.parts[1:])
    else:
        parts=p.parts
        p=(_PREFIXES[parts[0]].joinpath(*parts[1:]) if parts and parts[0] in _PREFIXES else ROOT/p)
    p=p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f'Path is outside project: {p}')
    return p

def read(path):
    return json.loads(local(path).read_text(encoding='utf-8-sig'))

def write(path, obj):
    p = local(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def validate_section(s):
    required = ['id', 'frame', 'units', 'station', 'bounds', 'layers']
    if any(k not in s for k in required):
        raise ValueError('Section missing required metadata')
    if not s['frame'] or not s['units']:
        raise ValueError('Common coordinate frame and units are required')
    if not isinstance(s['station'],(int,float)) or not math.isfinite(s['station']):
        raise ValueError('Station must be finite')
    b = s['bounds']
    if len(b) != 4 or not all(math.isfinite(v) for v in b) or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError('Invalid physical bounds')
    ids = [x['id'] for x in s['layers']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate layer id')
    orders=[x['order'] for x in s['layers']]
    if orders!=sorted(orders) or len(set(orders))!=len(orders):
        raise ValueError('Layers must be sorted with unique order indices')
    for x in s['layers']:
        for key in ['centroid','area','perimeter','extent','mean_thickness','thickness_variation','dip','upper_boundary','lower_boundary','outline','order','quality','upper_neighbors','lower_neighbors','other_contact_neighbors']:
            if key not in x:
                raise ValueError(f"{x['id']} missing {key}")
        if x['area'] <= 0 or x['mean_thickness'] <= 0 or not 0 <= x['quality'] <= 1:
            raise ValueError('Invalid geometry/quality')
        for key in ['upper_boundary','lower_boundary','outline']:
            if len(x[key]) < 8:
                raise ValueError('Boundary needs >=8 samples')
            if any(len(p)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) for p in x[key]):
                raise ValueError('Invalid boundary point')
        if not isinstance(x['lithology'],str): raise ValueError('Lithology must be a canonical string or UNKNOWN')
        for key in ['upper_neighbors','lower_neighbors','other_contact_neighbors']:
            if any(i not in ids or i == x['id'] for i in x[key]):
                raise ValueError('Invalid topology reference')
    # JSON round-trip rejects non-finite numeric features at the import boundary.
    json.dumps(s, allow_nan=False)
    return s
