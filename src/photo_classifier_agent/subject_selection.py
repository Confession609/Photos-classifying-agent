"""Label-blind photographic subject proposals and LightGBM ranking features.

No category, caption, filename text or reference annotation is an input to
proposal generation, feature extraction, or selection. References are used
separately by training/evaluation code. Full-frame is a real composition
option, not a fabricated successful object detection.
"""
from __future__ import annotations

import math
from typing import Any

FEATURE_VERSION = 'subject_selector_visual_v1'
PROPOSAL_CONFIG = dict(max_detections=8, nms_iou=0.85, min_area=0.002,
                       context_padding=0.08, max_context=4)


def valid_box(box: list[float]) -> list[float]:
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError('Expected four finite normalized xywh coordinates')
    x, y, w, h = map(float, box)
    if w <= 0 or h <= 0:
        raise ValueError('Non-positive box size')
    left, top = max(0., x), max(0., y)
    right, bottom = min(1., x+w), min(1., y+h)
    if right <= left or bottom <= top:
        raise ValueError('Box outside image')
    return [left, top, right-left, bottom-top]


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = valid_box(a)
    bx, by, bw, bh = valid_box(b)
    intersection = max(0., min(ax+aw, bx+bw)-max(ax,bx)) * max(0., min(ay+ah,by+bh)-max(ay,by))
    return intersection / max(aw*ah+bw*bh-intersection, 1e-12)


def pixel_box(box: list[float], size: tuple[int, int]) -> list[int]:
    x,y,w,h = valid_box(box)
    width,height = size
    if width < 1 or height < 1:
        raise ValueError('Invalid image dimensions')
    left,top = min(width-1,math.floor(x*width)),min(height-1,math.floor(y*height))
    return [left,top,max(left+1,min(width,math.ceil((x+w)*width))),
            max(top+1,min(height,math.ceil((y+h)*height)))]


def make_candidates(boxes: list[list[float]], scores: list[float],
                    config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = dict(PROPOSAL_CONFIG if config is None else config)
    if len(boxes) != len(scores) or not boxes:
        raise ValueError('Non-empty boxes and scores must have matching lengths')
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Non-finite detector score')
    proposals: list[dict[str, Any]] = []
    for index in sorted(range(len(scores)), key=lambda k:(-scores[k],k)):
        try:
            box = valid_box(boxes[index])
        except ValueError:
            continue
        # Keep the valid top-1 even when tiny, to retain an honest baseline.
        if proposals and box[2]*box[3] < config['min_area']:
            continue
        if any(iou(box,p['box_xywh_norm']) >= config['nms_iou'] for p in proposals):
            continue
        proposals.append(dict(candidate_id=f'gd_{index}',kind='gd',query_index=index,
                              box_xywh_norm=box,detector_score=float(scores[index])))
        if len(proposals) >= config['max_detections']:
            break
    if not proposals:
        raise ValueError('Detector produced no valid baseline box')
    contexts = []
    for p in proposals[:config['max_context']]:
        x,y,w,h = p['box_xywh_norm']
        padding = config['context_padding']
        box = valid_box([x-w*padding,y-h*padding,w*(1+2*padding),h*(1+2*padding)])
        if any(iou(box,q['box_xywh_norm']) > .98 for q in proposals+contexts):
            continue
        contexts.append(dict(candidate_id=p['candidate_id']+'_context',kind='context',
                             query_index=p['query_index'],box_xywh_norm=box,
                             detector_score=p['detector_score']))
    return proposals+contexts+[dict(candidate_id='full_frame',kind='full_frame',
                                   query_index=None,box_xywh_norm=[0.,0.,1.,1.],detector_score=0.)]


def geometry_features(candidate: dict[str, Any]) -> list[float]:
    x,y,w,h = valid_box(candidate['box_xywh_norm'])
    cx,cy = x+w/2,y+h/2
    thirds = min(math.hypot(cx-a,cy-b) for a in (1/3,2/3) for b in (1/3,2/3))
    return [x,y,w,h,w*h,cx,cy,math.hypot(cx-.5,cy-.5),thirds,
            math.log(max(w/h,1e-8)),min(x,y,1-x-w,1-y-h),
            float(candidate['detector_score']),float(candidate['kind']=='full_frame'),
            float(candidate['kind']=='context')]


def photographic_features(image: Any, box: list[float]) -> list[float]:
    """Cheap exposure, saturation and sharpness contrasts, without labels."""
    import numpy as np
    rgb = np.asarray(image.resize((256,256)).convert('RGB'),dtype=np.float32)/255
    gray = rgb.mean(axis=2)
    edge = np.zeros_like(gray)
    edge[:,1:] += np.abs(gray[:,1:]-gray[:,:-1])
    edge[1:,:] += np.abs(gray[1:,:]-gray[:-1,:])
    saturation = (rgb.max(axis=2)-rgb.min(axis=2))/np.maximum(rgb.max(axis=2),1e-6)
    l,t,r,b = pixel_box(box,(256,256))
    mask = np.zeros(gray.shape,dtype=bool)
    mask[t:b,l:r] = True
    values = []
    for field in (gray,saturation,edge):
        inside = float(field[mask].mean())
        outside = float(field[~mask].mean()) if (~mask).any() else inside
        values.extend([inside,outside,inside-outside])
    return values


def visual_rank_features(original: Any, crop: Any, candidate: dict[str, Any],
                         photography: list[float]) -> Any:
    import numpy as np
    a,b = np.asarray(original,dtype=np.float32),np.asarray(crop,dtype=np.float32)
    if a.ndim != 1 or a.shape != b.shape or len(photography) != 9:
        raise ValueError('Feature shape mismatch')
    return np.concatenate((a,b,np.abs(a-b),np.array(geometry_features(candidate)+photography+[float(a@b)],dtype=np.float32)))


def choose_candidate(candidates: list[dict[str, Any]], scores: list[float],
                     min_margin: float = .1) -> dict[str, Any]:
    if not candidates or len(candidates)!=len(scores) or min_margin < 0:
        raise ValueError('Invalid selector inputs')
    if not all(math.isfinite(float(v)) for v in scores):
        raise ValueError('Non-finite ranker scores')
    global_index = next(i for i,c in enumerate(candidates) if c['kind']=='full_frame')
    order = sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    margin = float(scores[order[0]]-scores[order[1]]) if len(order)>1 else 0.
    uncertain = margin < min_margin
    selected = global_index if uncertain else order[0]
    return dict(selected_index=selected,raw_selected_index=order[0],
                rank_margin_uncalibrated=margin,needs_review=uncertain,
                full_frame_fallback=uncertain,
                reason='ambiguous_ranking_preserve_composition' if uncertain else 'highest_learned_rank')
