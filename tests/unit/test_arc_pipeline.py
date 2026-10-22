"""ARC boundary contracts: source IDs, sparse caches and ambiguous numbers."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
import supervision as sv
from handball_cv.tracking.arc import ArcDriver, ArcTracker, number_box_pairs, require_dense_cache, load_arc_cache
from handball_cv.tracking.geometry_cache import GeometryCache
from scripts.process_video import validate_tracker_resume
from scripts.render_full_pipeline import parse_args


def test_dense_cache_allows_genuinely_empty_frames_but_refuses_sparse_or_unverified():
    dense={'stride':1,'scored_frames':3,'total_frames':3,'offsets':np.zeros(4,int)}
    require_dense_cache(dense,3)
    for field,value in [('stride',5),('scored_frames',1),('total_frames',2)]:
        with pytest.raises(ValueError,match='stride-1'):require_dense_cache({**dense,field:value},3)
    with pytest.raises(ValueError):require_dense_cache({'offsets':np.zeros(4,int)},3)


def test_numbers_abstain_when_an_unconfirmed_raw_player_also_owns_crop():
    player=[[0,0,100,100]];number=[[40,30,55,50]]
    assert number_box_pairs(player,player,number)==[(0,0)]
    assert number_box_pairs(player,player+[[35,20,90,90]],number)==[]
    assert number_box_pairs(player+player,player,number)==[]
    assert number_box_pairs([],player,number)==[]
    assert number_box_pairs(player,player,[[120,120,130,130]])==[]


def test_no_mask_is_stored_as_absent_not_a_fake_player_silhouette(tmp_path):
    path=tmp_path/'geometry.npz';g=GeometryCache(80,60,2)
    g.add(0,[7],[[1,2,10,20]],[None],[0],[0]);g.add(1,[],[],[],[],[]);g.save(path)
    restored=GeometryCache.load(path)
    assert restored.frame_indices==[0,1]
    assert restored.rows(0)[0].tolist()==[7]
    assert not restored.row_mask(0).any()
    assert restored.frame(1).masks.shape==(0,60,80)


def test_resume_cannot_silently_change_tracker_or_arc_model(tmp_path):
    (tmp_path/'track.json').write_text(json.dumps({'tracker':'sam2'}))
    validate_tracker_resume(tmp_path,'sam2',None)
    for force in [None,'bundle']:
        with pytest.raises(ValueError,match='different tracker'):validate_tracker_resume(tmp_path,'arc',{},force=force)
    validate_tracker_resume(tmp_path,'arc',{},force='track')
    (tmp_path/'track.json').write_text(json.dumps({'tracker':'arc','arc':{'weight':'a'}}))
    with pytest.raises(ValueError):validate_tracker_resume(tmp_path,'arc',{'weight':'b'})


def test_sam2_remains_default_and_arc_is_explicit():
    common=['video.mp4','--detections','d.npz','--team-model','t.pkl','--output','out.mp4']
    assert parse_args(common).tracker=='sam2'
    arc=parse_args(common+['--tracker','arc'])
    assert arc.tracker=='arc' and arc.arc_device=='cpu'


def test_pause_reset_never_reuses_an_expired_local_id():
    adapter=ArcTracker.__new__(ArcTracker)
    adapter.highest_id=37;adapter.offset=0
    adapter.native=SimpleNamespace(tracks=[])
    adapter._new_native=lambda:SimpleNamespace(tracks=[])
    adapter.reset()
    assert adapter.offset==37


def test_driver_keeps_source_local_id_separate_and_retires_before_new_identity(monkeypatch,tmp_path):
    import handball_cv.video as video
    class Frames:
        def __init__(self,source):pass
        def __len__(self):return 3
        def rgb(self,index):return np.zeros((30,40,3),np.uint8)
    monkeypatch.setattr(video,'VideoFrames',Frames)
    class Tracker:
        alive_ids=[101]
        def update(self,det,rgb):return sv.Detections(xyxy=np.array([[1,2,10,20.]]),tracker_id=np.array([101]),class_id=np.array([2]))
        def reset(self):self.alive_ids=[]
    class Identity:
        registry=SimpleNamespace()
        def __init__(self):self.calls=[]
        def retire_missing(self,f,ids):self.calls.append(('retire',f,list(ids)))
        def update(self,f,rgb,tracked):
            assert self.calls[-1][0]=='retire'
            return np.array([7])
    identity=Identity();driver=ArcDriver(Tracker(),identity)
    frames=list(driver.frames('video',lambda f:sv.Detections.empty(),lambda f:f!=1))
    assert frames[0].player_ids.tolist()==[7] and frames[1].player_ids.tolist()==[]
    assert frames[0].masks==[None]
    path=tmp_path/'local.npz';driver.save_local_tracks(path)
    with np.load(path) as data:
        assert data['local_track_id'].tolist()==[101,101]
        assert data['identity_id'].tolist()==[7,7]
        assert data['frame'].tolist()==[0,2]
    assert driver.events==[{'frame':1,'type':'pause'},{'frame':2,'type':'resume'}]


def test_dense_file_metadata_survives_loading(tmp_path):
    path=tmp_path/'dense.npz'
    np.savez(path, stride=1, scored_frames=2, total_frames=2, offsets=[0,0,0],
             boxes=np.empty((0,4)),confidence=[],class_id=[])
    loaded=load_arc_cache(path,2)
    assert int(loaded['scored_frames'])==2
    np.savez(path, offsets=[0,0,0],boxes=np.empty((0,4)),confidence=[],class_id=[])
    with pytest.raises(ValueError):load_arc_cache(path,2)


@pytest.mark.parametrize('tracker,expected', [('sam2',[0,5]),('arc',list(range(6)))])
def test_real_scan_cadence_is_every_frame_only_for_arc(tmp_path,monkeypatch,tracker,expected):
    from scripts import process_video as process
    from handball_cv.detection import rfdetr_torchscript
    from handball_cv.court import rfdetr_keypoints
    from handball_cv.teams.model import TeamModel
    from handball_cv.tracking import arc
    class StopAfterScan(Exception):pass
    calls=[]
    class Frames:
        def __init__(self,*args,**kwargs):pass
        def __len__(self):return 6
        def rgb(self,index):return np.full((64,64,3),index,np.uint8)
    class Detector:
        def __init__(self,*args,**kwargs):pass
        def infer(self,rgb,threshold):
            calls.append(int(rgb[0,0,0]))
            return np.array([[1,1,30,50.]]),np.array([.9]),np.array([2])
    class Court:
        def infer(self,frame):return np.zeros((37,3)),None
    def stop(*args,**kwargs):raise StopAfterScan
    source=tmp_path/'source.mp4';source.write_bytes(b'not decoded by this fixture')
    work=tmp_path/'work';out=tmp_path/'bundle'
    monkeypatch.setattr(process,'VideoFrames',Frames)
    monkeypatch.setattr(process,'prepare',lambda *a:{'source':str(source),'video':str(source),
        'frames':6,'fps':25.,'width':64,'height':64,'source_fps':25.})
    monkeypatch.setattr(rfdetr_torchscript,'TorchScriptDetector',Detector)
    monkeypatch.setattr(rfdetr_keypoints,'load_court_keypoints',lambda *a,**kw:Court())
    monkeypatch.setattr(TeamModel,'fit_from_detection_cache',stop)
    monkeypatch.setattr(arc,'assets',lambda *a:{})
    monkeypatch.setattr(arc,'import_arc',lambda *a:None)
    monkeypatch.setattr('sys.argv',['process_video',str(source),'--tracker',tracker,
                                   '--device','cpu','--work',str(work),'--out',str(out)])
    with pytest.raises(StopAfterScan):process.main()
    assert calls==expected
    with np.load(work/'detections.npz') as data:
        assert int(data['scored_frames'])==len(expected)
        assert int(data['stride'])==(1 if tracker=='arc' else 5)
